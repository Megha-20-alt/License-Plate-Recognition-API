import gc
import os

# Limit PyTorch CPU threading.
# Render Free has very limited CPU resources.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
import easyocr
import cv2
import numpy as np

from sqlalchemy.orm import Session

from .database import get_db
from .models import PlateDetection


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="License Plate Recognition API",
    description="Detects license plates using YOLO and reads them using EasyOCR.",
    version="1.0.0"
)


# --------------------------------------------------
# Frontend
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# YOLO model
# --------------------------------------------------

model = YOLO("models/best.pt")


# --------------------------------------------------
# EasyOCR
# --------------------------------------------------

# Do NOT initialize EasyOCR at startup.
#
# This saves startup memory and can allow the web
# server to start before OCR is needed.

reader = None


def get_ocr_reader():
    """
    Create the EasyOCR reader only when OCR is
    actually required.
    """

    global reader

    if reader is None:

        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False
        )

    return reader


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():

    return FileResponse(
        "frontend/index.html"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # --------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------

    contents = await file.read()

    if not contents:

        return {
            "error": "Empty file"
        }


    # --------------------------------------------------
    # Convert bytes -> NumPy array
    # --------------------------------------------------

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )


    # --------------------------------------------------
    # Decode image
    # --------------------------------------------------

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    # We no longer need the original byte array.
    del image_array
    del contents


    if image is None:

        return {
            "error": "Invalid image"
        }


    # --------------------------------------------------
    # Limit input image resolution
    # --------------------------------------------------

    MAX_WIDTH = 1280

    height, width = image.shape[:2]

    if width > MAX_WIDTH:

        scale = MAX_WIDTH / width

        new_width = MAX_WIDTH
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )


    # --------------------------------------------------
    # YOLO inference
    # --------------------------------------------------

    with torch.inference_mode():

        results = model.predict(
            source=image,
            conf=0.4,
            device="cpu",
            imgsz=416,
            max_det=10,
            verbose=False
        )


    # Get first result only.
    result = results[0]

    plates = []


    # --------------------------------------------------
    # Process detected plates
    # --------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            # ------------------------------------------
            # Bounding box
            # ------------------------------------------

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1 = int(coordinates[0])
            y1 = int(coordinates[1])
            x2 = int(coordinates[2])
            y2 = int(coordinates[3])


            # ------------------------------------------
            # Keep coordinates inside image
            # ------------------------------------------

            x1 = max(0, min(x1, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0]))
            x2 = max(0, min(x2, image.shape[1]))
            y2 = max(0, min(y2, image.shape[0]))


            if x2 <= x1 or y2 <= y1:
                continue


            # ------------------------------------------
            # Confidence
            # ------------------------------------------

            confidence = float(
                box.conf[0].cpu().item()
            )


            # ------------------------------------------
            # Crop license plate
            # ------------------------------------------

            crop = image[
                y1:y2,
                x1:x2
            ]


            if crop.size == 0:
                continue


            # ------------------------------------------
            # Grayscale
            # ------------------------------------------

            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY
            )


            # ------------------------------------------
            # Smaller OCR enlargement
            # ------------------------------------------

            gray = cv2.resize(
                gray,
                None,
                fx=1.5,
                fy=1.5,
                interpolation=cv2.INTER_CUBIC
            )


            # ------------------------------------------
            # EasyOCR
            # ------------------------------------------

            ocr_reader = get_ocr_reader()

            ocr_result = ocr_reader.readtext(
                gray,
                detail=1,
                paragraph=False
            )


            # ------------------------------------------
            # Extract text
            # ------------------------------------------

            text = ""

            if ocr_result:

                text = " ".join(
                    result[1]
                    for result in ocr_result
                )


            # ------------------------------------------
            # Clean text
            # ------------------------------------------

            text = "".join(
                c for c in text
                if c.isalnum() or c == " "
            )

            text = text.strip().upper()


            # ------------------------------------------
            # Save database record
            # ------------------------------------------

            detection = PlateDetection(
                plate_number=text,
                detection_confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2
            )

            db.add(detection)


            # ------------------------------------------
            # API response
            # ------------------------------------------

            plates.append({
                "text": text,
                "confidence": confidence,
                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            })


            # ------------------------------------------
            # Release OCR temporary memory
            # ------------------------------------------

            del gray
            del crop


    # --------------------------------------------------
    # Commit database changes
    # --------------------------------------------------

    db.commit()


    # --------------------------------------------------
    # Release temporary YOLO objects
    # --------------------------------------------------

    del result
    del results
    del image


    # Ask Python to release unused objects.
    gc.collect()


    # --------------------------------------------------
    # Return response
    # --------------------------------------------------

    return {
        "plates": plates
    }


# --------------------------------------------------
# Detection history
# --------------------------------------------------

@app.get("/detections")
def get_detections(
    db: Session = Depends(get_db)
):

    detections = db.query(
        PlateDetection
    ).all()


    return {
        "detections": [

            {
                "id": detection.id,

                "plate_number": detection.plate_number,

                "confidence": detection.detection_confidence,

                "box": [
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2
                ],

                "created_at": detection.created_at
            }

            for detection in detections
        ]
    }