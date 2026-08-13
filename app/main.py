
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


app = FastAPI(
    title="License Plate Recognition API",
    description="Detects license plates using YOLO and reads them using EasyOCR.",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


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
# Load YOLO model
# --------------------------------------------------

model = YOLO("models/best.pt")


# --------------------------------------------------
# Load EasyOCR
# --------------------------------------------------

reader = easyocr.Reader(["en"], gpu=False)


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {
            "error": "Invalid image"
        }


    # YOLO detection

    results = model(
        image,
        conf=0.4,
        verbose=False
    )[0]


    plates = []


    for box in results.boxes:

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


        confidence = float(
            box.conf[0]
        )


        # Crop plate

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue


        # Grayscale

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )


        # Resize

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )


        # OCR

        ocr_result = reader.readtext(gray)

        text = ""

        if ocr_result:

            text = " ".join(
                result[1]
                for result in ocr_result
            )


        # Clean OCR text

        text = "".join(
            c for c in text
            if c.isalnum() or c == " "
        )

        text = text.strip().upper()


        # Save to database

        detection = PlateDetection(
            plate_number=text,
            detection_confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )

        db.add(detection)


        # Add to response

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


    db.commit()


    return {
        "plates": plates
    }


# --------------------------------------------------
# Get previous detections
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

