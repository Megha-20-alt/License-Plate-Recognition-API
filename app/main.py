
from fastapi import FastAPI, UploadFile, File, Depends
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


# Load YOLO model
model = YOLO("models/best.pt")


# Load EasyOCR
reader = easyocr.Reader(["en"], gpu=False)


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "License Plate Recognition API is running!"
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # Read uploaded image
    contents = await file.read()

    # Convert bytes to NumPy array
    image_array = np.frombuffer(contents, np.uint8)

    # Decode image
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {
            "error": "Invalid image"
        }


    # --------------------------------------------------
    # YOLO detection
    # --------------------------------------------------

    results = model(
        image,
        conf=0.4,
        verbose=False
    )[0]


    plates = []


    # --------------------------------------------------
    # Process detected plates
    # --------------------------------------------------

    for box in results.boxes:

        # Get bounding box
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


        # Confidence
        confidence = float(
            box.conf[0]
        )


        # --------------------------------------------------
        # Crop license plate
        # --------------------------------------------------

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue


        # --------------------------------------------------
        # Convert to grayscale
        # --------------------------------------------------

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )


        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )


        # --------------------------------------------------
        # OCR
        # --------------------------------------------------

        ocr_result = reader.readtext(gray)

        text = ""

        if ocr_result:

            text = " ".join(
                result[1]
                for result in ocr_result
            )


        # Keep only letters, numbers and spaces
        text = "".join(
            c for c in text
            if c.isalnum() or c == " "
        )

        text = text.strip().upper()


        # --------------------------------------------------
        # Save detection to database
        # --------------------------------------------------

        detection = PlateDetection(
            plate_number=text,
            detection_confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )

        db.add(detection)


        # --------------------------------------------------
        # Add to API response
        # --------------------------------------------------

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


    # Commit database changes
    db.commit()


    # --------------------------------------------------
    # Return response
    # --------------------------------------------------

    return {
        "plates": plates
    }
@app.get("/detections")
def get_detections(
    db: Session = Depends(get_db)
):
    detections = db.query(PlateDetection).all()

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

