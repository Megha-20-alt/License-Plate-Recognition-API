import os

# Limit CPU thread usage
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gc
import streamlit as st
from app.database import engine, Base, SessionLocal
from app.models import PlateDetection

Base.metadata.create_all(bind=engine)
import torch
import cv2
import numpy as np

from ultralytics import YOLO
import easyocr

from sqlalchemy.orm import Session



# --------------------------------------------------
# PyTorch CPU settings
# --------------------------------------------------

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗",
    layout="centered"
)


st.title("🚗 License Plate Recognition")
st.write(
    "Upload an image to detect license plates "
    "using YOLO and read them using EasyOCR."
)


# --------------------------------------------------
# Load YOLO only once
# --------------------------------------------------

@st.cache_resource
def load_yolo():

    model = YOLO("models/best.pt")

    return model


# --------------------------------------------------
# Load EasyOCR only once
# --------------------------------------------------

@st.cache_resource
def load_ocr():

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )

    return reader


# --------------------------------------------------
# Load models
# --------------------------------------------------

with st.spinner("Loading YOLO model..."):
    model = load_yolo()


# --------------------------------------------------
# Image uploader
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file is not None:

    # ----------------------------------------------
    # Read image
    # ----------------------------------------------

    contents = uploaded_file.read()

    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error("Invalid image.")

        st.stop()


    # ----------------------------------------------
    # Limit image size
    # ----------------------------------------------

    MAX_WIDTH = 1280

    height, width = image.shape[:2]

    if width > MAX_WIDTH:

        scale = MAX_WIDTH / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )


    # ----------------------------------------------
    # Display uploaded image
    # ----------------------------------------------

    display_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        display_image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ----------------------------------------------
    # YOLO detection
    # ----------------------------------------------

    with st.spinner("Detecting license plates..."):

        results = model.predict(
            source=image,
            conf=0.4,
            device="cpu",
            imgsz=640,
            verbose=False
        )[0]


    plates = []


    # ----------------------------------------------
    # Load OCR only when needed
    # ----------------------------------------------

    if len(results.boxes) > 0:

        with st.spinner("Reading license plates..."):

            reader = load_ocr()


            # --------------------------------------
            # Process detections
            # --------------------------------------

            for box in results.boxes:

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x1 = max(0, int(coordinates[0]))
                y1 = max(0, int(coordinates[1]))
                x2 = min(image.shape[1], int(coordinates[2]))
                y2 = min(image.shape[0], int(coordinates[3]))


                confidence = float(
                    box.conf[0]
                )


                # ----------------------------------
                # Crop license plate
                # ----------------------------------

                crop = image[
                    y1:y2,
                    x1:x2
                ]


                if crop.size == 0:
                    continue


                # ----------------------------------
                # Convert to grayscale
                # ----------------------------------

                gray = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2GRAY
                )


                # ----------------------------------
                # Smaller OCR enlargement
                # ----------------------------------

                gray = cv2.resize(
                    gray,
                    None,
                    fx=1.5,
                    fy=1.5,
                    interpolation=cv2.INTER_CUBIC
                )


                # ----------------------------------
                # OCR
                # ----------------------------------

                ocr_result = reader.readtext(
                    gray
                )


                text = ""


                if ocr_result:

                    text = " ".join(
                        result[1]
                        for result in ocr_result
                    )


                # ----------------------------------
                # Clean OCR result
                # ----------------------------------

                text = "".join(
                    c
                    for c in text
                    if c.isalnum() or c == " "
                )

                text = text.strip().upper()


                # ----------------------------------
                # Save to database
                # ----------------------------------

                db: Session = SessionLocal()

                try:

                    detection = PlateDetection(
                        plate_number=text,
                        detection_confidence=confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2
                    )

                    db.add(detection)
                    db.commit()

                finally:

                    db.close()


                # ----------------------------------
                # Store result
                # ----------------------------------

                plates.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "box": [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    }
                )


    # ----------------------------------------------
    # Display results
    # ----------------------------------------------

    st.subheader("Detection Results")


    if plates:

        for i, plate in enumerate(
            plates,
            start=1
        ):

            st.write(
                f"### Plate {i}"
            )

            st.write(
                f"**Plate Number:** "
                f"{plate['text'] or 'Not recognized'}"
            )

            st.write(
                f"**Confidence:** "
                f"{plate['confidence']:.2%}"
            )

            st.write(
                f"**Bounding Box:** "
                f"{plate['box']}"
            )

    else:

        st.warning(
            "No license plates detected."
        )


    # ----------------------------------------------
    # Cleanup temporary objects
    # ----------------------------------------------

    del image_array
    del image
    del contents

    gc.collect()