\# License Plate Recognition API



A backend License Plate Recognition system built with \*\*FastAPI, YOLO, EasyOCR, OpenCV, SQLAlchemy, and SQLite\*\*.



The API accepts vehicle images, detects license plates using a custom-trained YOLO model, extracts the plate text using EasyOCR, and stores the detection results in a SQLite database.



\## Features



\* License plate detection using YOLO

\* License plate text extraction using EasyOCR

\* Image processing using OpenCV

\* REST API built with FastAPI

\* Detection results stored in SQLite

\* SQLAlchemy database integration

\* API documentation through FastAPI/Swagger UI

\* CPU-based inference



\## Technologies Used



\* Python

\* FastAPI

\* Uvicorn

\* Ultralytics YOLO

\* EasyOCR

\* OpenCV

\* NumPy

\* SQLAlchemy

\* SQLite



\## Project Structure



```text

License-Plate-Recognition-API/

│

├── app/

│   ├── main.py

│   ├── database.py

│   └── models.py

│

├── models/

│   └── best.pt

│

├── requirements.txt

├── README.md

├── .gitignore

└── license\_plates.db

```



\## How It Works



The system follows this pipeline:



```text

Vehicle Image

&#x20;     ↓

&#x20;  FastAPI

&#x20;     ↓

&#x20;  YOLO Model

&#x20;     ↓

License Plate Detection

&#x20;     ↓

&#x20;  Crop Plate

&#x20;     ↓

&#x20;OpenCV Processing

&#x20;     ↓

&#x20;   EasyOCR

&#x20;     ↓

Extracted Plate Text

&#x20;     ↓

&#x20;  SQLite Database

```



\## API Endpoints



\### 1. Home



```http

GET /

```



Used to check whether the API is running.



Example response:



```json

{

&#x20;   "message": "License Plate Recognition API is running!"

}

```



\### 2. Predict License Plate



```http

POST /predict

```



Accepts an image file and performs license plate detection and OCR.



The response contains:



\* Detected plate text

\* Detection confidence

\* Bounding box coordinates



Example:



```json

{

&#x20;   "plates": \[

&#x20;       {

&#x20;           "text": "WB12AB1234",

&#x20;           "confidence": 0.95,

&#x20;           "box": \[120, 210, 350, 280]

&#x20;       }

&#x20;   ]

}

```



\### 3. Get Detection History



```http

GET /detections

```



Returns previously stored license plate detections from the SQLite database.



Example:



```json

{

&#x20;   "detections": \[

&#x20;       {

&#x20;           "id": 1,

&#x20;           "plate\_number": "WB12AB1234",

&#x20;           "confidence": 0.95,

&#x20;           "box": \[120, 210, 350, 280],

&#x20;           "created\_at": "2026-08-13T10:30:00"

&#x20;       }

&#x20;   ]

}

```



\## Running the Project Locally



\### 1. Clone the repository



```bash

git clone https://github.com/Megha-20-alt/License-Plate-Recognition-API.git

cd License-Plate-Recognition-API

```



\### 2. Create a virtual environment



Windows:



```powershell

python -m venv .venv

```



Activate it:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```powershell

pip install -r requirements.txt

```



\### 4. Start the FastAPI server



```powershell

python -m uvicorn app.main:app --reload

```



The API will be available at:



```text

http://127.0.0.1:8000

```



\## API Documentation



FastAPI automatically provides interactive API documentation.



Open:



```text

http://127.0.0.1:8000/docs

```



You can use Swagger UI to upload an image and test the `/predict` endpoint directly from your browser.



\## Model



The project uses a custom YOLO license plate detection model:



```text

models/best.pt

```



The model was evaluated on the validation dataset with:



\* mAP50: approximately \*\*0.966\*\*

\* mAP50-95: approximately \*\*0.728\*\*



\## Database



The project uses SQLite with SQLAlchemy.



Detection records contain:



\* ID

\* License plate number

\* Detection confidence

\* Bounding box coordinates

\* Detection timestamp



\## Future Improvements



Possible future additions include:



\* Deploying the API to a cloud platform

\* Adding authentication

\* Adding a frontend interface

\* Supporting video input

\* Improving OCR preprocessing

\* Adding API rate limiting

\* Containerizing the application with Docker



\## Project Goal



The primary goal of this project is to build and deploy a practical backend API that combines computer vision, OCR, and database storage into a complete License Plate Recognition system.



