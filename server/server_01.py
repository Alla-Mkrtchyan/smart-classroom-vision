import fastapi
from fastapi import HTTPException
import cv2
import numpy as np
from datetime import datetime
import time
import uvicorn

app = fastapi.FastAPI(
    title="Smart classroom vision -server",
    version= "1.0.0"
)

result_data = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def calculate_occupancy_status(count: int) -> str:
  if count == 0:
    return "empty"
  elif count <= 5:
    return "low"
  elif count <= 15:
    return "medium"
  else:
    return "high"


@app.get("/")
async def root():
    return {"message": "Smart classroom vision is running"}

@app.post("/api/v1/process-frame")
async def process_frame(file: fastapi.UploadFile = fastapi.File(...)):
    start_processing_time = time.time()

    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Not an image")

    try:
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image not found")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize = (40, 40)
        )
        counts = len(faces)
        processing_time_ms = round((time.time() - start_processing_time) * 1000, 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        frame_data = {
            "frame_id": len(result_data) + 1,
            "detected_faces": counts,
            "time": datetime.now().isoformat(),
            "is_empty": counts == 0,
            "occupancy_status": calculate_occupancy_status(counts),
            "processing_time_ms": processing_time_ms
        }
        result_data.append(frame_data)
        return frame_data

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

@app.get("/api/v1/analytics/raw-data")
async def get_raw_data():
  return {"total_frames_processed": len(result_data), "data": result_data}

if __name__ == "__main__":
  uvicorn.run("server_01:app", host="0.0.0.0", port=8000, reload=True)
