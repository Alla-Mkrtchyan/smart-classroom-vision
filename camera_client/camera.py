import cv2
import time
import os
import requests

CACHE_DIR = "cached_frames"
SERVER_URL = "http://localhost:8000/api/v1/process-frame"

TIMEOUT = 5
JPEG_QUALITY = 80
SEND_INTERVAL = 1.0
MAX_CACHE_FILES = 20

LAST_ATTEMPT_TIME = 0

def frame_to_jpeg(frame):
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not success:
        print("Չհաջողվեց frame-ը JPEG դարձնել")
        return None
    return buffer.tobytes()

def send_to_server(file_path, camera_id="CAM001"):
    try:
        if not os.path.exists(file_path):
            return False

        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        files = {
            "file": ("frame.jpg", file_bytes, "image/jpeg")
        }
        data = {
            "camera_id": camera_id,
            "timestamp": str(time.time())
        }

        response = requests.post(
            SERVER_URL,
            files=files,
            data=data,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        os.remove(file_path)
        return True

    except requests.exceptions.RequestException:
        return False
#

def setup_camera():
    global LAST_ATTEMPT_TIME
    os.makedirs(CACHE_DIR, exist_ok=True)

    camera_index = 0
    print(f"Connecting to camera {camera_index}...")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ALERT: Could not open camera device!")
        return

    print('Camera successfully opened! Press "q" to exit.')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ALERT: Can't receive frame from camera. Retrying...")
                time.sleep(2)
                continue

            current_time = time.time()

            if current_time - LAST_ATTEMPT_TIME >= SEND_INTERVAL:
                image_bytes = frame_to_jpeg(frame)

                if image_bytes:
                    file_name = f"frame_{int(time.time() * 1000)}.jpg"
                    file_path = os.path.join(CACHE_DIR, file_name)

                    with open(file_path, 'wb') as f:
                        f.write(image_bytes)

                cached_files = sorted(os.listdir(CACHE_DIR))
                if len(cached_files) > MAX_CACHE_FILES:
                    oldest_file = os.path.join(CACHE_DIR, cached_files[0])
                    if os.path.exists(oldest_file):
                        os.remove(oldest_file)
                    cached_files = sorted(os.listdir(CACHE_DIR))

                if cached_files:
                    target_file = os.path.join(CACHE_DIR, cached_files[0])
                    if send_to_server(target_file):
                        print("[SERVER] Frame sent successfully and removed from disk!")
                    else:
                        print(f"[OFFLINE] Server offline. Saved files in folder: {len(cached_files)}/{MAX_CACHE_FILES}")

                LAST_ATTEMPT_TIME = current_time

            files_count = len(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else 0
            text_status = f"Folder Queue: {files_count}/{MAX_CACHE_FILES}"
            cv2.putText(frame, text_status, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Smart Classroom - Camera Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Closing stream gracefully...')
                break

    except Exception as e:
        print(f"ALERT: An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == '__main__':
    setup_camera()