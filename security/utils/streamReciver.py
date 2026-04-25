import cv2
import time
import threading
import queue
import os
import django

# ======================
# DJANGO SETUP (IMPORTANT)
# ======================
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "your_project.settings")
django.setup()

# ======================
# IMPORT MODEL FUNCTION
# ======================
from security.predicate import predict_video 

# ======================
# THREAD MANAGER
# ======================
class ThreadManager:
    def __init__(self):
        self.threads = []
        self.stop_event = threading.Event()

    def start(self, target):
        t = threading.Thread(target=target, daemon=True)
        self.threads.append(t)
        t.start()
        return t

    def stop_all(self):
        print("🧹 Stopping all threads...")
        self.stop_event.set()

        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)

        self.threads.clear()
        print("✅ All threads cleaned")


# ======================
# CONFIG
# ======================
STREAM_URL = "http://192.168.100.106:4747/video"
CLIP_DURATION = 5
SAVE_FOLDER = "clips"

os.makedirs(SAVE_FOLDER, exist_ok=True)

clip_queue = queue.Queue()
prediction_queue = queue.Queue()

manager = ThreadManager()
stop_event = manager.stop_event

latest_prediction = None


# ======================
# SAVE CLIP
# ======================
from datetime import datetime

clip_counter = 1  # global counter

def save_clip(frames, fps=20):
    global clip_counter

    if not frames:
        return None

    h, w, _ = frames[0].shape

    # ⏱️ get datetime
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H-%M")  # safe format

    # 📁 filename
    filename = f"{SAVE_FOLDER}/{clip_counter}_{timestamp}.mp4"

    clip_counter += 1  # increment

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    for f in frames:
        out.write(f)

    out.release()
    return filename


# ======================
# STREAM WORKER
# ======================
def stream_worker():
    global latest_prediction

    while not stop_event.is_set():
        print("🔌 Connecting to stream...")

        cap = cv2.VideoCapture(STREAM_URL)

        if not cap.isOpened():
            print("❌ Failed to connect. Retrying...")
            time.sleep(3)
            continue

        print("✅ Stream connected")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 20

        buffer = []
        start_time = time.time()

        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                print("⚠️ Stream lost...")
                break

            # رسم النتيجة
            if latest_prediction:
                text = f"{latest_prediction['label']} ({latest_prediction['score']:.2f}%)"
                color = (0, 0, 255) if latest_prediction['score'] > 50 else (0, 255, 0)

                cv2.putText(frame, text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("LIVE STREAM", frame)

            buffer.append(frame)

            # تسجيل clip
            if time.time() - start_time >= CLIP_DURATION:
                clip_path = save_clip(buffer, fps)

                if clip_path:
                    clip_queue.put(clip_path)
                    print(f"💾 Clip saved: {clip_path}")

                buffer = []
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        cap.release()

    cv2.destroyAllWindows()


# ======================
# MODEL WORKER (🔥 UPDATED)
# ======================
def model_worker():
    global latest_prediction

    while not stop_event.is_set():
        try:
            video_path = clip_queue.get(timeout=1)

            print(f"🧠 Running model on: {video_path}")

            results, scores = predict_video(video_path)

            if scores:
                best_score, start, end = max(scores, key=lambda x: x[0])

                latest_prediction = {
                    "label": "SHOPLIFTING" if best_score > 50 else "NORMAL",
                    "score": best_score,
                    "video": video_path
                }

                prediction_queue.put(latest_prediction)

        except:
            pass


# ======================
# LOGGER
# ======================
def prediction_logger():
    while not stop_event.is_set():
        try:
            pred = prediction_queue.get(timeout=1)
            print("🔥 FINAL RESULT:", pred)
        except:
            pass


# ======================
# MAIN
# ======================
def run():

    print("🚀 Starting AI Surveillance System...")

    # manager.stop_all()
    manager = ThreadManager()
    stop_event = manager.stop_event

    manager.start(stream_worker)
    manager.start(model_worker)
    manager.start(prediction_logger)

    try:
        while not stop_event.is_set():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        manager.stop_all()

    print("✅ Shutdown complete")

