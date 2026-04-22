import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from django.conf import settings
from security.models import Shoplifting
from django.core.files import File

# =========================
# 1. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = "./testVidios"   # 👈 change this
usedModel =  os.path.join(settings.BASE_DIR,'security/resources/predictionModels/train3_2.pth')
# =========================
# 2. MODEL (same as training)
# =========================
class SimpleI3D(nn.Module):
    def __init__(self):
        super(SimpleI3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(128, 256, 3, padding=1)
        self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)


# =========================
# 3. LOAD MODEL
# =========================
model = SimpleI3D().to(device)
model.load_state_dict(torch.load(usedModel, map_location=device))
model.eval()


# =========================
# 4. FRAME PROCESSING
# =========================
def extract_clip(frames, start, end, size=(112, 112), num_frames=16):
    segment = frames[start:end]

    processed = []
    for f in segment:
        f = cv2.resize(f, size)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        processed.append(f)

    if len(processed) == 0:
        return None

    # fix length
    while len(processed) < num_frames:
        processed.append(processed[-1])

    processed = processed[:num_frames]

    clip = np.stack(processed) / 255.0
    clip = torch.tensor(clip, dtype=torch.float32)

    # (1, 1, T, H, W)
    clip = clip.unsqueeze(0).unsqueeze(0)

    return clip


# =========================
# 5. PREDICT SINGLE VIDEO
# =========================
def predict_video(video_path, chunk_seconds=2):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    chunk_size = int(fps * chunk_seconds)

    results = []
    shoplifting_scores = []

    for start in range(0, len(frames), chunk_size):
        end = min(start + chunk_size, len(frames))

        clip = extract_clip(frames, start, end)
        if clip is None:
            continue

        clip = clip.to(device)

        with torch.no_grad():
            output = model(clip)
            prob = torch.softmax(output, dim=1)

            shoplifting_prob = prob[0, 1].item() * 100  # 🔥 percentage

        start_sec = start / fps
        end_sec = end / fps

        print("===================================")
        print(f"Time: {start_sec:.2f}s → {end_sec:.2f}s")
        print(f"Shoplifting risk: {shoplifting_prob:.2f}%")
        print("===================================")

        results.append((start_sec, end_sec, shoplifting_prob))
        shoplifting_scores.append((shoplifting_prob, start_sec, end_sec))

    return results, shoplifting_scores

def test_predict_video(video_path, chunk_seconds=2):
    # تأكد من وجود مجلد media/shoplifting_videos
    media_folder = os.path.join(settings.MEDIA_ROOT, 'shoplifting_videos')
    os.makedirs(media_folder, exist_ok=True)

    # اسم الملف
    filename = os.path.basename(video_path)
    destination_path = os.path.join(media_folder, filename)

    # نسخ الفيديو إلى media
    with open(video_path, 'rb') as src, open(destination_path, 'wb') as dst:
        dst.write(src.read())

    # 🔥 إنشاء object في DB
    shoplifting = Shoplifting.objects.create(
        location="Shop 1",        # تقدر تبدلها ديناميك
        camera="Camera 1",
        viewed=False
    )

    # حفظ الفيديو في FileField
    with open(destination_path, 'rb') as f:
        shoplifting.video_path.save(filename, File(f), save=True)

    return


# =========================
# 6. RUN ON FOLDER
# =========================
def predict_folder(folder_path):
    all_results = {}

    for file in os.listdir(folder_path):
        if file.endswith(".mp4") or file.endswith(".avi"):
            video_path = os.path.join(folder_path, file)

            print(f"\n🔥 Processing: {file}\n")

            results, scores = predict_video(video_path)

            all_results[file] = results

    return all_results

def resize_to_screen(frame, max_width=1280, max_height=720):
    h, w = frame.shape[:2]

    scale = min(max_width / w, max_height / h)

    if scale < 1:  # only resize if larger than screen
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    return frame

def visualize_video(video_path, chunk_seconds=2):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    chunk_size = int(fps * chunk_seconds)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    def extract_clip(frames, start, end, size=(112, 112), num_frames=16):
        segment = frames[start:end]

        processed = []
        for f in segment:
            f = cv2.resize(f, size)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            processed.append(f)

        if len(processed) == 0:
            return None

        while len(processed) < num_frames:
            processed.append(processed[-1])

        processed = processed[:num_frames]

        clip = np.stack(processed) / 255.0
        clip = torch.tensor(clip, dtype=torch.float32)
        clip = clip.unsqueeze(0).unsqueeze(0)

        return clip

    for start in range(0, len(frames), chunk_size):
        end = min(start + chunk_size, len(frames))

        clip = extract_clip(frames, start, end)
        if clip is None:
            continue

        clip = clip.to(device)

        with torch.no_grad():
            output = model(clip)
            prob = torch.softmax(output, dim=1)

            shoplifting_prob = prob[0, 1].item() * 100

        label = "SHOPLIFTING" if shoplifting_prob > 50 else "NORMAL"

        color = (0, 0, 255) if shoplifting_prob > 50 else (0, 255, 0)

        start_sec = start / fps

        # show frames in chunk
        for i in range(start, end):
            frame = frames[i].copy()

            cv2.putText(
                frame,
                f"{label} | {shoplifting_prob:.2f}%",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA
            )

            frame = resize_to_screen(frame)
            cv2.imshow("Detection", frame)

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

# =========================
# 7. RUN
# =========================

# results = predict_folder(folder_path)
# visualize_video("./testVidios/22.mp4")
# predict_video("./testVidios/123.mp4")