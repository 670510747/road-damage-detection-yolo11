"""Optional original macOS prototype. Box-size alerts are not measured distances."""
import cv2
from ultralytics import YOLO
import cv2
import subprocess
import os
import tempfile
import threading
import pygame

import platform
from pathlib import Path

if __name__ != "__main__":
    raise RuntimeError("Run this example as a script, not by importing it.")
if platform.system() != "Darwin":
    raise SystemExit("This original Thai speech demo requires macOS (say, Kanya, Krungthep, MPS).")
weights = Path(os.environ.get("ROAD_DAMAGE_WEIGHTS", "workspace/runs/E4_obb_s_improved_lowlight/weights/best.pt"))
if not weights.is_file():
    raise SystemExit(f"Trained checkpoint not found: {weights}")
model = YOLO(str(weights))

# cap = cv2.VideoCapture(0)
# เปิดเชื่อมจากโทรศัพท์ตาม IP
camera_source = os.environ.get("ROAD_DAMAGE_CAMERA", "0")
cap = cv2.VideoCapture(int(camera_source) if camera_source.isdigit() else camera_source)
if not cap.isOpened():
    raise SystemExit("Cannot open camera; set ROAD_DAMAGE_CAMERA to your own source.")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# อ่านเฟรมใน thread แยก เก็บไว้แค่เฟรมล่าสุด กันเฟรมค้างใน buffer จนดีเลย์สะสม
latest_frame = None
running = True
frame_lock = threading.Lock()

def reader():
    global latest_frame, running
    while running:
        ret, f = cap.read()
        if not ret:
            running = False
            break
        with frame_lock:
            latest_frame = f

threading.Thread(target=reader, daemon=True).start()

frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_area = frame_h * frame_w
if frame_area <= 0:
    cap.release()
    raise SystemExit("Camera returned invalid frame dimensions.")

last_spoken = {}  # กันพูดซ้ำถี่เกินไป

# say/afplay เปิด process ใหม่ทุกครั้ง ช้า ~1 วินาที
# เลยอัดประโยคเตือนเป็นไฟล์ไว้ก่อน แล้วเล่นด้วย pygame.mixer ที่เปิดลำโพงค้างไว้ เสียงออกทันที
pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
speech_channel = pygame.mixer.Channel(0)
speech_cache = {}       # text -> pygame.mixer.Sound
current_priority = -1

def warning_text(thai, priority):
    return f"ระวัง {thai}ใกล้มาก" if priority == 2 else f"พบ{thai}"

def prerecord(texts):
    cache_dir = tempfile.mkdtemp(prefix="road_speech_")
    procs = []
    for i, text in enumerate(texts):
        path = os.path.join(cache_dir, f"{i}.wav")
        procs.append((text, path, subprocess.Popen(
            ["say", "-v", "Kanya", "-r", "220", "--data-format=LEI16@22050", "-o", path, text])))
    for text, path, p in procs:
        if p.wait() == 0:
            speech_cache[text] = pygame.mixer.Sound(path)

def speak(text, priority=0):
    """เล่นเสียงเตือนโดยไม่หยุด loop ไม่มีคิว — ถ้ากำลังพูดอยู่ จะพูดทับได้เฉพาะข้อความที่สำคัญกว่า
    คืนค่า True ถ้าได้พูด"""
    global current_priority
    if speech_channel.get_busy() and priority <= current_priority:
        return False
    current_priority = priority
    if text in speech_cache:
        speech_channel.play(speech_cache[text])   # play ทับ = ตัดเสียงเก่าทิ้งทันที
    else:
        subprocess.Popen(["say", "-v", "Kanya", "-r", "220", text])
    return True

def estimate_distance(box_area):
    ratio = box_area / frame_area
    if ratio > 0.15:
        return "ใกล้มาก", (0, 0, 255)
    elif ratio > 0.05:
        return "ระวัง", (0, 165, 255)
    else:
        return "ไกล", (0, 255, 0)

thai_name = {
    "pothole": "หลุมบ่อ",
    "crack": "รอยแตก",
    "manhole": "ฝาท่อ"
}

prerecord([warning_text(thai_name.get(n, n), p) for n in model.names.values() for p in (1, 2)])

import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ใช้ Krungthep เพราะ Pillow ไม่มี raqm ฟอนต์อย่าง Thonburi จะวางสระ/วรรณยุกต์เพี้ยน
thai_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Krungthep.ttf", 24)

def draw_thai_texts(img, texts):
    """วาดข้อความไทยทีเดียวทั้งเฟรม (cv2.putText วาดไทยไม่ได้)
    texts = [(ข้อความ, (x, y), สี BGR)]"""
    if not texts:
        return img
    # ไม่แปลง BGR->RGB เพราะส่งสี BGR เข้าไปตรง ๆ ค่าในอาเรย์จะถูกต้องเอง
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for text, (x, y), color in texts:
        draw.text((x, y), text, font=thai_font, fill=color,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(pil_img)

while running:
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        time.sleep(0.01)
        continue

    results = model(frame, verbose=False, conf=0.25, device="mps", imgsz=640)
    annotated = results[0].plot()
    texts = []
    most_urgent = None  # (priority, box_area, label, thai) — เฟรมละเสียงเดียว

    for box in results[0].obb:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls_id]
        thai = thai_name.get(label, label)

        obb_pts = box.xyxyxyxy[0].cpu().numpy()
        x_coords = obb_pts[:, 0]
        y_coords = obb_pts[:, 1]
        box_area = (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min())

        level, color = estimate_distance(box_area)

        priority = {"ใกล้มาก": 2, "ระวัง": 1}.get(level, 0)
        if priority > 0 and (most_urgent is None or (priority, box_area) > most_urgent[:2]):
            most_urgent = (priority, box_area, label, thai)

        x, y = int(x_coords.min()), int(y_coords.min())
        texts.append((f"{thai} {level}", (x, max(y - 32, 0)), color))

    # พูดแจ้งเตือนเฉพาะอันที่ด่วนสุด ไม่พูดซ้ำ label+ระดับเดิมถี่เกิน 3 วินาที
    if most_urgent is not None:
        priority, _, label, thai = most_urgent
        key = (label, priority)
        now = time.time()
        if now - last_spoken.get(key, 0) > 3:
            if speak(warning_text(thai, priority), priority):
                last_spoken[key] = now

    annotated = draw_thai_texts(annotated, texts)
    cv2.imshow("Road Damage Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running = False
cap.release()
cv2.destroyAllWindows()
