# ============================================================
# DETEKSI KESEGARAN BUAH REAL-TIME (Mac + Iriun Webcam / iPhone)
# ============================================================
import cv2
import time
import threading
import torch
from ultralytics import YOLO

# ---------- 1. KONFIGURASI ----------
MODEL_PATH     = 'best.pt'
CAMERA_INDEX   = 1        # 0 = kamera Mac, 1 = Iriun (coba 0,1,2)
CONF_THRESH    = 0.35     # naikkan jika banyak salah deteksi
IOU_THRESH     = 0.5
IMG_SIZE       = 480      # lebih kecil = lebih ringan (320/480/640)
SKIP_FRAME     = 2        # proses AI tiap 2 frame
MIN_AREA_RATIO = 0.01     # abaikan kotak < 1% luas frame
MAX_MISS       = 3        # hapus kotak setelah 3 pengecekan kosong berturut-turut

# ---------- 2. PILIH PERANGKAT ----------
if torch.backends.mps.is_available():
    device = 'mps'        # GPU Apple Silicon, jauh lebih cepat dari CPU
else:
    device = 'cpu'
print("Perangkat:", device)

# ---------- 3. LOAD MODEL ----------
print("Memuat model...")
model = YOLO(MODEL_PATH)
print("Class model:", model.names)

# ---------- 4. BUKA KAMERA ----------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # kurangi antrean frame lama
# Resolusi sengaja TIDAK dipaksa agar gambar tidak blur / diperbesar

if not cap.isOpened():
    print("Kamera tidak bisa dibuka. Ganti CAMERA_INDEX (0/1/2).")
    raise SystemExit

# Pemanasan: tunggu frame pertama (maks ~5 detik)
print("Menunggu kamera siap...")
siap = False
for _ in range(50):
    ret, frame = cap.read()
    if ret:
        siap = True
        break
    time.sleep(0.1)

if not siap:
    print("Kamera terbuka tapi tidak ada gambar.")
    print("Cek: izin kamera VS Code/Terminal, Iriun aktif di iPhone & Mac, index kamera.")
    cap.release()
    raise SystemExit

print("Ukuran frame kamera:", frame.shape[1], "x", frame.shape[0])

# ---------- 5. THREAD PEMBACA KAMERA (selalu simpan frame terbaru) ----------
frame_terbaru = frame
frame_baru = True
kamera_jalan = True
kunci = threading.Lock()

def baca_kamera():
    global frame_terbaru, frame_baru, kamera_jalan
    gagal = 0
    while kamera_jalan:
        ret, f = cap.read()
        if not ret:
            gagal += 1
            if gagal >= 50:
                kamera_jalan = False
                break
            time.sleep(0.02)
            continue
        gagal = 0
        with kunci:
            frame_terbaru = f
            frame_baru = True

thread_kamera = threading.Thread(target=baca_kamera, daemon=True)
thread_kamera.start()

# ---------- 6. WARM-UP MODEL (supaya frame pertama tidak macet) ----------
model.predict(frame, imgsz=IMG_SIZE, device=device, verbose=False)

print("Siap. Tekan 'q' untuk keluar, 's' untuk simpan screenshot.")

# ---------- 7. LOOP DETEKSI ----------
frame_count = 0
deteksi_terakhir = []
miss_count = 0
waktu_lalu = time.time()
fps = 0.0

while kamera_jalan:
    # ambil frame terbaru (lewati jika belum ada frame baru)
    with kunci:
        if not frame_baru:
            ada_baru = False
        else:
            ada_baru = True
            frame = frame_terbaru.copy()
            frame_baru = False

    if not ada_baru:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame_count += 1
    tinggi, lebar = frame.shape[:2]
    luas_frame = tinggi * lebar

    # AI hanya memproses sebagian frame agar ringan
    if frame_count % SKIP_FRAME == 0:
        results = model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH,
                                imgsz=IMG_SIZE, device=device,
                                verbose=False)[0]

        deteksi_baru = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            nama = model.names[int(box.cls[0])]

            # abaikan class background (jika masih ada)
            if "background" in nama.lower():
                continue

            # abaikan objek yang terlalu kecil
            if (x2 - x1) * (y2 - y1) < MIN_AREA_RATIO * luas_frame:
                continue

            deteksi_baru.append((x1, y1, x2, y2, nama, conf))

        if len(deteksi_baru) > 0:
            deteksi_terakhir = deteksi_baru
            miss_count = 0
        else:
            miss_count += 1
            if miss_count >= MAX_MISS:
                deteksi_terakhir = []

    # gambar hasil deteksi terakhir
    for (x1, y1, x2, y2, nama, conf) in deteksi_terakhir:
        busuk = ("rotten" in nama.lower()) or ("busuk" in nama.lower())
        warna = (0, 0, 255) if busuk else (0, 255, 0)   # merah = busuk, hijau = segar

        teks = f"{nama} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), warna, -1)
        cv2.putText(frame, teks, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # status & FPS
    if len(deteksi_terakhir) == 0:
        cv2.putText(frame, "Tidak ada buah terdeteksi", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    sekarang = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(sekarang - waktu_lalu, 1e-6))
    waktu_lalu = sekarang
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Deteksi Kesegaran Buah", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"capture_{frame_count}.jpg", frame)
        print("Screenshot tersimpan.")

# ---------- 8. BERSIHKAN ----------
kamera_jalan = False
time.sleep(0.2)
cap.release()
cv2.destroyAllWindows()