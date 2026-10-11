# ============================================================
# TRAINING MODEL DETEKSI KESEGARAN BUAH (YOLOv8)
# Kelas: Apel / Jeruk / Pisang x (Segar, Busuk)
# Jalankan di Google Colab (GPU T4) atau lokal.
# Di Colab, jalankan dulu di sel terpisah:  !pip install -q ultralytics roboflow
# ============================================================
import os
import glob
import shutil
import yaml
from collections import Counter

import torch
from ultralytics import YOLO

# ---------- 1. KONFIGURASI ----------
SUMBER_DATASET = "roboflow"       # "roboflow" atau "lokal"

# Jika "roboflow" (JANGAN upload API key asli ke GitHub)
ROBOFLOW_API_KEY   = "ISI_API_KEY_ANDA"
ROBOFLOW_WORKSPACE = "ISI_WORKSPACE"
ROBOFLOW_PROJECT   = "ISI_NAMA_PROJECT"
ROBOFLOW_VERSION   = 1

# Jika "lokal": isi lokasi data.yaml
DATA_YAML_LOKAL = "dataset_buah/data.yaml"

MODEL_BASE   = "yolov8n.pt"       # n = ringan, s = lebih akurat
EPOCHS       = 150
IMG_SIZE     = 640
BATCH_SIZE   = 8
RUN_NAME     = "model_sortir_buah"
OUTPUT_DIR   = "runs"
VAL_SAMA_DENGAN_TRAIN = True      # True jika semua gambar dipakai untuk train (data sangat sedikit)

# ---------- 2. CEK PERANGKAT ----------
if torch.cuda.is_available():
    device = 0
    print("Perangkat: GPU", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = "mps"
    print("Perangkat: GPU Apple (MPS)")
else:
    device = "cpu"
    print("Perangkat: CPU (training akan lambat)")

# ---------- 3. AMBIL DATASET ----------
if SUMBER_DATASET == "roboflow":
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
    DATA_YAML = os.path.join(dataset.location, "data.yaml")
else:
    DATA_YAML = os.path.abspath(DATA_YAML_LOKAL)

if not os.path.exists(DATA_YAML):
    hasil_cari = glob.glob(os.path.join(os.path.dirname(DATA_YAML), "**", "data.yaml"), recursive=True)
    if not hasil_cari:
        raise SystemExit(f"data.yaml tidak ditemukan: {DATA_YAML}")
    DATA_YAML = hasil_cari[0]
print("data.yaml:", DATA_YAML)

# ---------- 4. PERBAIKI PATH DI data.yaml ----------
base_dir = os.path.dirname(DATA_YAML)
with open(DATA_YAML, "r") as f:
    cfg = yaml.safe_load(f)

cfg["path"]  = base_dir
cfg["train"] = "train/images"
if VAL_SAMA_DENGAN_TRAIN or not os.path.exists(os.path.join(base_dir, "valid/images")):
    cfg["val"] = "train/images"
else:
    cfg["val"] = "valid/images"
if os.path.exists(os.path.join(base_dir, "test/images")) and not VAL_SAMA_DENGAN_TRAIN:
    cfg["test"] = "test/images"
else:
    cfg.pop("test", None)

with open(DATA_YAML, "w") as f:
    yaml.dump(cfg, f)

# ---------- 5. CEK ISI DATASET ----------
names = cfg["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
print("\nClass:", names)

if any("background" in n.lower() for n in names):
    print("PERINGATAN: ada class Background. Sebaiknya dihapus, pakai gambar negatif (Null).")

jumlah_gambar = len(os.listdir(os.path.join(base_dir, "train/images")))
print("Jumlah gambar train:", jumlah_gambar)

hitung = Counter()
gambar_negatif = 0
for lf in glob.glob(os.path.join(base_dir, "train/labels/*.txt")):
    baris = [b for b in open(lf).read().strip().splitlines() if b]
    if not baris:
        gambar_negatif += 1
    for b in baris:
        hitung[int(b.split()[0])] += 1

print("Gambar negatif (tanpa label):", gambar_negatif)
print("Jumlah anotasi per class:")
for i, nama in enumerate(names):
    print(f"  {nama}: {hitung[i]}")
    if hitung[i] == 0:
        print(f"  PERINGATAN: class '{nama}' tidak punya anotasi di train")

# ---------- 6. TRAINING ----------
model = YOLO(MODEL_BASE)

print("\nMemulai training...")
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=device,
    project=os.path.abspath(OUTPUT_DIR),
    name=RUN_NAME,
    exist_ok=True,
    patience=0,               # tidak berhenti dini (data kecil)
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    workers=2,
    seed=42,
    plots=True,
    # --- augmentasi ---
    hsv_h=0.015,              # kecil agar warna segar/busuk tidak tertukar
    hsv_s=0.6,
    hsv_v=0.5,
    degrees=20,
    translate=0.15,
    scale=0.6,
    fliplr=0.5,
    flipud=0.2,
    mosaic=1.0,
    mixup=0.15,
    close_mosaic=20,
)

# ---------- 7. EVALUASI ----------
hasil_dir = os.path.join(os.path.abspath(OUTPUT_DIR), RUN_NAME)
best_path = os.path.join(hasil_dir, "weights", "best.pt")
model = YOLO(best_path)

metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, device=device)
print("\n=== HASIL EVALUASI ===")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
if VAL_SAMA_DENGAN_TRAIN:
    print("CATATAN: validasi memakai data yang sama dengan train, angka ini bukan akurasi sebenarnya.")

# ---------- 8. SIMPAN HASIL ----------
shutil.copy(best_path, "best.pt")
print("\nModel disalin ke: best.pt")
print("Grafik & confusion matrix ada di:", hasil_dir)

# Khusus Google Colab: backup ke Drive dan download otomatis
try:
    from google.colab import drive, files
    drive.mount("/content/drive")
    shutil.copytree(hasil_dir, f"/content/drive/MyDrive/{RUN_NAME}", dirs_exist_ok=True)
    print("Backup tersimpan di Google Drive:", f"MyDrive/{RUN_NAME}")
    files.download("best.pt")
except ImportError:
    pass   # bukan Colab, lewati

print("\nTraining selesai.")