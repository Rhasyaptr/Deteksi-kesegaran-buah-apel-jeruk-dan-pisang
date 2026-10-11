# Deteksi Kesegaran Buah dengan YOLOv8

Sistem deteksi kesegaran buah secara real-time menggunakan YOLOv8 dan kamera iPhone (via Iriun Webcam). Sistem mengklasifikasikan buah apel, jeruk, dan pisang menjadi **segar** atau **busuk**.

## Daftar Class

| ID | Class |
|----|-------|
| 0 | Apel Busuk |
| 1 | Apel Segar |
| 2 | Jeruk Busuk |
| 3 | Jeruk Segar |
| 4 | Pisang Busuk |
| 5 | Pisang Segar |

Saat kamera tidak menampilkan buah, tidak ada objek yang terdeteksi (ditangani dengan gambar negatif tanpa anotasi, bukan class background).

## Struktur Repository

```
.
├── train.py             # Kode training model
├── detector_yolo.py     # Kode deteksi real-time lewat kamera
├── best.pt              # Model hasil training
├── data.yaml            # Konfigurasi dataset
├── requirements.txt     # Daftar library
└── README.md
```

## Teknologi

- Python 3.9
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- Roboflow (anotasi dan pengelolaan dataset)
- Google Colab (training dengan GPU T4)
- Iriun Webcam (kamera iPhone sebagai webcam)

## Dataset

- Dibuat sendiri, dianotasi di Roboflow: (https://app.roboflow.com/mamadz-_/kesegaran-buah-rfz4n/3)
- Total gambar: 69
- Sumber: foto dan frame video dengan kamera iPhone
- Gambar negatif (tanpa buah): 2
- Augmentasi: rotasi, flip, variasi brightness/saturasi, mosaic, mixup

**Keterbatasan:** dataset dibuat dari jumlah buah yang terbatas (satu/dua buah per kondisi), sehingga model dapat kurang akurat pada buah lain yang bentuk atau warnanya berbeda. Karena data sedikit, seluruh gambar dipakai untuk training dan validasi memakai data yang sama, sehingga angka mAP **bukan** akurasi sebenarnya. Kinerja sebenarnya dinilai dari pengujian langsung dengan kamera.

## Instalasi

```bash
git clone https://github.com/Rhasyaptr/Deteksi-kesegaran-buah-apel-jeruk-dan-pisang.git
cd Rhasyaptr
pip install -r requirements.txt
```

## Cara Training

1. Buka Google Colab, pilih **Runtime → Change runtime type → T4 GPU**.
2. Jalankan `!pip install -q ultralytics roboflow` di satu sel.
3. Buka `train.py`, isi `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, dan `ROBOFLOW_VERSION` (atau set `SUMBER_DATASET = "lokal"`).
4. Tempel seluruh isi `train.py` ke satu sel dan jalankan.
5. Hasil training dan `best.pt` tersimpan otomatis.

Parameter utama:

| Parameter | Nilai |
|-----------|-------|
| Model dasar | YOLOv8n |
| Epoch | 150 |
| Ukuran gambar | 640 |
| Batch | 8 |
| Optimizer | AdamW |

## Cara Menjalankan Detector

1. Install Iriun Webcam di iPhone dan Mac, hubungkan lewat WiFi yang sama atau kabel USB.
2. Pastikan `best.pt` ada di folder yang sama dengan `detector_yolo.py`.
3. Atur `CAMERA_INDEX` di `detector_yolo.py` (biasanya 1 untuk Iriun).
4. Jalankan:

```bash
python detector_yolo.py
```

Tombol: `q` untuk keluar, `s` untuk menyimpan screenshot.

Warna kotak: **hijau** = segar, **merah** = busuk.


## Pengembangan Selanjutnya

- Menambah jumlah dan variasi buah pada dataset
- Menguji pada buah lain di luar dataset
- Menambah jenis buah

## Pembuat

Muhammad Rhasya Ekasaputra
Program Studi Teknik Elektro, Universitas Negeri Semarang
