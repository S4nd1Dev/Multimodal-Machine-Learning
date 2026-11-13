# M1 · Baseline Pencarian Lintas Modal
IF25-40304 Pembelajaran Mesin Multimodal · Ganjil 2026/2027 · Kelompok 09 (vegapunk)

Baseline pencarian gambar↔teks (Indonesia dan Inggris) pada SEA-VL Indonesian subset
dengan encoder pra-latih yang dibekukan, tanpa pelatihan, berjalan di CPU.

## Isi arsip
- `m1_baseline.ipynb` : kode end-to-end, sudah dijalankan (semua output tersimpan)
- `README.md` : berkas ini
- `⟦audit_data.py⟧` : kode audit duplikat dan leakage (peran 1)
- `hasil/` : `val_empat_arah.csv`, `contoh_query_train.csv`

## Prasyarat
- Python 3.10+ (dijalankan pada 3.12.10)
- `pip install numpy torch transformers sentence-transformers pillow pandas ipython jupyter`
- Versi saat dijalankan: numpy 2.5.4, torch 2.15.0+cpu, sentence-transformers 6.3.0
- Bundel data dari LMS (`train.csv`, `val.csv`, `images/`). Tidak disertakan di arsip.

## Cara menjalankan
1. Letakkan bundel data di `../bundle-mahasiswa` (relatif terhadap notebook),
   atau ubah variabel `DIR` di sel kode pertama.
2. Jalankan: `jupyter nbconvert --to notebook --execute m1_baseline.ipynb`
   (atau buka notebook lalu Run All).
3. Hasil mentah ditulis ke `hasil/`, embedding ke `cache/`.

## Seed dan waktu
- Seed: 42 (`SEED` di sel kode pertama). Embedding bersifat deterministik; seed
  menentukan pemilihan 10 contoh query dan bootstrap CI (1000 resample).
- Waktu embedding dari cache kosong: ±376 detik (±6,3 menit) di CPU 12 core,
  belum termasuk unduhan model dari Hugging Face.

## Aturan yang dipatuhi
Encoder dibeku; tidak ada data di luar bundel; hiperparameter dan contoh query hanya
dari `train.csv`; `val.csv` hanya untuk tabel hasil.

## Atribusi data
SEA-VL Indonesian subset (KORIKA-AI/sea-vl_crowdsourcing_id), diturunkan dari
SEACrowd/sea-vl_crowdsourcing, lisensi CC BY-SA 4.0
(creativecommons.org/licenses/by-sa/4.0). Cahyawijaya et al. (2025),
*Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural Vision-Language
Dataset for Southeast Asia*, ACL.