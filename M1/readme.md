# M1: Baseline pencarian lintas modal (Kelompok 09)

IF25-40304 Pembelajaran Mesin Multimodal, ITERA

## Cara menjalankan

1. Atur `DIR` di notebook ke folder bundel SEA-VL (berisi `train.csv`, `val.csv`, `images/`); default `../bundle-mahasiswa`. `cache/` dan `hasil/` dibuat di folder notebook.
2. Dependensi dipasang oleh sel pertama (`%pip install ...`), atau manual: `pip install numpy torch transformers sentence-transformers pillow pandas jupyter`
3. Buka notebook, **Run All** (atau `jupyter nbconvert --to notebook --execute --inplace m1_baseline.ipynb`).
4. Hanya CPU. Embedding disimpan di `cache/` (hapus folder ini untuk mengukur waktu jalan dari cache kosong).

## Isi

- Encoder gambar dan teks Inggris: `sentence-transformers/clip-ViT-B-32` (beku)
- Encoder teks Indonesia: `sentence-transformers/clip-ViT-B-32-multilingual-v1` (beku)
- Kesamaan kosinus pada embedding ternormalisasi, empat arah: gambar→teks ID, gambar→teks EN, teks ID→gambar, teks EN→gambar
- Metrik: Recall@1/5/10 dan mAP@10 pada `val.csv` (satu jawaban benar per query; skor sama dihitung pesimistik), plus CI 95% bootstrap untuk R@1
- Sepuluh contoh query Indonesia diambil acak dari `train.csv`, bukan dari val

## Seed dan keluaran

- Seed: 42 (NumPy, PyTorch, sampel contoh, bootstrap)
- `hasil/val_empat_arah.csv`: tabel hasil validasi
- `hasil/contoh_query_train.csv`: sepuluh contoh query
- `cache/*.npy`, `cache/waktu.json`: embedding dan waktu komputasi

## Atribusi data

SEA-VL Indonesian subset (KORIKA-AI/sea-vl_crowdsourcing_id), diturunkan dari SEACrowd/sea-vl_crowdsourcing, lisensi CC BY-SA 4.0. Cahyawijaya et al. (2025), _Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural Vision-Language Dataset for Southeast Asia_, ACL.
