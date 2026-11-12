"""Langkah 1 M1: cek lingkungan, data, dan model sebelum menulis pipeline.
Pemakaian:  python langkah1_cek.py --data-dir .
            python langkah1_cek.py --data-dir . --skip-model   (hanya cek data)
"""
import argparse
import os
import sys
import time

import pandas as pd

NAMA_GAMBAR = "sentence-transformers/clip-ViT-B-32"
NAMA_TEKS_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=".", help="folder yang berisi train.csv, val.csv, images/")
ap.add_argument("--skip-model", action="store_true", help="lewati pengecekan model")
args = ap.parse_args()

print("Python :", sys.version.split()[0])

# ---------- 1. Berkas dan kolom ----------
kolom_wajib = ["image_id", "file", "caption_id", "caption_en", "category"]
data = {}
for nama in ("train", "val"):
    path = os.path.join(args.data_dir, nama + ".csv")
    if not os.path.exists(path):
        sys.exit("TIDAK ADA: %s  (cek --data-dir)" % path)
    df = pd.read_csv(path)
    hilang = [k for k in kolom_wajib if k not in df.columns]
    if hilang:
        sys.exit("Kolom hilang di %s: %s" % (nama, hilang))
    data[nama] = df
    print("%-5s : %d baris" % (nama, len(df)))

# ---------- 2. Gambar ada semua? ----------
for nama, df in data.items():
    tidak_ada = [f for f in df["file"] if not os.path.exists(os.path.join(args.data_dir, f))]
    print("%-5s : gambar hilang = %d dari %d" % (nama, len(tidak_ada), len(df)))
    if tidak_ada:
        print("         contoh:", tidak_ada[:3])

# ---------- 3. Model bisa dimuat? ----------
if args.skip_model:
    print("Model  : dilewati (--skip-model)")
    sys.exit(0)

try:
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    sys.exit("Paket belum terpasang: %s\n  pip install torch sentence-transformers pillow pandas" % e)

print("torch  :", torch.__version__, "| GPU tersedia:", torch.cuda.is_available(), "(tidak dipakai)")

t0 = time.perf_counter()
model_gambar = SentenceTransformer(NAMA_GAMBAR, device="cpu")
model_teks_id = SentenceTransformer(NAMA_TEKS_ID, device="cpu")
print("Model dimuat dalam %.1f detik" % (time.perf_counter() - t0))

baris = data["val"].iloc[0]
gambar = Image.open(os.path.join(args.data_dir, baris["file"])).convert("RGB")
e_gambar = model_gambar.encode([gambar], normalize_embeddings=True)
e_id = model_teks_id.encode([baris["caption_id"]], normalize_embeddings=True)
e_en = model_gambar.encode([baris["caption_en"]], normalize_embeddings=True)
print("Dimensi embedding: gambar", e_gambar.shape, "| teks ID", e_id.shape, "| teks EN", e_en.shape)
print("Kosinus gambar-ID: %.3f | gambar-EN: %.3f" % ((e_gambar @ e_id.T).item(), (e_gambar @ e_en.T).item()))
print("SELESAI: lingkungan siap.")
