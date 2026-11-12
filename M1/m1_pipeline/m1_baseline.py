"""M1 baseline pencarian lintas modal.
Langkah 2-3: muat data, embedding dengan cache, peringkat empat arah.

Pemakaian:
    python m1_baseline.py --data-dir . --seed 0

Encoder dibekukan (hanya inferensi), semua berjalan di CPU.
Data: SEA-VL Indonesian subset (CC BY-SA 4.0), Cahyawijaya et al. (2025).
"""
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd

NAMA_GAMBAR = "sentence-transformers/clip-ViT-B-32"                  # gambar + teks Inggris
NAMA_TEKS_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"  # teks Indonesia
BATCH_GAMBAR = 32
BATCH_TEKS = 64


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def sidik(daftar):
    """Sidik jari isi masukan: cache dianggap basi bila isi atau urutan CSV berubah."""
    h = hashlib.md5()
    for x in daftar:
        h.update(str(x).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def embed_dengan_cache(nama, kunci, model_nama, hitung, cache_dir):
    """Muat embedding dari cache bila sah; bila tidak, hitung lalu simpan.
    Mengembalikan (embedding, info waktu)."""
    path_npy = os.path.join(cache_dir, nama + ".npy")
    path_meta = os.path.join(cache_dir, nama + ".json")
    if os.path.exists(path_npy) and os.path.exists(path_meta):
        with open(path_meta, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("sidik") == kunci and meta.get("model") == model_nama:
            E = np.load(path_npy)
            print("  [cache]  %-14s %s" % (nama, E.shape))
            return E, {"detik_hitung": meta["detik_hitung"], "dari_cache": True}
    t0 = time.perf_counter()
    E = hitung()
    detik = round(time.perf_counter() - t0, 2)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path_npy, E)
    with open(path_meta, "w", encoding="utf-8") as f:
        json.dump({"sidik": kunci, "model": model_nama, "shape": list(E.shape),
                   "detik_hitung": detik}, f)
    print("  [hitung] %-14s %s  %.1f detik" % (nama, E.shape, detik))
    return E, {"detik_hitung": detik, "dari_cache": False}


def embed_gambar(model, data_dir, daftar_file):
    from PIL import Image
    hasil = []
    for i in range(0, len(daftar_file), BATCH_GAMBAR):
        batch = []
        for f in daftar_file[i:i + BATCH_GAMBAR]:
            with Image.open(os.path.join(data_dir, f)) as im:
                batch.append(im.convert("RGB"))
        hasil.append(model.encode(batch, batch_size=BATCH_GAMBAR,
                                  normalize_embeddings=True, show_progress_bar=False))
        print("    gambar %d/%d" % (min(i + BATCH_GAMBAR, len(daftar_file)), len(daftar_file)))
    return np.vstack(hasil).astype("float32")


def embed_teks(model, daftar_teks):
    return model.encode(list(daftar_teks), batch_size=BATCH_TEKS,
                        normalize_embeddings=True, show_progress_bar=False).astype("float32")


# kode, nama arah, modalitas query, modalitas kandidat
ARAH = [
    ("g2id", "gambar->teks ID", "gambar", "teks_id"),
    ("g2en", "gambar->teks EN", "gambar", "teks_en"),
    ("id2g", "teks ID->gambar", "teks_id", "gambar"),
    ("en2g", "teks EN->gambar", "teks_en", "gambar"),
]


def peringkat_jawaban_benar(S):
    """S[i, j] = kesamaan kosinus query i dengan kandidat j; jawaban benar adalah j == i.
    Peringkat pesimis: bila kandidat lain bernilai sama atau lebih tinggi daripada
    jawaban benar, jawaban benar dianggap di bawahnya (mencegah angka terlalu bagus
    pada caption identik). Mengembalikan (peringkat mulai 1, jumlah seri per query)."""
    benar = np.diag(S)[:, None]
    peringkat = (S >= benar).sum(axis=1)          # termasuk dirinya sendiri -> mulai dari 1
    seri = (S == benar).sum(axis=1) - 1
    return peringkat, seri


def recall_at_k(peringkat, ks=(1, 5, 10)):
    return {k: float((peringkat <= k).mean()) for k in ks}


def map_at_k(peringkat, k=10):
    """Tiap query punya satu jawaban benar, jadi AP@K = 1/peringkat bila peringkat <= K, selain itu 0."""
    return float(np.where(peringkat <= k, 1.0 / peringkat, 0.0).mean())


def evaluasi_empat_arah(E):
    """E: dict embedding {'gambar','teks_id','teks_en'}, semua (N, D) dan ternormalisasi.
    Kosinus = perkalian titik. Mengembalikan (ringkasan, peringkat per arah, top-10 per arah)."""
    ringkasan, semua_peringkat, semua_top10 = {}, {}, {}
    for kode, nama, q, c in ARAH:
        S = E[q] @ E[c].T
        peringkat, seri = peringkat_jawaban_benar(S)
        r = recall_at_k(peringkat)
        ringkasan[kode] = {"arah": nama, "R@1": r[1], "R@5": r[5], "R@10": r[10],
                           "mAP@10": map_at_k(peringkat, 10),
                           "median_peringkat": float(np.median(peringkat)),
                           "query_dengan_seri": int((seri > 0).sum())}
        semua_peringkat[kode] = peringkat
        semua_top10[kode] = np.argsort(-S, axis=1)[:, :10]
    return ringkasan, semua_peringkat, semua_top10


def cetak_tabel(ringkasan, N):
    print("\nHasil pada split val (N = %d), seed tidak memengaruhi angka ini:" % N)
    print("%-17s %7s %7s %7s %9s %10s" % ("Arah", "R@1", "R@5", "R@10", "mAP@10", "med.rank"))
    for r in ringkasan.values():
        print("%-17s %7.4f %7.4f %7.4f %9.4f %10.1f" % (
            r["arah"], r["R@1"], r["R@5"], r["R@10"], r["mAP@10"], r["median_peringkat"]))
    print("%-17s %7.4f %7.4f %7.4f" % ("acak (1,5,10/N)", 1 / N, 5 / N, 10 / N))
    print("Gap R@1 (EN - ID): gambar->teks = %+.4f | teks->gambar = %+.4f" % (
        ringkasan["g2en"]["R@1"] - ringkasan["g2id"]["R@1"],
        ringkasan["en2g"]["R@1"] - ringkasan["id2g"]["R@1"]))
    seri = sum(r["query_dengan_seri"] for r in ringkasan.values())
    if seri:
        print("Catatan: %d query punya skor seri dengan jawaban lain (caption identik); dihitung pesimis." % seri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="folder berisi train.csv, val.csv, images/")
    ap.add_argument("--cache-dir", default="cache", help="folder penyimpanan embedding")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    t_total = time.perf_counter()

    # ---- 1. Muat data ----
    df = pd.read_csv(os.path.join(args.data_dir, args.split + ".csv"))
    files = list(df["file"])
    cap_id = list(df["caption_id"])
    cap_en = list(df["caption_en"])
    print("Split %s: %d baris | seed %d" % (args.split, len(df), args.seed))

    # ---- 2. Model dimuat malas: hanya bila ada cache yang perlu dihitung ----
    model_cache = {}

    def model(nama):
        if nama not in model_cache:
            from sentence_transformers import SentenceTransformer
            t0 = time.perf_counter()
            model_cache[nama] = SentenceTransformer(nama, device="cpu")
            print("  model dimuat: %s (%.1f detik)" % (nama.split("/")[-1], time.perf_counter() - t0))
        return model_cache[nama]

    # ---- 3. Embedding dengan cache ----
    print("Embedding (cache di '%s'):" % args.cache_dir)
    info = {}
    E_gambar, info["gambar"] = embed_dengan_cache(
        args.split + "_gambar", sidik(files), NAMA_GAMBAR,
        lambda: embed_gambar(model(NAMA_GAMBAR), args.data_dir, files), args.cache_dir)
    E_id, info["teks_id"] = embed_dengan_cache(
        args.split + "_teks_id", sidik(cap_id), NAMA_TEKS_ID,
        lambda: embed_teks(model(NAMA_TEKS_ID), cap_id), args.cache_dir)
    E_en, info["teks_en"] = embed_dengan_cache(
        args.split + "_teks_en", sidik(cap_en), NAMA_GAMBAR,
        lambda: embed_teks(model(NAMA_GAMBAR), cap_en), args.cache_dir)

    # ---- 4. Pemeriksaan masuk akal ----
    for nama, E in (("gambar", E_gambar), ("teks_id", E_id), ("teks_en", E_en)):
        assert E.shape[0] == len(df), "jumlah baris %s tidak sama dengan CSV" % nama
        norma = np.linalg.norm(E, axis=1)
        assert np.allclose(norma, 1.0, atol=1e-3), "embedding %s tidak ternormalisasi" % nama
    print("Pemeriksaan: jumlah baris cocok, norma tiap vektor = 1  -> OK")

    # ---- 5. Peringkat dan metrik untuk empat arah ----
    t_eval = time.perf_counter()
    E = {"gambar": E_gambar, "teks_id": E_id, "teks_en": E_en}
    ringkasan, semua_peringkat, semua_top10 = evaluasi_empat_arah(E)
    cetak_tabel(ringkasan, len(df))
    waktu_peringkat = round(time.perf_counter() - t_eval, 3)
    arsip = {}
    for kode in semua_peringkat:
        arsip[kode + "_peringkat"] = semua_peringkat[kode]
        arsip[kode + "_top10"] = semua_top10[kode]
    np.savez(os.path.join(args.cache_dir, "peringkat_%s.npz" % args.split), **arsip)
    with open(os.path.join(args.cache_dir, "hasil_%s.json" % args.split), "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, indent=2, ensure_ascii=False)
    print("Hasil disimpan: hasil_%s.json dan peringkat_%s.npz (peringkat + top-10 tiap query)"
          % (args.split, args.split))

    # ---- 6. Catat perangkat dan waktu untuk laporan ----
    try:
        import torch
        versi_torch = torch.__version__
    except ImportError:
        versi_torch = "-"
    catatan = {
        "split": args.split, "seed": args.seed, "n": len(df),
        "python": sys.version.split()[0], "torch": versi_torch, "numpy": np.__version__,
        "platform": platform.platform(), "perangkat": "CPU (tanpa GPU)",
        "waktu_embedding_detik": info,
        "waktu_peringkat_detik": waktu_peringkat,
        "waktu_sesi_ini_detik": round(time.perf_counter() - t_total, 2),
    }
    os.makedirs(args.cache_dir, exist_ok=True)
    with open(os.path.join(args.cache_dir, "catatan_%s.json" % args.split), "w", encoding="utf-8") as f:
        json.dump(catatan, f, indent=2, ensure_ascii=False)
    print("Catatan perangkat dan waktu disimpan di %s/catatan_%s.json" % (args.cache_dir, args.split))
    print("Waktu sesi ini: %.1f detik" % catatan["waktu_sesi_ini_detik"])
    return E_gambar, E_id, E_en


if __name__ == "__main__":
    main()
