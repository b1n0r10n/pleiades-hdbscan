# Pleiades HDBSCAN (Gaia) — Clustering & Validation

Proyek ini melakukan **klasterisasi anggota gugus Pleiades** memakai **HDBSCAN** pada data **Gaia**, lalu membandingkan hasilnya dengan **dataset valid** (anggota referensi) melalui ringkasan numerik dan visualisasi. Kode dipisah menjadi modul yang rapi (`src/pleiades`), skrip reprodusibel (`scripts/*.py`), uji otomatis (`tests/*.py`), dan sebuah notebook laporan (`notebooks/00_report.ipynb`).

---

## Fitur Utama
- Pipeline modular: **IO → features → preprocessing → clustering → validation → plotting**  
- Konfigurasi terpusat di `configs/default.yaml`
- Uji otomatis dengan **pytest**
- Notebook laporan yang hanya **mengonsumsi output** pipeline (bukan logic utama)

---

## Struktur Repo
```
pleiades-hdbscan/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ .gitattributes
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ raw/        # letakkan file mentah di sini (tidak di-commit)
│  ├─ interim/    # hasil harmonisasi
│  └─ processed/  # hasil clustering, validasi, dan figur
├─ notebooks/
│  └─ 00_report.ipynb
├─ scripts/
│  ├─ 01_preprocess.py
│  ├─ 02_cluster.py
│  ├─ 03_validate.py
│  └─ 04_make_figures.py
├─ src/
│  └─ pleiades/
│     ├─ __init__.py
│     ├─ io.py
│     ├─ features.py
│     ├─ preprocessing.py
│     ├─ clustering.py
│     ├─ validation.py
│     └─ plotting.py
└─ tests/
   ├─ conftest.py
   ├─ test_io.py
   ├─ test_features.py
   ├─ test_preprocessing.py
   ├─ test_clustering.py
   ├─ test_validation.py
   └─ test_plotting_smoke.py

## Instalasi

### Opsi A — conda (disarankan)
```bash
conda env create -f environment.yml
conda activate pleiades-hdbscan
python -m ipykernel install --user --name pleiades-hdbscan
```

### Opsi B — pip/virtualenv
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data yang Dibutuhkan
1. **Dataset Gaia** hasil query: `data/raw/pleiadesdataset.csv`  
2. **Dataset valid** (anggota referensi dari sumber lain): `data/raw/pleiadesvalid.csv`

Sesuaikan **pemetaan kolom** & **parameter** di `configs/default.yaml`.

---

## Menjalankan Pipeline
```bash
# 1) Preprocess (harmonisasi kolom, filter RUWE, hitung color & M_G)
python scripts/01_preprocess.py

# 2) Clustering (standarisasi fitur → HDBSCAN), opsional simpan model
python scripts/02_cluster.py --save-model

# 3) Validasi (ringkasan klaster, silhouette* jika tersedia, komparasi dgn valid)
python scripts/03_validate.py

# 4) Figur PNG standar (sky scatter/quiver, PM plane, CMD, histogram, condensed tree*)
python scripts/04_make_figures.py
```
Output utama akan berada di `data/processed/` dan figur di `data/processed/figures/`.
Buka laporan naratif:
```bash
jupyter lab notebooks/00_report.ipynb
```

---

## Contoh `configs/default.yaml`
```yaml
paths:
  raw_gaia: "data/raw/pleiadesdataset.csv"
  raw_valid: "data/raw/pleiadesvalid.csv"
  interim: "data/interim"
  processed: "data/processed"

schema:
  gaia:
    ra: ra
    dec: dec
    parallax: parallax
    pmra: pmra
    pmdec: pmdec
    g: phot_g_mean_mag
    bp: phot_bp_mean_mag
    rp: phot_rp_mean_mag
    ruwe: ruwe
  valid:
    ra: RA_ICRS
    dec: DE_ICRS
    parallax: plx
    pmra: pmRA
    pmdec: pmDE
    gmag: Gmag
    color: Bp-Rp

params:
  ruwe_max: 1.4
  features: [parallax, pmra, pmdec, color, M_G]
  membership_prob_threshold: 0.5
  hdbscan:
    min_cluster_size: 30
    min_samples:
    metric: euclidean
    cluster_selection_epsilon: 0.0
    cluster_selection_method: eom
    gen_min_span_tree: true
    allow_single_cluster: false
  plots:
    max_arrows: 800
```

---

## Pengujian
Jalankan semua tes untuk memastikan pipeline stabil:
```bash
pytest -q
```

---

## Sitasi & Kredit
- **CITATION.cff** disediakan untuk format sitasi perangkat lunak.
- **Data:** Gaia Archive (ESA). Harap sertakan kredit misi Gaia pada publikasi/grafik.
- **Algoritme:** https://hdbscan.readthedocs.io/

---

## Lisensi
Kode dirilis di bawah lisensi **MIT** (lihat `LICENSE`).

---

## FAQ Singkat
**Q:** Mengapa hasil CMD saya kosong?  
**A:** Pastikan kolom `phot_bp_mean_mag`, `phot_rp_mean_mag`, `phot_g_mean_mag`, dan `parallax > 0` tersedia di Gaia; pipeline akan menghitung `color = BP−RP` dan `M_G` otomatis.

**Q:** Mengapa tidak bisa dibandingkan baris-per-baris dengan dataset valid?  
**A:** Dataset berasal dari sumber berbeda dan **tidak memiliki kunci identik**. Perbandingan dilakukan pada **statistik global** (mean parallax/PM, dll.) dan **overlay grafis**.

**Q:** Bagaimana mengubah ambang keanggotaan?  
**A:** Ubah `membership_prob_threshold` di `configs/default.yaml` lalu jalankan ulang langkah 2–4.
