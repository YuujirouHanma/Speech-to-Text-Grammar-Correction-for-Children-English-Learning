# SpeakFix

Layanan **transkripsi ucapan + koreksi grammar bahasa Inggris** untuk memantau pembelajaran anak.
Suara anak ditranskripsi oleh **Whisper** (lokal), lalu dikoreksi oleh model **T5** hasil fine-tuning.
Setiap respons berisi daftar perubahan per kata (*edits*) yang bisa langsung ditampilkan sebagai umpan balik.

Ini adalah versi produksi dari project riset tim (`T5PRETRAINASLI.ipynb`, `WHISPER.ipynb`, `GEMINI2.0.py`). File
riset, dataset, dan bobot model tidak ikut di repo ini. Path seperti `..\eval.csv` dan `..\saved_model` pada perintah
di bawah adalah lokasinya di laptop pengembang.

---

## Hasil: sebelum vs sesudah

Semua angka di bawah diukur ulang pada 1–2 November 2026 di laptop pengembang (Intel i7-13620H + RTX 4070
Laptop). Kolom SpeakFix memakai konfigurasi default: model v2 dengan filter keyakinan 2,5, di CPU. Script-nya
ada di repo ini (lihat [Evaluasi](#evaluasi-benchmark-dan-training)), dan laporan mentahnya ada di `reports/`.

| Aspek | Project awal | SpeakFix |
|---|---|---|
| JFLEG test (747 kalimat), BLEU 4 referensi | 0,653 | **0,871** |
| JFLEG test, GLEU (metrik resmi JFLEG) | 0,395 | **0,545** |
| JFLEG test, ERRANT F0.5 | 0,432 | **0,639** |
| JFLEG test, exact match (salah satu referensi) | 22,5% | **35,7%** |
| Data uji Kaggle (302 kalimat held-out), exact match | 23,2% | **40,7%** |
| Kalimat benar yang keliru diubah, JFLEG | 56,5% | **7,2%** |
| Kalimat benar yang keliru diubah, Kaggle | 11,6% | **1,0%** |
| Kalimat salah yang dibiarkan tanpa saran, Kaggle | 58,0% | **46,4%** |
| Kalimat salah yang dibiarkan tanpa saran, JFLEG | 20,6% (banyak outputnya terpotong) | 48,3% |
| Waktu generate per kalimat (p50, CPU) | 1.477 ms (PyTorch) | **262 ms** (CTranslate2) |
| Memori untuk generate | 1.677 MB | **279 MB** |
| Ukuran model | 895 MB | **228 MB** |
| Kesalahan yang tetap tertulis apa adanya oleh STT (15 rekaman) | 47–53% | 47–67% (CPU), **73%** (GPU) |
| Waktu training ulang | ±1 jam 45 menit (CPU) | **3 menit** (GPU) |
| Kebocoran data validasi ke data latih | 57–65% baris | **0** (dicek otomatis) |

Yang perlu dibaca bersama tabel ini:

- **Filter keyakinan sengaja menukar sebagian saran dengan lebih sedikit koreksi keliru.** Tanpa filter, v2
  mencapai BLEU JFLEG 0,875 dan exact match Kaggle 53,3%, tetapi 32,3% kalimat
  benar JFLEG ikut diubah. Di JFLEG, kalimat tanpa saran (48,3%) malah lebih banyak daripada pipeline
  lama. Lihat [Mengurangi koreksi keliru](#mengurangi-koreksi-keliru-filter-keyakinan).
- **Waktu dan memori di tabel hanya untuk langkah generate**, supaya adil dibandingkan dengan PyTorch.
  Layanan utuh (filter + tipe kesalahan ERRANT) butuh 371 ms per kalimat di CPU dan 230 ms
  di GPU. Server lengkap dengan Whisper memakai ±0,7 GB RAM di CPU, atau ±1,3 GB RAM + ±1 GB
  VRAM di GPU.
- **Data STT kecil** (15 rekaman, 1 rekaman = 6,7%) dan didominasi suara dewasa/TTS. Angka CPU berupa rentang
  karena berubah antar-run (lihat [Evaluasi speech-to-text](#evaluasi-speech-to-text)). Suara anak belum diuji.

Asal peningkatannya:

1. **Decoding diperbaiki** (model yang sama): kalimat diproses satu per satu, panjang output dinamis
   (dulu `max_length=20` memotong 57% referensi), dan tanda baca akhir ditambahkan sebelum inferensi
   (tanpa titik, model mengarang lanjutan seperti *"I am a student **and I am a teacher**"*).
   Hasilnya: BLEU JFLEG 0,653 → 0,856.
2. **Model dilatih ulang (v2)** dengan JFLEG + dataset Kaggle Grammar Correction, memakai split bebas
   kebocoran dan pasangan identitas. v2 sekarang memperbaiki kesalahan subject-verb agreement yang dulu
   lolos, misalnya *"He don't like apples"*, *"They is here today"*, dan *"She don't has"*.
   Hasilnya: BLEU JFLEG 0,856 → 0,875, exact match Kaggle 24,5% → 53,3%.
3. **Filter keyakinan**: saran hanya ditampilkan jika model cukup yakin. Kalimat benar yang keliru diubah
   turun dari 32,3% ke 7,2% (JFLEG) dan dari 6,3% ke 1,0% (Kaggle).
4. **Runtime CTranslate2 int8** menggantikan PyTorch untuk inferensi: generate 5,6× lebih cepat dan
   6,0× lebih hemat RAM. Untuk kalimat tunggal, CTranslate2 di CPU bahkan lebih cepat daripada PyTorch
   di GPU (262 ms vs 747 ms).
5. **Whisper + verbatim prompt**: dengan `small.en` di GPU, 73% rekaman tertulis apa adanya termasuk kesalahannya,
   sehingga kesalahan itu sampai ke model koreksi. Di CPU, efek prompt belum bisa dibedakan dari noise pada 15 rekaman
   ini, dan masih ada kesalahan yang dinormalkan Whisper (lihat [Evaluasi speech-to-text](#evaluasi-speech-to-text)).

---

## Arsitektur

```
                 HTTP (JSON / multipart)
                        │
┌───────────────────────▼──────────────────────────────────────────────┐
│ api/          FastAPI                                                 │
│   middleware  request-id · batas ukuran body · security header · metrik│
│   deps        API key · rate limit · consent orang tua                 │
│   routes      /v1/text  /v1/audio  /v1/jobs  /healthz /readyz /metrics │
└───────────────────────┬──────────────────────────────────────────────┘
                        │ run_inference(): thread pool + limiter + timeout + backpressure
┌───────────────────────▼──────────────────────────────────────────────┐
│ services/     use case (tanpa framework)                              │
│   AnalysisService ─► Transcriber (STT + fallback)                     │
│                  └─► GrammarCorrector (kalimat, batching, OOM, fallback)│
│   JobManager (antrean async, TTL, hasil terenkripsi) · privacy · ratelimit│
└───────────────────────┬──────────────────────────────────────────────┘
                        │ ports (Protocol): SpeechToText · Seq2SeqBackend · EditClassifier
┌───────────────────────▼──────────────────────────────────────────────┐
│ inference/    Whisper (faster-whisper) · T5 CTranslate2 | PyTorch      │
│               Gemini / Deepgram (opt-in)                               │
│ audio/        validasi magic bytes → decode PyAV 16 kHz di memori →    │
│               high-pass · normalisasi loudness · denoise opsional      │
│ text/         normalisasi · pemecah kalimat · edits (difflib | ERRANT) │
└──────────────────────────────────────────────────────────────────────┘
```

Alur satu request `/v1/audio/analyze`:

1. Cek API key, rate limit, dan consent. Batas ukuran body ditolak sebelum upload selesai dibaca.
2. Format divalidasi lewat *magic bytes*, bukan nama file. Audio di-decode ke 16 kHz mono **di memori**,
   dan decoding berhenti lebih awal jika audio terlalu panjang.
3. Preprocessing: high-pass 80 Hz, normalisasi loudness (penting untuk anak yang bicara pelan, karena
   VAD sensitif terhadap volume), lalu denoise opsional.
4. Whisper dijalankan dengan bahasa dipaksa `en`, VAD Silero, dan *verbatim prompt*.
5. Transkrip dipecah per kalimat lalu dikoreksi T5 dengan beam search. Postprocessing mencakup
   detokenisasi, pemangkasan kalimat karangan, *faithfulness guard*, dan filter keyakinan.
6. Edit per kata dihitung, dan respons JSON dikirim. Tidak ada data yang disimpan.

## Struktur folder

```
speakfix/
├── src/speakfix/
│   ├── api/          app factory, routes, schemas, middleware, deps, static UI
│   ├── services/     analysis, corrector, transcriber, jobs, privacy, ratelimit, container
│   ├── inference/    t5_ct2, t5_torch, whisper_stt, cloud_stt, devices, factory
│   ├── audio/        validation, decode, preprocess, encode
│   ├── text/         normalize, edits
│   ├── domain/       models, errors, ports
│   ├── config.py     semua pengaturan (env SPEAKFIX_*)
│   ├── logs.py       structlog + redaksi data anak
│   └── metrics.py    metrik Prometheus
├── scripts/          prepare_models, train_t5, evaluate_gec (+ gec_metrics), calibrate_correction, evaluate_stt,
│                     benchmark, load_test, check_api, check_browser, run_public_demo, backup_models,
│                     freeze_constraints
├── constraints.txt   versi dependensi yang sudah teruji (dipakai Docker, CI, dan setup lokal)
├── tests/            unit/ dan integration/ (185 test)
├── docker/           Dockerfile (CPU), Dockerfile.gpu (CUDA 12.8), model-card.md
├── Dockerfile.vercel image untuk Vercel, model ikut di dalamnya (+ vercel.json)
├── monitoring/       prometheus.yml, alerts.yml, dashboard Grafana
├── reports/          hasil evaluasi, benchmark, uji beban, uji rekaman, dan pemeriksaan sebelum testing
└── models/           artefak model (tidak di-commit)
```

---

## Menjalankan secara lokal

Prasyarat: Python 3.12 untuk versi terkunci (`constraints.txt`; Python 3.10–3.11 tetap bisa tanpa lock).
GPU tidak wajib.

```powershell
cd speakfix
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # Linux/macOS: source .venv/bin/activate
pip install -c constraints.txt -e ".[dev,torch,errant]"   # [torch]: konversi/training/fallback
python -m spacy download en_core_web_sm                     # model spaCy untuk tipe kesalahan (ERRANT)

# 1) Siapkan model
python scripts/prepare_models.py t5 --source ..\saved_model               # model asli  -> models/t5-gec(-ct2)
python scripts/prepare_models.py whisper --whisper base.en                 # -> models/whisper
python scripts/prepare_models.py whisper --whisper small.en                # dipakai otomatis jika ada GPU
# 2) (disarankan) latih ulang v2. Butuh GPU, ±3 menit (lihat bagian Training)

# 3) Jalankan
$env:HF_HUB_OFFLINE = "1"
python -m speakfix
```

Buka **http://localhost:8000** untuk UI demo (rekam, upload, atau ketik) atau **/docs** untuk
dokumentasi API interaktif. Model dimuat di background: `/healthz` langsung merespons 200, sedangkan
`/readyz` baru merespons 200 setelah model siap.

Kalau model v2 belum dibuat, pakai model asli: `$env:SPEAKFIX_T5_MODEL_NAME = "t5-gec"`.

> **GPU**: `SPEAKFIX_DEVICE=auto` hanya memilih GPU jika driver terdeteksi **dan** library CUDA
> (cuBLAS 12 + cuDNN 9) bisa dimuat. Jika tidak, CPU dipakai. Pemeriksaan ini penting karena
> faster-whisper sempat *hang* (bukan error) di Windows tanpa cuBLAS.
>
> Di Windows tidak perlu memasang CUDA Toolkit. Jika cuBLAS tidak ada di sistem, SpeakFix memakai
> salinan yang dibawa wheel PyTorch CUDA (`torch/lib`) atau wheel `nvidia-cublas-cu12` dan
> `nvidia-cudnn-cu12`, lalu menambahkan foldernya ke `PATH` proses. Wheel PyTorch dari PyPI hanya
> versi CPU, jadi pasang versi CUDA:
> `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
> Jalur PyTorch sudah diuji di RTX 4070 Laptop: Whisper base.en 6× lebih cepat dan T5 3× lebih
> cepat untuk paragraf (lihat Benchmark). Jalur wheel `nvidia-*` belum diuji. Untuk GPU produksi,
> gunakan image Docker Linux.

---

## API

| Method | Path | Fungsi |
|---|---|---|
| POST | `/v1/text/correct` | Koreksi teks (JSON) |
| POST | `/v1/audio/transcribe` | Transkripsi saja (multipart) |
| POST | `/v1/audio/analyze` | Transkripsi + koreksi (multipart, sinkron) |
| POST | `/v1/jobs/analyze` | Sama seperti analyze, asinkron → `202` + `Location` |
| GET / DELETE | `/v1/jobs/{job_id}` | Ambil status/hasil · hapus segera |
| GET | `/v1/info` | Model aktif, batas, pengaturan privasi |
| POST | `/v1/warmup` | Memuat model sekarang, untuk host serverless (lihat Vercel) |
| GET | `/healthz`, `/readyz`, `/metrics` | Liveness, readiness, Prometheus |

Contoh:

```bash
curl -s localhost:8000/v1/text/correct -H "Content-Type: application/json" \
     -H "X-API-Key: $KEY" -d '{"text": "He don'"'"'t like apples. They is happy"}'

curl -s localhost:8000/v1/audio/analyze -H "X-API-Key: $KEY" \
     -F "file=@rekaman.wav" -F "consent=true"
```

Contoh respons `/v1/text/correct`:

```json
{
  "request_id": "5f0c…",
  "correction": {
    "original": "He don't like apples. They is happy",
    "corrected": "He doesn't like apples. They are happy.",
    "changed": true,
    "edits": [
      {"op": "replace", "original": "do", "corrected": "does", "start": 3, "end": 5,
       "error_type": "R:VERB:SVA", "description": "Kata kerja harus sesuai dengan subjek"},
      {"op": "replace", "original": "is", "corrected": "are", "start": 27, "end": 29,
       "error_type": "R:VERB:SVA", "description": "Kata kerja harus sesuai dengan subjek"},
      {"op": "insert", "original": "", "corrected": ".", "start": 35, "end": 35,
       "error_type": "M:PUNCT", "description": "Kurang tanda baca"}
    ],
    "sentences": 2,
    "backend": "ctranslate2:int8_float32@cuda",
    "warnings": []
  },
  "timings_ms": {"total_ms": 152.4}
}
```

`description` memakai Bahasa Indonesia secara default (`SPEAKFIX_FEEDBACK_LANGUAGE=en` untuk Inggris).
Contoh di atas adalah teks ketikan, jadi titik yang hilang ikut dikoreksi. Untuk rekaman suara, perubahan
yang hanya menyangkut tanda baca atau huruf kapital tidak dilaporkan, karena tanda baca transkrip berasal
dari Whisper, bukan dari anak.

`error_type` (kode ERRANT) dan `description` hanya terisi jika fitur ERRANT terpasang:
`pip install -e ".[errant]"` lalu `python -m spacy download en_core_web_sm`. Untuk Docker, build dengan
`--build-arg EXTRAS=errant`, dan model spaCy akan ikut diunduh. Tanpa ERRANT, `edits` tetap ada (diff per
kata), tetapi kedua field itu bernilai `null`. Contohnya, *don't → doesn't* menjadi satu edit tanpa tipe.

Semua error memakai format yang sama:
`{"error": {"code": "consent_required", "message": "...", "request_id": "..."}}`.
Kode status yang dipakai: 401 (API key), 403 (consent / cloud STT dimatikan), 413 (file terlalu besar),
415 (bukan audio), 422 (audio rusak, durasi tidak valid, atau teks kosong), 429 (rate limit, disertai
`Retry-After`), 503 (model belum siap, server sibuk, atau OOM) dan 504 (timeout).
Input pengguna tidak pernah digemakan kembali di pesan error.

Warning yang mungkin muncul di respons:

- **Kualitas audio**: `very_quiet_audio`, `little_or_no_speech_detected`, `clipping_detected`.
- **Transkrip/koreksi**: `empty_transcript`, `correction_unavailable` (koreksi gagal, tapi transkrip
  tetap dikirim).
- **Model**: `trimmed_extra_model_output`, `low_confidence_correction_skipped`,
  `uncertain_correction_skipped` (filter keyakinan), `empty_model_output`.

---

## Konfigurasi

Semua pengaturan dibaca dari environment variable berprefix `SPEAKFIX_` atau dari file `.env`
(lihat `.env.example`). Daftar lengkapnya ada di `src/speakfix/config.py`. Yang terpenting:

| Variabel | Default | Keterangan |
|---|---|---|
| `SPEAKFIX_API_KEYS` | *(kosong)* | Daftar key dipisah koma. Kosong berarti tanpa autentikasi (hanya untuk lokal). |
| `SPEAKFIX_REQUIRE_CONSENT` | `true` | Endpoint audio mewajibkan `consent=true`. |
| `SPEAKFIX_RATE_LIMIT_PER_MINUTE` | `60` | Batas request per key/IP (`0` = mati). Satu kelas di satu Wi-Fi berbagi satu IP, jadi demo publik memakai `600`. |
| `SPEAKFIX_CORS_ORIGINS` | *(kosong)* | Origin browser yang diizinkan. |
| `SPEAKFIX_T5_MODEL_NAME` | `t5-gec-v2` | Memakai `models/<nama>` dan `models/<nama>-ct2`. |
| `SPEAKFIX_CORRECTOR_BACKEND` | `ctranslate2` | `ctranslate2` atau `torch`. |
| `SPEAKFIX_CORRECTOR_FALLBACK_BACKEND` | `torch` | Dipakai bila backend utama gagal/OOM (`none` = mati). |
| `SPEAKFIX_DEVICE` | `auto` | `cpu`, `cuda`, atau `auto`. |
| `SPEAKFIX_WHISPER_MODEL` | `auto` | `small.en` di GPU jika sudah disiapkan, selain itu `base.en` (lihat Evaluasi speech-to-text). Bisa juga ukuran lain atau path lokal. |
| `SPEAKFIX_WHISPER_VERBATIM_PROMPT` | `true` | Menjaga kesalahan grammar tetap tertulis. |
| `SPEAKFIX_AUDIO_DENOISE` | `false` | Spectral gating, hanya untuk ruangan bising. |
| `SPEAKFIX_MAX_UPLOAD_MB` / `SPEAKFIX_MAX_AUDIO_SECONDS` | `10` / `60` | Batas input. |
| `SPEAKFIX_INFERENCE_WORKERS` | `1` | Inferensi paralel. Gunakan `2` di GPU (lihat Uji beban). |
| `SPEAKFIX_MAX_PENDING_REQUESTS` | `16` | Lebih dari ini akan dijawab 503 (backpressure). Demo publik memakai `32`, supaya satu kelas bisa menekan tombol bersamaan. |
| `SPEAKFIX_JOB_TTL_SECONDS` | `600` | Masa hidup hasil job async. |
| `SPEAKFIX_ALLOW_CLOUD_STT` | `false` | Mengizinkan Gemini/Deepgram (baca bagian Privasi). |
| `SPEAKFIX_MIN_CORRECTION_MARGIN` | `2.5` | Saran hanya ditampilkan jika model cukup yakin (lihat *Mengurangi koreksi keliru*). `1.0` = lebih banyak saran tetapi lebih banyak koreksi keliru; `-inf` = filter mati. |
| `SPEAKFIX_PUNCTUATION_FEEDBACK` | `true` | Umpan balik tanda baca dan huruf kapital untuk teks ketikan. Transkrip suara tidak pernah mendapatkannya. |
| `SPEAKFIX_FEEDBACK_LANGUAGE` | `id` | Bahasa deskripsi kesalahan di API (`id` atau `en`). |
| `SPEAKFIX_DEMO_MODE` | `false` | Demo publik: UI meminta suara orang dewasa saja. |
| `SPEAKFIX_LOAD_MODELS_ON_DEMAND` | `false` | Memuat model saat dibutuhkan, untuk host serverless (lihat Vercel). |

---

## Evaluasi, benchmark, dan training

### Training ulang model (v2)

```powershell
python scripts/train_t5.py --base models/t5-gec --out models/t5-gec-v2 `
    --jfleg-train ..\train.csv --jfleg-test ..\eval.csv `
    --kaggle "..\..\FP\Grammar Correction.csv" --epochs 4
python scripts/prepare_models.py t5 --source models/t5-gec-v2 --t5-name t5-gec-v2
```

Script ini memperbaiki masalah metodologi pada eksperimen awal:

- Duplikat dibuang **sebelum** split, dan split dibuat per kalimat sumber. Script gagal (assert) jika
  ada kalimat validasi/test yang bocor ke data latih.
- JFLEG test dan potongan test Kaggle tidak pernah dipakai untuk training maupun pemilihan model.
- Seed tetap, dan manifest split disimpan di `models/t5-gec-v2/split.json`.
- Pasangan identitas (kalimat benar → tetap) mencegah koreksi berlebihan.
- Teks JFLEG dide-tokenisasi supaya mirip input nyata.

Data tambahan (misalnya file Skenario 3 terbaru) cukup ditambahkan dengan
`--extra "path/ke/file.csv"` (kolom `input`, `target`). Baris yang tumpang tindih dengan data evaluasi
dibuang otomatis.

Hasil v2: 4.309 pasangan latih, eval loss terbaik 0,307 (epoch 2), 186 detik di RTX 4070.

### Evaluasi koreksi grammar

```powershell
python scripts/evaluate_gec.py --model models/t5-gec-v2-ct2 --jfleg ..\eval.csv `
    --kaggle "..\..\FP\Grammar Correction.csv" --kaggle-split models/t5-gec-v2/split.json `
    --systems speakfix --min-margin 2.5 --correct-sentences --gec-metrics --label v2-margin2.5-text
```

Tanpa `--min-margin`, filter keyakinan mati (label `v2-baseline`). `--systems legacy` menjalankan pipeline
notebook lama pada model yang sama. Semua hasil, termasuk 25 contoh output per data, ada di
`reports/gec_eval.json`.

| Sistem | Data | BLEU 1-ref | BLEU 4-ref | Exact match | Tidak diubah |
|---|---|---|---|---|---|
| Pipeline lama (notebook), model asli | JFLEG test (747) | 0,519 | 0,653 | 22,5% | 20,6% |
| SpeakFix tanpa filter, model asli | JFLEG test | 0,711 | 0,856 | 33,6% | 35,7% |
| SpeakFix tanpa filter, model v2 | JFLEG test | 0,727 | 0,875 | 36,4% | 25,8% |
| **SpeakFix default (v2 + filter 2,5)** | JFLEG test | **0,723** | **0,871** | **35,7%** | 48,3% |
| Pipeline lama, model asli | Kaggle held-out (302) | 0,688 | – | 23,2% | 58,0% |
| SpeakFix tanpa filter, model asli | Kaggle held-out | 0,685 | – | 24,5% | 59,9% |
| SpeakFix tanpa filter, model v2 | Kaggle held-out | 0,815 | – | 53,3% | 23,5% |
| **SpeakFix default (v2 + filter 2,5)** | Kaggle held-out | **0,757** | – | **40,7%** | 46,4% |

Cara membaca tabel:

- Semua skor dihitung pada teks yang sudah dide-tokenisasi. Pipeline lama mendapat 0,519 di sini,
  sedangkan di notebook 0,497 karena memakai versi tokenized dan implementasi BLEU HuggingFace.
- Faithfulness guard hanya membatalkan 2 dari 747 koreksi JFLEG dan 0 dari 302 koreksi Kaggle.
- Data uji Kaggle berisi 302 kalimat. Evaluasi sebelum 1 November hanya menemukan 300 karena pemuat data
  membandingkan teks mentah dengan teks yang sudah dibersihkan, dan membaca `split.json` dengan encoding
  yang salah. Sekarang pemuat membersihkan teks dengan cara yang sama seperti saat training, dan script
  berhenti jika ada kalimat yang hilang. Ada test untuk ini.
- Gaya input "raw" setara dengan gaya tokenisasi JFLEG, sehingga yang dipakai adalah raw.

#### Metrik GEC standar: GLEU dan ERRANT F0.5

Dengan `--gec-metrics` (butuh fitur `errant` dan `en_core_web_sm`), script juga menghitung GLEU (metrik
resmi JFLEG) dan ERRANT F0.5 (metrik standar BEA/CoNLL). Implementasinya ada di `scripts/gec_metrics.py`.
Pada output yang sama, hasilnya identik dengan skrip resmi `gleu.py` (repo keisks/jfleg) dan
`errant_parallel` + `errant_compare`, dan ini juga dikunci dengan unit test.

| Sistem | JFLEG GLEU | JFLEG ERRANT P / R / F0.5 | Kaggle GLEU | Kaggle ERRANT F0.5 |
|---|---|---|---|---|
| Tanpa koreksi (output = input) | 0,407 | – / 0 / 0 | 0,315 | 0 |
| Pipeline lama (notebook), model asli | 0,395 | 0,473 / 0,321 / 0,432 | 0,491 | 0,473 |
| SpeakFix tanpa filter, model asli | 0,527 | 0,714 / 0,387 / 0,611 | 0,485 | 0,506 |
| SpeakFix tanpa filter, model v2 | 0,571 | 0,713 / 0,487 / 0,652 | 0,727 | 0,694 |
| **SpeakFix default (v2 + filter 2,5)** | **0,545** | **0,732** / 0,424 / **0,639** | **0,612** | **0,660** |

- Teks ditokenisasi dengan spaCy. Baseline "tanpa koreksi" mendapat 0,407, hampir sama dengan baseline
  SOURCE resmi di leaderboard JFLEG (40,54). Jadi tokenisasi ini cukup dekat dengan yang resmi.
- Pipeline lama mendapat GLEU **di bawah** baseline tanpa koreksi (0,395 < 0,407). Secara rata-rata,
  outputnya lebih buruk daripada membiarkan kalimat apa adanya, salah satunya karena 10,8% output
  terpotong di 20 token.
- Sebagai konteks, leaderboard JFLEG mencatat REFERENCE (manusia) 62,37 dan sistem 2017–2018 di kisaran
  54–62 (misalnya Chollampatt & Ng 2018: 57,47). v2 berada di kisaran itu, tetapi angkanya tidak bisa
  dibandingkan langsung. Tokenisasinya sedikit berbeda, dan v2 dilatih dengan JFLEG dev (`train.csv`),
  yang pada sistem di leaderboard biasanya hanya dipakai untuk tuning.
- Pada data uji ini, peningkatan v2 terutama ada di recall (0,387 → 0,487), dengan presisi tetap sekitar
  0,713. Tetapi data ini hanya berisi kalimat yang perlu dikoreksi, jadi tidak menunjukkan koreksi keliru
  pada kalimat benar. Bagian filter di bawah mengukurnya.

Model v2 yang sama di GPU, tanpa filter (`--device cuda --compute-type ...`, label `v2-gpu-*` di
`reports/gec_eval.json`):

| Compute type CTranslate2 | JFLEG BLEU 4-ref | Kaggle exact match | p50 per kalimat (150 kalimat JFLEG) |
|---|---|---|---|
| int8_float32, CPU (acuan di atas) | 0,875 | 53,3% | 280 ms |
| **int8_float32, GPU (default)** | **0,872** | **52,6%** | **198 ms** |
| float32, GPU | 0,872 | 52,3% | 144 ms |
| int8_bfloat16, GPU (default lama) | 0,867 | 52,3% | 192 ms |

Selisih akurasi ini kecil (1 kalimat Kaggle = 0,33%), tetapi BLEU bfloat16 paling rendah, jadi bfloat16 tidak lagi
dipakai. float32 lebih cepat, tetapi model T5 memakai ±1,2 GB VRAM, sedangkan `int8_float32` ±0,4 GB. Default GPU tetap
`int8_float32`; jika VRAM cukup, ganti lewat `SPEAKFIX_CT2_COMPUTE_TYPE=float32`. Hasil CPU dan GPU tidak identik karena
pembulatan presisi berbeda.

#### Mengurangi koreksi keliru (filter keyakinan)

Tanpa filter, model v2 masih mengubah sebagian kalimat yang **sudah benar**, misalnya *"The group of
volunteers works tirelessly to help the community."* diubah menjadi *"work"*. Untuk anak, saran yang keliru
lebih merugikan daripada kesalahan yang terlewat. Karena itu ada dua pengaman:

1. **Filter keyakinan** (`SPEAKFIX_MIN_CORRECTION_MARGIN`, default `2.5`). Model memberi skor
   log-probabilitas pada saran koreksinya dan pada kalimat asli. Saran hanya ditampilkan jika selisihnya
   mencapai ambang. Skor dihitung dalam satu *forward pass* tambahan, hanya untuk kalimat yang berubah.
   Saran yang ditahan diberi catatan `uncertain_correction_skipped`.
2. **Tanpa umpan balik tanda baca untuk suara.** Tanda baca dan huruf kapital dalam transkrip berasal
   dari Whisper, bukan dari anak. Perubahan yang hanya menyangkut keduanya diabaikan, begitu pula titik
   yang ditambahkan sistem untuk model. Teks ketikan tetap mendapat umpan balik tanda baca.

Ambang dikalibrasi di **data validasi**, bukan di data uji
(`python scripts/calibrate_correction.py --jfleg-train ..\train.csv --jfleg-test ..\eval.csv --kaggle "..\..\FP\Grammar Correction.csv"`).
Data validasinya adalah 238 kalimat salah dan 328 kalimat benar yang tidak pernah dipakai untuk
training. Skrip memeriksa bahwa split yang direkonstruksi identik dengan `split.json`. Hasil lengkapnya ada
di `reports/correction_calibration.json`, dan menjalankan ulang skrip ini pada 2 November menghasilkan angka
yang sama. ERRANT di tabel ini tidak menghitung edit tanda baca.

| Ambang | Presisi | Recall | F0.5 | Kalimat benar diubah | Kalimat salah tanpa saran |
|---|---|---|---|---|---|
| tanpa filter | 0,598 | 0,524 | 0,582 | 15,2% | 24,4% |
| 1,0 | 0,656 | 0,487 | 0,613 | 6,7% | 33,6% |
| 2,0 | 0,686 | 0,463 | 0,626 | 4,9% | 40,3% |
| **2,5 (default)** | **0,715** | 0,447 | **0,638** | **3,0%** | 44,5% |
| 4,0 | 0,744 | 0,381 | 0,625 | 1,8% | 55,5% |

Hasil di **data uji** (model v2; label `v2-baseline`, `v2-margin2.5-text`, `v2-margin2.5-speech`, dan
`v2-margin2.5-text-gpu` di `reports/gec_eval.json`). Kalimat benar diukur dengan memasukkan referensi pertama
setiap kalimat sebagai input (`--correct-sentences`):

| Konfigurasi | Kalimat benar Kaggle diubah | Kalimat benar JFLEG diubah | ERRANT P / R / F0.5 Kaggle | ERRANT P / R / F0.5 JFLEG | Kalimat salah Kaggle tanpa saran |
|---|---|---|---|---|---|
| Tanpa filter | 6,3% | 32,3% | 0,739 / 0,559 / 0,694 | 0,713 / 0,487 / 0,652 | 23,5% |
| **Ambang 2,5, teks ketikan (default)** | **1,0%** | **7,2%** | **0,778** / 0,411 / 0,660 | **0,732** / 0,424 / 0,639 | 46,4% |
| Ambang 2,5, teks ketikan, GPU | 1,0% | 7,5% | 0,767 / 0,418 / 0,657 | 0,725 / 0,423 / 0,634 | 45,7% |
| Ambang 2,5, gaya transkrip suara | 1,0% | 7,0% | 0,782 / 0,354 / 0,630 | 0,720 / 0,404 / 0,622 | 51,3% |

Cara membaca:

- Koreksi keliru pada kalimat benar turun **6,4×** (Kaggle) dan **4,5×** (JFLEG), dan presisi naik.
  Bayarannya, sekitar separuh kalimat salah di data uji dibiarkan tanpa saran.
- F0.5 di data uji sedikit turun karena data ujinya hanya berisi kalimat salah, sehingga manfaat filter
  pada kalimat benar tidak ikut terhitung. Pada campuran kalimat salah dan benar (data validasi), F0.5
  naik dari 0,582 ke 0,638.
- Contoh yang tetap dikoreksi: *"My mother cook rice every morning."* (margin 7,8) dan *"I have went to Bali."*
  (margin 4,9). Yang tertahan: *"I am agree with your opinion."* (margin 1,13). Hasil rekaman suara
  ada di [Uji nyata](#uji-nyata-server-sungguhan-laptop).
- Jika tim lebih memilih lebih banyak saran, pakai `SPEAKFIX_MIN_CORRECTION_MARGIN=1.0`. Jika ingin
  perilaku lama tanpa filter, pakai `-inf`.

### Evaluasi speech-to-text

```powershell
python scripts/evaluate_stt.py --audio-dir ..\..\FP --models base.en small.en `
    --extra "..\recorded.wav=She can sings well." "..\..\FP\recorded.wav=She go to school." `
    --device cpu --out reports/stt_eval.md          # --device cuda untuk GPU (reports/stt_eval_gpu.md)
```

Dataset evaluasi berisi 15 rekaman: 13 mp3 kalimat salah dari folder FP dan 2 rekaman suara asli.
"Kesalahan tetap tertulis" berarti transkrip sama persis dengan kalimat yang diucapkan, termasuk kesalahannya.

| Perangkat | Model | Verbatim prompt | WER | Kesalahan tetap tertulis | Real-time factor |
|---|---|---|---|---|---|
| CPU int8 | base.en | tidak | 0,050–0,061 | 47–53% | 0,151 |
| **CPU int8 (default CPU)** | **base.en** | **ya** | **0,039–0,056** | **47–67%** | **0,172** |
| CPU int8 | small.en | tidak | 0,039–0,044 | 53–60% | 0,542 |
| CPU int8 | small.en | ya | 0,022–0,039 | 53–73% | 0,551 |
| GPU float16 | base.en | tidak | 0,050 | 53% | 0,022 |
| GPU float16 | base.en | ya | 0,056 | 53% | 0,026 |
| GPU float16 | small.en | tidak | 0,039 | 60% | 0,039 |
| **GPU float16 (default GPU)** | **small.en** | **ya** | **0,022** | **73%** | **0,042** |

**Angka CPU berupa rentang dari lima run.** Rekaman dan modelnya sama, hanya jumlah thread OpenBLAS yang diubah (1, 2,
4, 8, 16). Ini hanya mengubah pembulatan spektrogram mel (selisih terbesar sekitar 1e-7), tetapi sudah mengubah sampai
4 dari 15 transkrip di CPU. Di GPU, kelima run menghasilkan transkrip yang persis sama, dan pada jumlah thread
yang sama hasil CPU juga selalu sama. Mengapa model int8 di CPU sesensitif ini belum ditelusuri. Service berjalan dengan
1 thread, yang pada base.en kebetulan berada di ujung bawah rentang (47%), jadi jangan membandingkan satu angka
CPU antar-versi. Rinciannya ada di `reports/stt_eval.md`.

`SPEAKFIX_WHISPER_MODEL=auto` memilih:

- **`small.en` di GPU**: hasil terbaik dan stabil, dan tetap cepat (0,16 s untuk audio 5 detik).
- **`base.en` di CPU**: rentang hasilnya bertumpang tindih dengan `small.en`, tetapi jauh lebih cepat (0,43 s vs 2,58 s).
  Jika waktu tunggu tidak masalah, `small.en` juga bisa dipakai di CPU.

Jika `small.en` belum disiapkan, GPU tetap memakai `base.en`.

Yang perlu diingat saat membaca tabel ini:

- **Datanya kecil.** 1 rekaman = 6,7%, jadi selisih antar baris hanya beberapa rekaman. Data ini juga didominasi
  suara dewasa/TTS. Sebelum dipakai ke anak-anak sungguhan, kumpulkan rekaman anak (dengan consent) dan ulangi
  evaluasi ini.
- **Whisper tidak selalu menulis kesalahan apa adanya.** *"He readed the book"* selalu menjadi *"He read the book."*,
  karena *readed* tidak ada di kamus. Dua rekaman suara asli juga belum tertangkap sempurna di GPU: `small.en`
  menulis *"She can sing well."* untuk *"She can sings well."*, dan `base.en` menulis *"She goes to school."* untuk
  *"She go to school."*.
- **Whisper juga bisa salah dengar.** Di GPU, *"mistakes is happens"* terdengar *"mistakes as happens"*, dan saran
  untuk kalimat itu tertahan oleh filter keyakinan.
- `reports/stt_eval_gpu.md` juga mencatat compute type GPU lain. Hasilnya berbeda-beda (base.en dengan
  int8_float16 turun ke 33%), jadi default GPU tetap `float16`.

### Benchmark latensi

```powershell
python scripts/benchmark.py --backends ct2-cpu ct2-cuda torch-cpu torch-cuda `
    --whisper base.en small.en --whisper-devices cpu cuda --audio ..\recorded.wav
python scripts/benchmark.py --service-config --backends ct2-cpu ct2-cuda    # + filter dan ERRANT
```

Run 2 November 2026, model v2, 50 kalimat JFLEG, beam 5. Satu pekerjaan dalam satu waktu; browser dan editor
tetap terbuka seperti pemakaian sehari-hari. Angka absolut bergantung pada beban laptop, jadi bandingkan di
dalam satu tabel.

| Runtime, generate saja | p50 | p95 | Paragraf 8 kalimat | Tambahan RAM | Waktu load |
|---|---|---|---|---|---|
| **CTranslate2 int8_float32, CPU** | **262 ms** | **668 ms** | 2,14 s | **279 MB** | **0,4 s** |
| CTranslate2 int8_float32, GPU | 210 ms | 538 ms | **0,63 s** | 485 MB | 0,6 s |
| PyTorch float32, CPU | 1.477 ms | 4.093 ms | 17,31 s | 1.677 MB | 6,0 s |
| PyTorch bfloat16, GPU | 747 ms | 1.962 ms | 2,26 s | 893 MB | 1,1 s |

| Layanan utuh (filter 2,5 + ERRANT) | p50 | p95 | Paragraf 8 kalimat | Waktu load |
|---|---|---|---|---|
| CPU | 371 ms | 1.194 ms | 3,40 s | 2,7 s |
| GPU | 230 ms | 586 ms | 0,90 s | 0,9 s |

| Whisper (audio 5 s) | CPU int8 | GPU float16 |
|---|---|---|
| base.en | 0,43 s | 0,07 s |
| small.en | 2,58 s | 0,16 s |

Untuk kalimat tunggal, GPU hampir tidak lebih cepat karena beam search T5 berjalan token demi token. GPU baru
unggul untuk input panjang dan untuk Whisper. Detail lengkap ada di `reports/benchmark.md`.

### Uji beban (berapa banyak pengguna bersamaan?)

```powershell
python -m speakfix                  # di terminal lain, dengan pengaturan yang ingin diuji
python scripts/load_test.py --users 30 60 --seconds 60 --think 15 --audio ..\recorded.wav
python scripts/load_test.py --users 30 --seconds 120 --think 15 --audio ..\recorded.wav --same-ip
python scripts/load_test.py --users 30 --audio ..\recorded.wav --same-ip --burst   # menekan tombol bersamaan
```

Setiap pengguna virtual mengirim rekaman 5 detik, menunggu hasilnya, berhenti sejenak (*think time*), lalu
mengulang. Masing-masing memakai IP berbeda lewat `X-Forwarded-For`, kecuali `--same-ip` yang meniru satu kelas
di belakang satu Wi-Fi. Hasil di RTX 4070 Laptop, 2 November 2026, satu konfigurasi dalam satu waktu:

| Skenario | Worker | Antrean | Rate limit/menit | Berhasil | Throughput | p50 | p95 |
|---|---|---|---|---|---|---|---|
| **Konfigurasi demo (GPU, `small.en`)** | | | | | | | |
| 30 pengguna, jeda ±15 s (tempo wajar) | 2 | 32 | 600 | 100% | 2,02/s | 0,35 s | 0,64 s |
| 30 pengguna, jeda ±15 s, satu Wi-Fi | 2 | 32 | 600 | 100% | 1,99/s | 0,37 s | 0,66 s |
| 60 pengguna, jeda ±15 s | 2 | 32 | 600 | 100% | 3,85/s | 0,53 s | 1,47 s |
| 30 pengguna, jeda ±3 s (agresif) | 2 | 32 | 600 | 100% | 5,51/s | 2,66 s | 3,28 s |
| **1 worker (GPU, `small.en`)** | | | | | | | |
| 30 pengguna, jeda ±15 s | 1 | 32 | 600 | 100% | 2,10/s | 0,30 s | 0,71 s |
| 60 pengguna, jeda ±15 s | 1 | 32 | 600 | 100% | 3,77/s | 1,42 s | 2,52 s |
| 30 pengguna, jeda ±3 s (agresif) | 1 | 32 | 600 | 100% | 4,22/s | 4,75 s | 5,52 s |
| **Pengaturan default (GPU, `small.en`)** | | | | | | | |
| 30 pengguna, jeda ±15 s | 1 | 16 | 60 | 100% | 1,97/s | 0,43 s | 0,92 s |
| 30 pengguna, jeda ±10 s, satu Wi-Fi | 1 | 16 | 60 | 66,5% | 1,98/s | 0,47 s | 1,15 s |
| **Isi image Docker (CPU, `base.en`)** | | | | | | | |
| 5 pengguna, jeda ±15 s | 1 | 16 | 60 | 100% | 0,32/s | 1,32 s | 2,33 s |
| 10 pengguna, jeda ±15 s | 1 | 16 | 60 | 100% | 0,63/s | 1,45 s | 2,93 s |
| 20 pengguna, jeda ±15 s | 1 | 16 | 60 | 100% | 1,00/s | 4,88 s | 8,37 s |

Membaca hasilnya:

- **Satu laptop dengan GPU cukup untuk satu kelas.** Dengan konfigurasi demo, 60 pengguna dengan jeda
  ±15 detik semuanya terlayani (p95 1,47 s), termasuk 30 pengguna di satu Wi-Fi. Saat
  30 orang menekan tombol bersamaan, semuanya mendapat jawaban: separuhnya dalam ±4 detik, paling
  lambat 6 detik.
- **Pakai 2 worker di GPU.** Kerja CPU satu request (decode audio, tokenisasi, ERRANT) bisa berjalan bersamaan
  dengan kerja GPU request lain. Dengan jeda ±3 detik, 2 worker melayani 5,5/detik (p50 2,7 s),
  sedangkan 1 worker 4,2/detik (p50 4,8 s). Jawaban tetap identik dengan request berurutan, termasuk
  tipe kesalahannya.
- **Di luar kapasitas, server tetap menjawab dengan cepat.** Permintaan yang melewati antrean dijawab 503 *"server
  sibuk"*, dan server tidak macet.
- **Rate limit default terlalu ketat untuk satu kelas.** Batas 60/menit dihitung per IP, sedangkan satu kelas di
  satu Wi-Fi berbagi satu IP. Dengan default, hanya 66,5% permintaan yang berhasil pada skenario di atas.
  Demo publik memakai 600/menit.
- **Tanpa GPU**, kapasitasnya jauh lebih kecil: 10 pengguna masih nyaman (p95 2,9 s), sedangkan
  20 pengguna mulai menunggu lama (p95 8,4 s).

Selama uji ini server memakai paling banyak 31 thread (sebelum perbaikan decoder: 944) dan log tidak
memuat isi kalimat. Hasil lengkap ada di `reports/load_test.md`.

> **Windows: perhatikan ruang drive C:.** Server memakai ±3–3,5 GB *commit memory*, meskipun RAM yang benar-benar
> terpakai lebih kecil. Jika drive C: penuh, Windows tidak bisa memperbesar page file, dan request bisa gagal dengan
> pesan "server sedang penuh" (503). Ini sempat terjadi saat pengujian ketika C: tersisa 0,2 GB. Skrip demo memberi
> peringatan jika commit memory yang bebas di bawah 6 GB.

---

## Deployment

### Demo publik dari laptop (paling cepat, gratis)

```powershell
cd speakfix
powershell -ExecutionPolicy Bypass -File scripts\run_public_demo.ps1
```

Skrip ini:
- menolak berjalan jika port 8000 sudah dipakai program lain, supaya tunnel tidak membuka server yang salah;
- memperingatkan jika *commit memory* yang bebas di bawah 6 GB;
- menjalankan SpeakFix di `127.0.0.1` (tidak bisa diakses langsung dari Wi-Fi), dengan GPU, 2 worker, rate limit
  600/menit per IP, antrean 32, dan **mode demo** (banner "gunakan suara orang dewasa", teks persetujuan untuk
  orang dewasa);
- membuka *Cloudflare Quick Tunnel* dan menampilkan link `https://<acak>.trycloudflare.com` yang bisa
  dibuka dari HP atau laptop mana pun. HTTPS membuat mikrofon browser bisa dipakai;
- saat pertama dijalankan, mengunduh `cloudflared.exe` resmi ke `.tools/` dan memeriksa tanda tangan
  digital Cloudflare-nya;
- menghentikan server dan tunnel saat Anda menekan **Ctrl+C**.

Sudah diuji 2 November 2026 lewat link publik: semua endpoint dan jalur error, 15 rekaman (median
458 ms), serta UI di Edge (52/52), Firefox (53/53), dan WebKit (43/43), termasuk merekam
dari mikrofon tiruan di Edge. Setelah Ctrl+C, link berhenti berfungsi. Tunnel meneruskan IP asli pengunjung
(`X-Forwarded-For`), sehingga rate limit berlaku per pengunjung, dan log server tidak memuat isi kalimat maupun IP.

Yang belum diuji: perekaman dari HP sungguhan (iPhone/Safari dan Android/Chrome). Server sudah menerima format
yang dipakai browser-browser itu (webm/Opus, MP4 terfragmentasi/AAC), tetapi mikrofon di HP perlu dicoba langsung.

Batasan: link berganti setiap kali skrip dijalankan, laptop harus tetap menyala, dan *quick tunnel* tidak
punya jaminan uptime. `/docs` dan `/metrics` (statistik teknis tanpa data pribadi) ikut terbuka. Untuk
layanan permanen, gunakan Vercel atau Docker di server (di bawah).

### Vercel (link permanen gratis, untuk portofolio)

Untuk pengunjung sedikit dan tanpa server sendiri: laptop boleh mati, dan linknya tetap. `Dockerfile.vercel` membuat
image berisi model v2 dan Whisper `base.en`, dan `vercel.json` memakai runtime container Vercel (masih beta).

1. Unggah model ke Hugging Face. Repo-nya harus publik, karena Vercel mengunduhnya tanpa token saat build:

   ```powershell
   .\.venv\Scripts\hf.exe auth login
   .\.venv\Scripts\hf.exe upload <user>/speakfix-t5-gec-v2-ct2 models\t5-gec-v2-ct2 .
   .\.venv\Scripts\hf.exe upload <user>/speakfix-t5-gec-v2-ct2 docker\model-card.md README.md
   ```

   Lisensinya CC BY-NC-SA 4.0, mengikuti lisensi JFLEG (lihat `docker/model-card.md`). Jika nama repo berbeda, ubah
   `T5_MODEL_REPO` di `Dockerfile.vercel`.
2. Di Vercel, pilih **Add New > Project**, lalu impor repo GitHub ini.
3. Tambahkan environment variable `PORT` dengan nilai `8000`, lalu klik **Deploy**.

Bedanya dengan server biasa:

- **Model dimuat saat dibutuhkan** (`SPEAKFIX_LOAD_MODELS_ON_DEMAND=true`). Vercel menjeda server setelah request
  selesai, jadi pemuatan di background bisa terhenti. Request yang sedang menunggu membuat server tetap berjalan.
- **Halaman mulai memuat model sejak dibuka.** Halaman langsung memanggil `POST /v1/warmup`, jadi waktu tunggunya
  terpakai selagi pengunjung membaca atau merekam. Kalau satu permintaan lebih dari 8 detik, halaman menjelaskan
  alasannya.
- **Batas input lebih kecil.** Vercel menolak request di atas 4,5 MB, jadi image memakai batas upload 4 MB. Rate limit-nya
  30 per menit per IP, karena kuota CPU dipakai bersama semua pengunjung.
- **Job async (`/v1/jobs`) tidak bisa diandalkan** karena server dijeda di antara request. UI tidak memakainya.

Batas paket gratis Vercel (dicek 2 November 2026): 1 vCPU, 2 GB, dan 4 jam CPU per bulan. Kalau kuota habis, layanan
bisa berhenti sampai 30 hari. Server dimatikan setelah 5 menit tanpa request.

Uji lokal 2 November 2026 dengan model asli dan pengaturan image ini, dibatasi ke satu CPU logis laptop (i7-13620H):

| Yang diukur | Hasil |
|---|---|
| Server menjawab setelah dinyalakan | 1,1 detik |
| Memuat model (`/v1/warmup`) | 7,8 detik, 3,3 detik CPU |
| Kalimat diketik | p50 0,16 detik, 0,14 detik CPU |
| Rekaman 5 detik | p50 1,24 detik, 1,10 detik CPU |
| 5 pengguna, jeda 15 detik | 13 dari 13 berhasil, p95 2,34 detik |
| 10 pengguna, jeda 15 detik | p95 3,26 detik; 1 dari 27 gagal saat pengaman memori alat uji mematikan server |
| Memori setelah model dimuat (RSS) | Sekitar 1 GB |

Dengan angka ini, 4 jam CPU setara sekitar 13.000 rekaman 5 detik per bulan. Hasil di Vercel kemungkinan lebih lambat:
vCPU-nya bisa lebih lemah dari inti laptop, dan waktu menyalakan container belum termasuk. `Dockerfile.vercel` belum
pernah di-build, dan aplikasi belum pernah dijalankan di Vercel.

### Docker (server permanen)

```bash
cp .env.example .env               # isi SPEAKFIX_API_KEYS dan GRAFANA_ADMIN_PASSWORD
docker compose up -d --build                                  # API CPU di :8000
docker compose --profile gpu up -d --build api-gpu            # API GPU di :8001 (NVIDIA Container Toolkit)
docker compose --profile monitoring up -d                     # Prometheus :9090, Grafana :3000
```

Isi image dan container:

- **Image CPU** (`docker/Dockerfile`): multi-stage build berbasis `python:3.12-slim`. Image ini tidak
  memuat PyTorch karena inferensi cukup dengan CTranslate2 + faster-whisper.
- **Versi dependensi dikunci** dengan `constraints.txt`, jadi build hari ini dan bulan depan memasang versi
  yang sama. Semua versi itu sudah dicek tersedia sebagai wheel Linux untuk Python 3.12 (dicek ulang 1 November
  2026).
- **ERRANT (tipe kesalahan) aktif secara default** (`EXTRAS=errant`), termasuk model spaCy-nya. Untuk
  image yang lebih kecil tanpa tipe kesalahan, build dengan `--build-arg EXTRAS=`.
- **Image GPU** (`docker/Dockerfile.gpu`): `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`
  (CUDA 12 + cuDNN 9 sesuai kebutuhan CTranslate2 4.x), dengan default Whisper `small.en`.
- **Hardening container**:
  - user non-root (uid 10001), root filesystem read-only, `cap_drop: ALL`, `no-new-privileges`;
  - `/tmp` berupa **tmpfs (RAM)**, jadi audio sementara tidak pernah menyentuh disk;
  - model di-mount read-only dari `./models`, dengan `HF_HUB_OFFLINE=1`.
- **HEALTHCHECK** memanggil `/healthz`. Untuk load balancer, gunakan `/readyz`.
- **Watchdog inferensi**: jika satu panggilan model berjalan lebih dari 2× `REQUEST_TIMEOUT_SECONDS`
  (misalnya kernel GPU macet), `/healthz` dan `/readyz` mengembalikan 503. Hal ini perlu karena thread
  yang macet menahan slot inferensi selamanya. Di Kubernetes, pasang `livenessProbe` ke `/healthz`.
  Docker biasa tidak me-restart container *unhealthy*, jadi gunakan orkestrator atau `autoheal`.

> Job `docker` di CI berhasil mem-build image CPU (2 November 2026), tetapi image-nya belum pernah dijalankan karena
> Docker tidak terpasang di laptop pengembang. Isi image CPU sudah diuji di venv bersih tanpa PyTorch dan tanpa editable
> install (`reports/e2e_recordings.md`, `reports/load_test.md`). Image GPU belum pernah di-build.

Checklist produksi:

- [ ] Isi `SPEAKFIX_API_KEYS` dengan key acak:
      `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- [ ] Pasang reverse proxy HTTPS (Caddy/nginx) di depan, lalu set `FORWARDED_ALLOW_IPS` ke IP proxy.
- [ ] Isi `SPEAKFIX_CORS_ORIGINS` hanya dengan domain aplikasi Anda.
- [ ] Jalankan 1 worker per container dan scale dengan menambah replika. Setiap worker memuat model
      sendiri, jadi jangan memakai `--workers`.
- [ ] Rate limiter dan antrean job bersifat per-instance. Untuk banyak replika, pindahkan rate limit ke
      reverse proxy.
- [ ] Simpan `models/` di volume. Artefaknya tercatat di `models/MANIFEST.json` (SHA-256).
- [ ] Simpan backup model **di luar laptop** (Google Drive, OneDrive, atau repo privat Hugging Face).
      `python scripts/backup_models.py --name t5-gec-v2 --out-dir <folder>` membuat satu zip berisi model
      HF dan CTranslate2 beserta manifest, lalu memverifikasi checksum-nya. Untuk memulihkan, ekstrak zip
      ke folder `speakfix/`.
- [x] API key Gemini lama yang pernah tertulis di `GEMINI2.0.py` sudah diblokir Google sebagai key bocor
      (dicek 8 Oktober 2026: HTTP 403 *"Your API key was reported as leaked"*). Key itu tidak bisa dipakai
      lagi. Menghapusnya dari project pemiliknya hanya untuk kebersihan.

## Monitoring

`/metrics` (format Prometheus) menyediakan:

| Metrik | Isi |
|---|---|
| `speakfix_http_requests_total`, `speakfix_http_request_duration_seconds` | Per template route |
| `speakfix_stage_duration_seconds{stage}` | `decode`, `preprocess`, `stt`, `correct` |
| `speakfix_inference_inflight`, `speakfix_jobs{state}` | Beban inferensi dan antrean job |
| `speakfix_inference_errors_total{component,kind}`, `speakfix_fallbacks_total` | OOM, timeout, fallback |
| `speakfix_process_resident_memory_bytes`, `speakfix_model_ready`, `speakfix_model_info` | Memori dan status model |
| `speakfix_audio_duration_seconds`, `speakfix_rate_limited_total` | Durasi klip dan rate limit |

Dashboard Grafana *SpeakFix* (11 panel) dan aturan alert ter-provision otomatis. Alert mencakup
instance down, model tidak siap, p95 di atas 8 detik, error 5xx di atas 5%, OOM, dan RSS di atas
2,5 GB.

Log ditulis sebagai JSON per baris (structlog) dengan `request_id`. Nilai field seperti `text`,
`transcript`, dan `audio` **selalu diredaksi** menjadi `<redacted len=N>`.

---

## Privasi & perlindungan data anak

**Kontrol teknis yang sudah ada di kode:**

- **Consent**: endpoint audio menolak request tanpa `consent=true` (403).
- **Tanpa penyimpanan**:
  - audio diproses di memori, termasuk upload multipart yang tidak di-spool ke disk;
  - folder temp privat (0700) disapu berkala dan dihapus saat shutdown;
  - di Docker, folder temp berada di tmpfs.
- **Job async**:
  - hasil dienkripsi (Fernet) dengan kunci per proses yang hilang saat restart;
  - kedaluwarsa otomatis (TTL), bisa dihapus segera lewat `DELETE`, dan hanya bisa dibaca oleh pemilik
    API key-nya.
- **Log dan cache**:
  - log tidak pernah berisi transkrip, teks, audio, atau API key;
  - respons API memakai `Cache-Control: no-store`.
- **STT cloud mati secara default.** Jika dinyalakan:
  - audio dikirim ulang sebagai WAV tanpa metadata;
  - Gemini menerima data *inline* (tidak memakai Files API yang menyimpan file 48 jam);
  - Deepgram dikirimi `mip_opt_out=true` agar data tidak dipakai untuk melatih model mereka.
- **Keamanan HTTP**:
  - API key dibandingkan secara constant-time dalam bentuk hash;
  - rate limit, batas ukuran body, dan validasi magic bytes;
  - header keamanan dan CSP ketat pada UI.

**Kewajiban operator** (tidak bisa diselesaikan oleh kode; ini bukan nasihat hukum):

- **COPPA** (AS, anak < 13 tahun): *verifiable parental consent* sebelum mengumpulkan suara anak,
  pemberitahuan privasi yang jelas, minimisasi data, dan kebijakan retensi.
- **UU PDP No. 27/2022** (Indonesia): pemrosesan data anak memerlukan persetujuan orang tua/wali.
- Jika memakai STT cloud: perjanjian pemrosesan data (DPA) dengan vendor.
- **Gemini API tidak boleh dipakai untuk layanan anak.** Ketentuan Gemini API melarang pemakaiannya di
  layanan yang ditujukan untuk, atau kemungkinan diakses oleh, pengguna di bawah 18 tahun. Pada tier gratis,
  input juga dipakai Google untuk mengembangkan produknya dan bisa dibaca reviewer manusia. Adapter Gemini
  hanya untuk eksperimen dengan suara orang dewasa, bukan untuk produksi SpeakFix.

---

## Kualitas kode & testing

```powershell
pytest -m "not slow"          # 178 test cepat, tanpa model (dipakai CI)
pytest                        # + 7 test dengan model asli / ERRANT (dilewati jika belum terpasang)
pytest --cov=speakfix         # coverage 91% (84% di CI, tanpa model)
ruff check src tests scripts ; ruff format --check src tests scripts
mypy src                      # mode strict, 0 error
```

Cakupan test:

- **Unit**: normalisasi teks, edits, corrector (OOM, fallback, halusinasi, chunking), audio (sniffing,
  decode, resampling, filter, denoise, normalisasi), job queue, enkripsi, temp sweeper, rate limiter,
  konfigurasi, redaksi log, adapter Deepgram (HTTP tiruan), dan factory.
- **Integrasi**: semua endpoint beserta kode errornya, consent, API key, rate limit, job async lintas
  pemilik, dan metrik.
- **End-to-end**: model asli, termasuk rekaman suara asli.

CI (`.github/workflows/ci.yml`) menjalankan:
- lint dan mypy;
- test di Python 3.12 dengan versi terkunci (yang benar-benar dipakai Docker), dengan coverage minimal 75%;
- test di Python 3.11 dengan versi terbaru, sebagai deteksi dini jika ada library baru yang merusak;
- `pip-audit` pada versi terkunci (hanya informasi, tidak menggagalkan CI);
- build image Docker CPU.

Hasil `pip-audit` 1 November 2026 pada versi terkunci:

- `setuptools` 81.0.0 (PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w): hanya menyangkut build sdist dan instalasi wheel
  legacy, tidak terjangkau saat service berjalan. Belum bisa di-upgrade karena `torch` mensyaratkan `setuptools<82`.
- `torch` dan `transformers`: advisory-nya menyangkut pemuatan model yang tidak tepercaya. Keduanya hanya untuk
  training dan fallback, tidak ada di image Docker, dan service hanya memuat model sendiri.
- Paket runtime lain bersih.

Setelah meng-upgrade dependensi dan semua test lulus, perbarui lock dengan
`python scripts/freeze_constraints.py`.

Uji end-to-end terhadap server yang sedang berjalan, lokal atau lewat link publik:

```powershell
python scripts/check_api.py http://127.0.0.1:8000       # rekaman tim, format audio, endpoint, jalur error
pip install playwright; playwright install firefox webkit
python scripts/check_browser.py http://127.0.0.1:8000 --engine msedge    # juga: firefox, webkit, chrome
```

Skenario uji manual untuk tim ada di `reports/pre_test_check.md`.

---

## Keputusan desain

- **CTranslate2, bukan ONNX Runtime/vLLM/Triton.**
  - CTranslate2 mendukung T5 dan Whisper sekaligus (satu runtime) dengan int8 di CPU. Hasilnya 5,6×
    lebih cepat dan tidak butuh PyTorch di image.
  - ONNX Runtime untuk T5 generatif memerlukan ekspor encoder/decoder terpisah dan beam search di luar
    graph, sehingga lebih rumit dengan manfaat serupa.
  - vLLM berfokus pada LLM decoder-only, sedangkan dukungan encoder-decoder seperti T5 terbatas.
  - Triton masuk akal untuk banyak model dan banyak GPU, terlalu berat untuk skala ini.
- **Antrean job di dalam proses, bukan Celery/Redis.** Setiap worker Celery akan memuat model sendiri
  (1–2 GB), dan audio anak akan singgah di broker. Antrean terbatas + limiter sudah cukup untuk API
  non-blocking. Scale out dilakukan dengan replika di belakang load balancer.
- **Beam search (5), bukan greedy**: diukur pada model v2.

  | Decoding | JFLEG BLEU 4-ref | JFLEG exact | Kaggle exact | Waktu |
  |---|---|---|---|---|
  | greedy | 0,866 | 34,1% | 52,6% | 1× |
  | beam 5 | 0,875 | 36,4% | 53,3% | ±1,8× |

  Beam 5 menjadi default. Klien bisa menurunkannya per request lewat `num_beams` bila latensi lebih
  penting.
- **`SPEAKFIX_WHISPER_MODEL=auto`**: `small.en` di GPU (hasil terbaik dan stabil, tetap cepat), `base.en` di CPU
  (hasilnya bertumpang tindih dengan `small.en`, tetapi jauh lebih cepat).
- **Faithfulness guard dan tanda baca akhir**: keduanya diambil dari bukti eksperimen (lihat bagian
  Hasil). Guard hanya aktif pada 0,3% kalimat JFLEG.

## Troubleshooting

- **`numpy.core.umath failed to import` / `float8_e4m3b11fnuz`** di Python global: TensorFlow lama
  bentrok dengan NumPy 2. Gunakan venv project ini, atau set `USE_TF=0`.
- **Log `ct2_cuda_unavailable_using_cpu` / `whisper_cuda_unavailable_using_cpu`**, atau `/v1/info`
  menunjukkan `@cpu` padahal ada GPU: library CUDA tidak bisa dimuat, sehingga service otomatis memakai
  CPU. Di Windows, pasang PyTorch CUDA di venv
  (`pip install torch --index-url https://download.pytorch.org/whl/cu128`). SpeakFix akan memakai
  cuBLAS/cuDNN yang dibawanya. Alternatifnya, pasang CUDA 12 + cuDNN 9 atau pakai image Docker GPU.
- **`TypeError: open() got an unexpected keyword argument 'metadata_errors'`** saat memanggil
  `faster_whisper.decode_audio` langsung: helper itu tidak cocok dengan PyAV 19. SpeakFix tidak
  terdampak karena memakai decoder sendiri (`speakfix.audio.decode`). Untuk script lain, kirim array
  NumPy ke `transcribe()`, bukan path file.
- **Peringatan symlink huggingface_hub di Windows**: aman diabaikan, hanya membuat cache sedikit lebih
  besar.
- **`/readyz` 503 dengan `FileNotFoundError`**: model belum disiapkan. Jalankan
  `scripts/prepare_models.py`, atau set `SPEAKFIX_T5_MODEL_NAME=t5-gec`.
- **Tombol rekam memberi pesan "mikrofon hanya lewat https://"**: browser memblokir mikrofon di alamat
  `http://` selain localhost, misalnya `http://192.168.x.x:8000`. Buka lewat link HTTPS (demo publik)
  atau `http://localhost:8000`, atau unggah file rekaman.
- **`run_public_demo.ps1 cannot be loaded because running scripts is disabled`**: jalankan lewat
  `powershell -ExecutionPolicy Bypass -File scripts\run_public_demo.ps1` (hanya untuk perintah itu).
- **Kalimat yang jelas salah tidak dikoreksi, dengan catatan "sistem belum cukup yakin"**: filter
  keyakinan menahan saran yang ragu-ragu (lihat *Mengurangi koreksi keliru*). Turunkan
  `SPEAKFIX_MIN_CORRECTION_MARGIN` (misalnya ke `1.0`) jika tim lebih memilih lebih banyak saran.
- **Banyak pengguna di satu Wi-Fi mendapat "Terlalu banyak permintaan" (429)**: rate limit dihitung per IP. Naikkan
  `SPEAKFIX_RATE_LIMIT_PER_MINUTE`, seperti yang dilakukan skrip demo (600).
- **`run_public_demo.ps1`: "Port 8000 is already in use"**: masih ada server lain di port itu. Hentikan dulu, atau
  jalankan dengan `-Port <port lain>`.
- **"Server sedang penuh" (503 `resource_exhausted`) atau peringatan commit memory**: memori Windows habis. Tutup
  aplikasi berat dan kosongkan drive C:, lalu restart supaya page file mengecil. Di Windows, server memakai ±3–3,5 GB
  *commit memory*, meskipun RAM yang benar-benar terpakai lebih kecil.

## Uji nyata (server sungguhan, laptop)

Diukur 2 November 2026 dengan `scripts/run_public_demo.ps1` lewat link publik (GPU, `small.en`), dan dengan isi image
Docker di venv bersih (CPU, `base.en`). Input-nya 13 mp3 kalimat salah di folder FP dan 2 rekaman suara asli. Kolom
"Total" adalah `total_ms` dari server. Daftar lengkap ada di `reports/e2e_recordings.md`, dan bisa diulang dengan
`scripts/check_api.py` (lihat [Kualitas kode & testing](#kualitas-kode--testing)).

| Input | Transkrip Whisper (GPU) | Hasil koreksi (GPU) | Total GPU |
|---|---|---|---|
| Rekaman suara asli | She can sing well. | *(tanpa saran)* | 280 ms |
| Rekaman suara asli | She go to school. | She goes to school. | 336 ms |
| `they is here today.mp3` | They is here today. | They are here today. | 240 ms |
| `She don’t has … hired ….mp3` | She don't has enough experience, so they didn't hired her for the job. | She doesn't have enough experience, so they didn't hire her for the job. | 493 ms |
| `He readed the book.mp3` | He read the book. | *(tanpa saran)* | 245 ms |
| `This kind of mistakes ….mp3` | This kind of mistakes as happens when we not focus during the test. | *(saran ditahan filter)* | 411 ms |

| Pengukuran | GPU, `small.en` | CPU, `base.en` |
|---|---|---|
| 10 kalimat dokumen tim: mendapat saran | 9/10 | 9/10 |
| 10 kalimat dokumen tim: sama persis dengan Expected Output (hasil lama: 1/10) | 2/10 | 2/10 |
| 15 rekaman: mendapat saran | 12/15 | 12/15 |
| 15 rekaman: sama persis dengan kalimat yang diharapkan | 7/15 | 7/15 |
| 15 kalimat yang sama, diketik: mendapat saran | 15/15 | 15/15 |
| 15 kalimat yang sama, diketik: sama persis | 8/15 | 9/15 |
| Waktu server (median / paling lambat) | 458 / 1.051 ms | 1.347 / 1.688 ms |

Baris pertama dan kelima di tabel atas menunjukkan batasan STT: Whisper sudah "membetulkan" kesalahannya, sehingga tidak
ada yang bisa dikoreksi. Di baris terakhir Whisper salah dengar, dan filter keyakinan menahan sarannya.

Dibandingkan dengan dokumen tim (`FP/WHISPER DENGAN INPUT AUDIO.docx`), jarak kata ke Expected Output pada 10
kalimat itu (159 kata): tanpa koreksi 43, hasil lama 17, SpeakFix 20 (GPU) dan 17 (CPU).
Sepuluh kalimat dengan satu referensi terlalu sedikit untuk menentukan mana yang lebih baik, dan di sampel ini
keduanya berdekatan. Perbandingan yang lebih kuat ada di data uji held-out (bagian Evaluasi).

## Batasan & langkah berikutnya

- **Belum ada suara anak yang diuji.** Evaluasi STT memakai 15 rekaman yang didominasi suara dewasa/TTS. Data suara
  anak diperlukan untuk evaluasi dan untuk fine-tuning Whisper (misalnya korpus MyST). Selama itu, demo publik
  meminta suara orang dewasa saja.
- **Trade-off filter keyakinan.** Dengan ambang 2,5, kalimat benar yang ikut diubah turun drastis, tetapi
  sekitar separuh kalimat salah di data uji dibiarkan tanpa saran. Kalimat pendek seperti *"I am agree with your
  opinion."* tertahan (margin 1,13). Ambang ini perlu dikalibrasi ulang dengan kalimat anak sungguhan.
- **Model belum mengenali sebagian pola**, misalnya *"She is very like cats"* dibiarkan apa adanya.
- **Kalimat campur bahasa bisa rusak.** Model dilatih untuk bahasa Inggris: *"aku suka makan nasi goreng and I likes apples"* menjadi
  *"Akan nasi goreng and I like apples."*. Kalimat bahasa Indonesia penuh biasanya dibiarkan apa adanya, tetapi hasilnya tidak dijamin.
- **Whisper hanya bahasa Inggris** (`base.en`/`small.en`), dan kadang menormalkan kesalahan atau salah dengar
  (lihat [Evaluasi speech-to-text](#evaluasi-speech-to-text)).
- **Image Docker belum pernah dijalankan.** CI berhasil mem-build image CPU, tetapi Docker tidak terpasang di laptop
  pengembang. Isi image CPU sudah diuji di venv bersih di Windows (paket dan pengaturan yang sama). Image GPU dan
  `Dockerfile.vercel` belum pernah di-build.
- **Perekaman dari HP sungguhan belum diuji** (lihat bagian Demo publik).
- Tipe kesalahan (ERRANT) sudah diuji di laptop pengembang (errant 3.0.2, spaCy 3.8). Pengujian ini
  menemukan bug: teks dulu dipecah per spasi saja, sehingga *don't → doesn't* diberi label `R:OTHER` dan
  tanda baca yang menempel pada kata ikut salah label. Sekarang spaCy yang melakukan tokenisasi, dan ada
  test untuk kasus ini.
- Skenario 3 ("data Grammar Correction Kaggle tanpa diolah"): file `Grammar Correction.csv.xls` yang
  diunduh 6 Oktober 2026 identik byte demi byte (MD5 `136568b4…`) dengan `FP/Grammar Correction.csv`.
  File itu sudah dipakai melatih v2 lewat `--kaggle`, jadi tidak perlu dilatih ulang. Untuk data baru,
  pakai `--extra`.

## Penulis

Jeremia Kevin Alexander Jagardo Malau
