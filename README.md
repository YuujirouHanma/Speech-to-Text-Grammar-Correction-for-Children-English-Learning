# SpeakFix

Layanan **transkripsi ucapan + koreksi grammar bahasa Inggris** untuk memantau pembelajaran anak.
Suara anak ditranskripsi oleh **Whisper** (lokal), lalu dikoreksi oleh model **T5** hasil fine-tuning.
Setiap respons berisi daftar perubahan per kata (*edits*) yang bisa langsung ditampilkan sebagai umpan balik.

Ini adalah versi produksi dari project riset di folder induk (`../T5PRETRAINASLI.ipynb`,
`../WHISPER.ipynb`, `../GEMINI2.0.py`). File riset tidak diubah, kecuali API key yang dihapus dari
`GEMINI2.0.py`.

---

## Hasil: sebelum vs sesudah

Semua angka di bawah diukur ulang di laptop pengembang (CPU Intel + RTX 4070 Laptop). Script dan data
ada di repo ini (lihat [Evaluasi](#evaluasi-benchmark-dan-training)). Laporan mentah ada di `reports/`.

| Aspek | Project awal | SpeakFix |
|---|---|---|
| JFLEG test, BLEU 4 referensi | 0,653 | **0,875** |
| JFLEG test, GLEU (metrik resmi JFLEG) | 0,395 | **0,571** |
| JFLEG test, ERRANT F0.5 | 0,432 | **0,652** |
| JFLEG test, exact match (salah satu referensi) | 22,5% | **36,4%** |
| Data uji Kaggle (300 kalimat held-out), exact match | 23,3% | **53,7%** |
| Kalimat salah yang dibiarkan tanpa koreksi (Kaggle) | 58% | **23,7%** |
| Latensi koreksi per kalimat (p50, CPU) | 603 ms (PyTorch fp32) | **156 ms** (CTranslate2 int8) |
| Memori model koreksi | ±1,9 GB | **±0,28 GB** |
| Ukuran model | 892 MB | **224 MB** |
| Kesalahan anak yang tetap tertulis apa adanya oleh STT | 53% | **67%** |
| Waktu training ulang | ±1 jam 45 menit (CPU) | **3 menit** (GPU) |
| Kebocoran data validasi ke data latih | 57–65% baris | **0** (dicek otomatis) |

Asal peningkatannya:

1. **Decoding diperbaiki** (model yang sama): kalimat diproses satu per satu, panjang output dinamis
   (dulu `max_length=20` memotong 57% referensi), dan tanda baca akhir ditambahkan sebelum inferensi
   (tanpa titik, model mengarang lanjutan seperti *"I am a student **and I am a teacher**"*).
   Hasilnya: BLEU JFLEG 0,653 → 0,856.
2. **Model dilatih ulang (v2)** dengan JFLEG + dataset Kaggle Grammar Correction, memakai split bebas
   kebocoran dan pasangan identitas. v2 sekarang memperbaiki kesalahan subject-verb agreement yang dulu
   lolos, misalnya *"He don't like apples"*, *"They is here today"*, dan *"She don't has"*.
3. **Runtime CTranslate2 int8** menggantikan PyTorch untuk inferensi: 3,9× lebih cepat dan 6,8× lebih
   hemat RAM. Untuk kalimat tunggal, hasilnya bahkan lebih cepat daripada PyTorch di GPU.
4. **Whisper + verbatim prompt**: STT tidak lagi diam-diam "membetulkan" kesalahan anak. Contohnya
   rekaman asli *"She can sings well."* sekarang tertulis apa adanya. Dulu tertulis *"She can sing well."*,
   sehingga kesalahan itu hilang dari umpan balik.

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
   detokenisasi, pemangkasan kalimat karangan, dan *faithfulness guard*.
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
├── scripts/          prepare_models, train_t5, evaluate_gec (+ gec_metrics), evaluate_stt, benchmark,
│                     backup_models, freeze_constraints
├── constraints.txt   versi dependensi yang sudah teruji (dipakai Docker, CI, dan setup lokal)
├── tests/            unit/ dan integration/ (156 test)
├── docker/           Dockerfile (CPU), Dockerfile.gpu (CUDA 12.8)
├── monitoring/       prometheus.yml, alerts.yml, dashboard Grafana
├── reports/          hasil evaluasi & benchmark
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
> Jalur PyTorch sudah diuji di RTX 4070 Laptop: Whisper base.en 5–8× lebih cepat dan T5 3–4× lebih
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
- **Model**: `trimmed_extra_model_output`, `low_confidence_correction_skipped`.

---

## Konfigurasi

Semua pengaturan dibaca dari environment variable berprefix `SPEAKFIX_` atau dari file `.env`
(lihat `.env.example`). Daftar lengkapnya ada di `src/speakfix/config.py`. Yang terpenting:

| Variabel | Default | Keterangan |
|---|---|---|
| `SPEAKFIX_API_KEYS` | *(kosong)* | Daftar key dipisah koma. Kosong berarti tanpa autentikasi (hanya untuk lokal). |
| `SPEAKFIX_REQUIRE_CONSENT` | `true` | Endpoint audio mewajibkan `consent=true`. |
| `SPEAKFIX_RATE_LIMIT_PER_MINUTE` | `60` | Batas request per key/IP (`0` = mati). |
| `SPEAKFIX_CORS_ORIGINS` | *(kosong)* | Origin browser yang diizinkan. |
| `SPEAKFIX_T5_MODEL_NAME` | `t5-gec-v2` | Memakai `models/<nama>` dan `models/<nama>-ct2`. |
| `SPEAKFIX_CORRECTOR_BACKEND` | `ctranslate2` | `ctranslate2` atau `torch`. |
| `SPEAKFIX_CORRECTOR_FALLBACK_BACKEND` | `torch` | Dipakai bila backend utama gagal/OOM (`none` = mati). |
| `SPEAKFIX_DEVICE` | `auto` | `cpu`, `cuda`, atau `auto`. |
| `SPEAKFIX_WHISPER_MODEL` | `base.en` | `small.en` lebih tahan aksen tetapi 3× lebih lambat di CPU. |
| `SPEAKFIX_WHISPER_VERBATIM_PROMPT` | `true` | Menjaga kesalahan grammar tetap tertulis. |
| `SPEAKFIX_AUDIO_DENOISE` | `false` | Spectral gating, hanya untuk ruangan bising. |
| `SPEAKFIX_MAX_UPLOAD_MB` / `SPEAKFIX_MAX_AUDIO_SECONDS` | `10` / `60` | Batas input. |
| `SPEAKFIX_INFERENCE_WORKERS` | `1` | Inferensi paralel. Di GPU, `2` membuat latensi 4× lebih rendah saat ramai (lihat Uji beban). |
| `SPEAKFIX_MAX_PENDING_REQUESTS` | `16` | Lebih dari ini akan dijawab 503 (backpressure). |
| `SPEAKFIX_JOB_TTL_SECONDS` | `600` | Masa hidup hasil job async. |
| `SPEAKFIX_ALLOW_CLOUD_STT` | `false` | Mengizinkan Gemini/Deepgram (baca bagian Privasi). |
| `SPEAKFIX_MIN_CORRECTION_MARGIN` | `2.5` | Saran hanya ditampilkan jika model cukup yakin (lihat *Mengurangi koreksi keliru*). `1.0` = lebih banyak saran tetapi lebih banyak koreksi keliru; `-inf` = filter mati. |
| `SPEAKFIX_PUNCTUATION_FEEDBACK` | `true` | Umpan balik tanda baca dan huruf kapital untuk teks ketikan. Transkrip suara tidak pernah mendapatkannya. |
| `SPEAKFIX_FEEDBACK_LANGUAGE` | `id` | Bahasa deskripsi kesalahan di API (`id` atau `en`). |
| `SPEAKFIX_DEMO_MODE` | `false` | Demo publik: UI meminta suara orang dewasa saja. |

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
    --systems legacy speakfix --gec-metrics --label v2
```

| Sistem | Data | BLEU 1-ref | BLEU 4-ref | Exact match | Tidak diubah |
|---|---|---|---|---|---|
| Pipeline lama (notebook), model asli | JFLEG test (747) | 0,519 | 0,653 | 22,5% | 20,6% |
| SpeakFix, model asli | JFLEG test | 0,711 | 0,856 | 33,6% | 35,7% |
| **SpeakFix, model v2** | JFLEG test | **0,727** | **0,875** | **36,4%** | 25,8% |
| Pipeline lama, model asli | Kaggle held-out (300) | 0,688 | – | 23,3% | 58,0% |
| SpeakFix, model asli | Kaggle held-out | 0,684 | – | 24,7% | 60,0% |
| **SpeakFix, model v2** | Kaggle held-out | **0,815** | – | **53,7%** | **23,7%** |

Cara membaca tabel:

- Semua skor dihitung pada teks yang sudah dide-tokenisasi. Pipeline lama mendapat 0,519 di sini,
  sedangkan di notebook 0,497 karena memakai versi tokenized dan implementasi BLEU HuggingFace.
- Faithfulness guard hanya membatalkan 2 dari 747 koreksi JFLEG, keduanya pada input yang sangat rusak.
- Split Kaggle berisi 302 kalimat uji, tetapi yang dievaluasi 300. Dua kalimat tersimpan di `split.json`
  dalam bentuk yang sudah dibersihkan (misalnya ditambah titik di akhir), sehingga tidak cocok persis
  dengan teks mentah di CSV. Keduanya tetap tidak pernah dipakai untuk training.
- Gaya input "raw" setara dengan gaya tokenisasi JFLEG, sehingga yang dipakai adalah raw.

#### Metrik GEC standar: GLEU dan ERRANT F0.5

Dengan `--gec-metrics` (butuh fitur `errant` dan `en_core_web_sm`), script juga menghitung GLEU (metrik
resmi JFLEG) dan ERRANT F0.5 (metrik standar BEA/CoNLL). Implementasinya ada di `scripts/gec_metrics.py`.
Pada output yang sama, hasilnya identik dengan skrip resmi `gleu.py` (repo keisks/jfleg) dan
`errant_parallel` + `errant_compare`, dan ini juga dikunci dengan unit test.

| Sistem | JFLEG GLEU | JFLEG ERRANT P / R / F0.5 | Kaggle GLEU | Kaggle ERRANT F0.5 |
|---|---|---|---|---|
| Tanpa koreksi (output = input) | 0,407 | – / 0 / 0 | 0,313 | 0 |
| Pipeline lama (notebook), model asli | 0,395 | 0,473 / 0,321 / 0,432 | 0,490 | 0,476 |
| SpeakFix, model asli | 0,527 | 0,714 / 0,387 / 0,611 | 0,484 | 0,505 |
| **SpeakFix, model v2** | **0,571** | 0,713 / **0,487** / **0,652** | **0,726** | **0,695** |

- Teks ditokenisasi dengan spaCy. Baseline "tanpa koreksi" mendapat 0,407, hampir sama dengan baseline
  SOURCE resmi di leaderboard JFLEG (40,54). Jadi tokenisasi ini cukup dekat dengan yang resmi.
- Pipeline lama mendapat GLEU **di bawah** baseline tanpa koreksi (0,395 < 0,407). Secara rata-rata,
  outputnya lebih buruk daripada membiarkan kalimat apa adanya, salah satunya karena 10,8% output
  terpotong di 20 token.
- Sebagai konteks, leaderboard JFLEG mencatat REFERENCE (manusia) 62,37 dan sistem 2017–2018 di kisaran
  54–62 (misalnya Chollampatt & Ng 2018: 57,47). v2 berada di kisaran itu, tetapi angkanya tidak bisa
  dibandingkan langsung. Tokenisasinya sedikit berbeda, dan v2 dilatih dengan JFLEG dev (`train.csv`),
  yang pada sistem di leaderboard biasanya hanya dipakai untuk tuning.
- Peningkatan v2 terutama ada di recall (0,387 → 0,487). Presisinya tetap sekitar 0,71, artinya v2
  memperbaiki lebih banyak kesalahan tanpa menambah koreksi yang keliru.

Model v2 yang sama di GPU (`--device cuda --compute-type ...`, label `v2-gpu-*` di
`reports/gec_eval.json`):

| Compute type CTranslate2 | JFLEG BLEU 4-ref | Kaggle exact match |
|---|---|---|
| int8, CPU (acuan di atas) | 0,875 | 53,7% |
| **int8_float32, GPU (default)** | **0,872** | **53,0%** |
| float32, GPU | 0,872 | 52,7% |
| int8_bfloat16, GPU (default lama) | 0,867 | 52,7% |

Selisih ini kecil (1 kalimat Kaggle = 0,33%), tetapi bfloat16 selalu paling rendah. Pada uji 120 kalimat
JFLEG, bfloat16 juga lebih lambat (p50 122 ms vs 98 ms). Karena itu default GPU sekarang `int8_float32`.

#### Mengurangi koreksi keliru (filter keyakinan)

Model v2 masih mengubah sebagian kalimat yang **sudah benar**, misalnya *"The group of volunteers works"*
diubah menjadi *"work"*. Untuk anak, saran yang keliru lebih merugikan daripada kesalahan yang terlewat.
Karena itu ada dua pengaman:

1. **Filter keyakinan** (`SPEAKFIX_MIN_CORRECTION_MARGIN`, default `2.5`). Model memberi skor
   log-probabilitas pada saran koreksinya dan pada kalimat asli. Saran hanya ditampilkan jika selisihnya
   mencapai ambang. Skor dihitung dalam satu *forward pass* tambahan, hanya untuk kalimat yang berubah.
   Saran yang ditahan diberi catatan `uncertain_correction_skipped`.
2. **Tanpa umpan balik tanda baca untuk suara.** Tanda baca dan huruf kapital dalam transkrip berasal
   dari Whisper, bukan dari anak. Perubahan yang hanya menyangkut keduanya diabaikan, begitu pula titik
   yang ditambahkan sistem untuk model. Teks ketikan tetap mendapat umpan balik tanda baca.

Ambang dikalibrasi di **data validasi**, bukan di data uji
(`python scripts/calibrate_correction.py --jfleg-train ..\train.csv --jfleg-test ..\eval.csv --kaggle "..\..\FP\Grammar Correction.csv"`).
Data validasinya adalah 238 kalimat salah dan 328 kalimat benar yang tidak pernah dipakai untuk training.
Skrip memeriksa bahwa split yang direkonstruksi identik dengan `split.json`. Hasil lengkapnya ada di
`reports/correction_calibration.json`. ERRANT di tabel ini tidak menghitung edit tanda baca.

| Ambang | Presisi | Recall | F0.5 | Kalimat benar diubah | Kalimat salah tanpa saran |
|---|---|---|---|---|---|
| tanpa filter | 0,598 | 0,524 | 0,582 | 15,2% | 24,4% |
| 1,0 | 0,656 | 0,487 | 0,613 | 6,7% | 33,6% |
| 2,0 | 0,686 | 0,463 | 0,626 | 4,9% | 40,3% |
| **2,5 (default)** | **0,715** | 0,447 | **0,638** | **3,0%** | 44,5% |
| 4,0 | 0,744 | 0,381 | 0,625 | 1,8% | 55,5% |

Hasil di **data uji** (CPU int8, model v2; label `v2-baseline`, `v2-margin2.5-text`, dan
`v2-margin2.5-speech` di `reports/gec_eval.json`). Kalimat benar diukur dengan memasukkan referensi
pertama setiap kalimat sebagai input (`--correct-sentences`):

| Konfigurasi | Kalimat benar Kaggle diubah | Kalimat benar JFLEG diubah | ERRANT P / R / F0.5 Kaggle | ERRANT P / R / F0.5 JFLEG | Kalimat salah Kaggle tanpa saran |
|---|---|---|---|---|---|
| Tanpa filter (sebelumnya) | 6,3% | 32,3% | 0,741 / 0,557 / 0,695 | 0,713 / 0,487 / 0,652 | 23,7% |
| **Ambang 2,5, teks ketikan (default)** | **1,0%** | **7,2%** | **0,778** / 0,408 / 0,659 | **0,732** / 0,424 / 0,639 | 46,3% |
| Ambang 2,5, gaya transkrip suara | 1,0% | 7,0% | 0,782 / 0,358 / 0,632 | 0,720 / 0,404 / 0,622 | 51,0% |

Cara membaca:

- Koreksi keliru pada kalimat benar turun **6×** (Kaggle) dan **4,5×** (JFLEG), dan presisi naik.
  Bayarannya, sekitar separuh kalimat salah di data uji dibiarkan tanpa saran.
- F0.5 di data uji sedikit turun karena data ujinya hanya berisi kalimat salah, sehingga manfaat filter
  pada kalimat benar tidak ikut terhitung. Pada campuran kalimat salah dan benar (data validasi), F0.5
  naik dari 0,582 ke 0,638.
- Ke-13 kalimat uji tim (mp3 di folder FP) tetap terkoreksi dengan ambang 2,5 (margin 5,2–29,9).
  Begitu juga kalimat tipikal lain seperti *"My mother cook rice"* (7,6) dan *"I have went to Bali"*
  (4,7). Yang tertahan misalnya *"I am agree with your opinion."* versi pendek (1,17).
- Jika tim lebih memilih lebih banyak saran, pakai `SPEAKFIX_MIN_CORRECTION_MARGIN=1.0`. Jika ingin
  perilaku lama tanpa filter, pakai `-inf`.

### Evaluasi speech-to-text

```powershell
python scripts/evaluate_stt.py --audio-dir ..\..\FP --models base.en small.en `
    --extra "..\recorded.wav=She can sings well." --out reports/stt_eval.md
```

Dataset evaluasi berisi 15 rekaman: 13 mp3 kalimat salah dari folder FP dan 2 rekaman suara asli.

| Model | Verbatim prompt | WER | Kesalahan tetap tertulis | Real-time factor (CPU) |
|---|---|---|---|---|
| base.en | tidak | 0,050 | 53% | 0,095 |
| **base.en** | **ya** | **0,039** | **67%** | **0,092** |
| small.en | tidak | 0,044 | 53% | 0,266 |
| small.en | ya | 0,039 | 60% | 0,291 |

Data ini kecil dan didominasi suara dewasa/TTS. Sebelum dipakai ke anak-anak sungguhan, kumpulkan
rekaman anak (dengan consent) dan ulangi evaluasi ini.

### Benchmark latensi

```powershell
python scripts/benchmark.py --backends ct2-cpu ct2-cuda torch-cpu torch-cuda `
    --whisper base.en small.en --whisper-devices cpu cuda --audio ..\recorded.wav
```

Run 7 Oktober 2026, laptop idle, CPU (model asli, arsitektur sama dengan v2):

| Runtime | p50 | p95 | Paragraf 8 kalimat | Tambahan RAM | Waktu load |
|---|---|---|---|---|---|
| **CTranslate2 int8, CPU** | **156 ms** | **393 ms** | **1,07 s** | **279 MB** | **0,4 s** |
| PyTorch fp32, CPU | 603 ms | 1.776 ms | 7,69 s | 1.903 MB | 4,8 s |
| PyTorch bf16, GPU RTX 4070 | 363 ms | 956 ms | 1,17 s | 1.079 MB | 0,7 s |

Untuk Whisper di CPU int8, audio 5 detik diproses dalam 0,45 s dengan base.en dan 1,37 s dengan
small.en.

Run 8 Oktober 2026, CPU vs GPU dalam satu run (model v2). Saat run ini laptop sedang sibuk (browser
terbuka), sehingga semua angka absolut lebih tinggi dari run 7 Oktober. Yang sah dibandingkan adalah
selisih CPU vs GPU di dalam run yang sama.

| Runtime | p50 | p95 | Paragraf 8 kalimat |
|---|---|---|---|
| CTranslate2 int8, CPU | 229 ms | 847 ms | 2,94 s |
| **CTranslate2 int8_float32, GPU** | **222 ms** | **586 ms** | **0,68 s** |
| PyTorch bf16, GPU | 1.058 ms | 2.108 ms | 2,38 s |

| Whisper (audio 5 s) | CPU int8 | GPU float16 |
|---|---|---|
| base.en | 1,01–1,07 s | **0,13–0,22 s** |
| small.en | 3,23–3,69 s | **0,18 s** |

Untuk kalimat tunggal, GPU hampir tidak lebih cepat karena beam search T5 berjalan token demi token.
GPU baru unggul untuk input panjang dan untuk Whisper. Dengan GPU, small.en secepat base.en, tetapi
base.en tetap jadi default karena lebih akurat di data evaluasi (lihat tabel STT). Detail lengkap ada
di `reports/benchmark.md`.

### Uji beban (berapa banyak pengguna bersamaan?)

```powershell
python -m speakfix                  # di terminal lain; SPEAKFIX_INFERENCE_WORKERS=2 untuk GPU
python scripts/load_test.py --users 15 30 --seconds 45 --think 3 --audio ..\recorded.wav
```

Setiap pengguna virtual mengirim rekaman 5 detik, menunggu hasilnya, berhenti sejenak (*think time*),
lalu mengulang. Masing-masing memakai IP berbeda lewat `X-Forwarded-For`, sehingga rate limit per
pengunjung tetap berlaku seperti di produksi. Hasil di RTX 4070 Laptop, 8 Oktober 2026, saat laptop
tidak menjalankan pekerjaan lain:

| Skenario | Worker | Berhasil | Throughput | p50 | p95 |
|---|---|---|---|---|---|
| 30 pengguna, jeda ±15 s (tempo realistis anak) | 1 | 100% | 2,0/s | 0,25 s | 0,64 s |
| 30 pengguna, jeda ±3 s (agresif) | 1 | 99,3% | 6,5/s | 1,69 s | 2,35 s |
| 30 pengguna, jeda ±3 s (agresif) | **2** | **100%** | **8,6/s** | **0,40 s** | **0,84 s** |

- **Satu laptop cukup untuk satu kelas** (±30 anak yang bergantian bicara dan membaca hasil).
- Di GPU, gunakan `SPEAKFIX_INFERENCE_WORKERS=2`: latensi saat ramai turun 4×, karena kerja CPU
  (decode audio, tokenisasi, ERRANT) satu request bisa berjalan bersamaan dengan kerja GPU request lain.
  Dengan 2 worker, 300 request paralel menghasilkan jawaban yang identik dengan request berurutan,
  termasuk tipe kesalahannya, jadi tidak ada masalah *thread-safety*.
- Jika beban melebihi kapasitas, server menjawab 503 *"server sibuk"* dengan cepat dan tidak macet.

---

## Deployment

### Demo publik dari laptop (paling cepat, gratis)

```powershell
cd speakfix
powershell -ExecutionPolicy Bypass -File scripts\run_public_demo.ps1
```

Skrip ini:
- menjalankan SpeakFix di `127.0.0.1` (tidak bisa diakses langsung dari Wi-Fi), dengan GPU, 2 worker, dan
  **mode demo** (banner "gunakan suara orang dewasa", teks persetujuan untuk orang dewasa);
- membuka *Cloudflare Quick Tunnel* dan menampilkan link `https://<acak>.trycloudflare.com` yang bisa
  dibuka dari HP atau laptop mana pun. HTTPS membuat mikrofon browser bisa dipakai;
- saat pertama dijalankan, mengunduh `cloudflared.exe` resmi ke `.tools/` dan memeriksa tanda tangan
  digital Cloudflare-nya;
- menghentikan server dan tunnel saat Anda menekan **Ctrl+C**.

Sudah diuji 8 Oktober 2026 lewat link publik: UI, koreksi teks, dan analisis rekaman (0,56 s) berjalan.
Tunnel meneruskan IP asli pengunjung (`X-Forwarded-For`), sehingga rate limit berlaku per pengunjung,
dan log server tidak memuat isi kalimat maupun IP.

Batasan: link berganti setiap kali skrip dijalankan, laptop harus tetap menyala, dan *quick tunnel* tidak
punya jaminan uptime. `/docs` dan `/metrics` (statistik teknis tanpa data pribadi) ikut terbuka. Untuk
layanan permanen, gunakan Docker di server (di bawah).

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
  yang sama. Semua versi itu sudah dicek tersedia sebagai wheel Linux untuk Python 3.12.
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

> Dockerfile belum di-build di laptop pengembang karena Docker tidak terpasang. Job `docker` di CI akan
> mem-build image CPU pada setiap push, tetapi baru berjalan setelah repo di-push ke GitHub.

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
pytest -m "not slow"          # 149 test cepat, tanpa model (dipakai CI)
pytest                        # + 7 test dengan model asli / ERRANT (dilewati jika belum terpasang)
pytest --cov=speakfix         # coverage 89%
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
- `pip-audit` pada versi terkunci;
- build image Docker CPU.

Setelah meng-upgrade dependensi dan semua test lulus, perbarui lock dengan
`python scripts/freeze_constraints.py`.

---

## Keputusan desain

- **CTranslate2, bukan ONNX Runtime/vLLM/Triton.**
  - CTranslate2 mendukung T5 dan Whisper sekaligus (satu runtime) dengan int8 di CPU. Hasilnya 3,9×
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
  | greedy | 0,866 | 34,1% | 53,0% | 1× |
  | beam 5 | 0,875 | 36,4% | 53,7% | ±1,9× |

  Beam 5 menjadi default. Klien bisa menurunkannya per request lewat `num_beams` bila latensi lebih
  penting.
- **`base.en` sebagai default STT**: paling akurat di data evaluasi yang ada dan 3× lebih cepat dari
  `small.en`. Image GPU memakai `small.en`.
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

## Uji nyata (server sungguhan, laptop)

Kolom CPU diukur 7 Oktober 2026. Kolom GPU diukur 8 Oktober 2026 dengan `SPEAKFIX_DEVICE=auto`, yang
otomatis memilih GPU (`/v1/info`: `ctranslate2:int8_float32@cuda`). Waktu di bawah adalah `total_ms`
dari server; request pertama setelah start masih termasuk pemanasan GPU.

| Input | Transkrip Whisper | Hasil koreksi | Total CPU | Total GPU |
|---|---|---|---|---|
| Rekaman suara asli | She can sings well. | She can sing well. | 641 ms | 748 ms (request pertama) |
| `they is here today.mp3` | They is here today. | They are here today. | 980 ms | **190 ms** |
| `She don’t has … hired …mp3` | She don't has enough experience, so they didn't hired her for the job. | She doesn't have enough experience, so they didn't hire her for the job. | 1,38 s | **589 ms** |
| `He readed the book.mp3` | He read the book. | He read the book. | 985 ms | **166 ms** |

Baris terakhir menunjukkan batasan: Whisper menormalkan kata yang tidak ada di kamus (*readed*), sehingga
kesalahannya tidak sampai ke model koreksi. Selama pengujian, log server tidak memuat satu pun kata
dari transkrip.

## Batasan & langkah berikutnya

- Evaluasi STT masih memakai 15 rekaman (didominasi suara dewasa/TTS). Data suara anak diperlukan untuk
  evaluasi dan untuk fine-tuning Whisper (misalnya korpus MyST).
- **Trade-off filter keyakinan.** Dengan ambang 2,5, kalimat benar yang ikut diubah turun drastis, tetapi
  sekitar separuh kalimat salah di data uji dibiarkan tanpa saran. Kesalahan khas pembelajar Indonesia
  pada kalimat uji tim tetap terkoreksi, tetapi kalimat sangat pendek seperti *"I am agree with your
  opinion."* tertahan (margin 1,17). Ambang ini perlu dikalibrasi ulang dengan kalimat anak sungguhan.
- Model belum mengenali sebagian pola, misalnya *"She is very like cats"* dibiarkan apa adanya.
- Tipe kesalahan (ERRANT) sudah diuji di laptop pengembang (errant 3.0.2, spaCy 3.8). Pengujian ini
  menemukan bug: teks dulu dipecah per spasi saja, sehingga *don't → doesn't* diberi label `R:OTHER` dan
  tanda baca yang menempel pada kata ikut salah label. Sekarang spaCy yang melakukan tokenisasi, dan ada
  test untuk kasus ini. Docker GPU/CPU dengan `EXTRAS=errant` belum di-build (Docker tidak terpasang).
- Hanya bahasa Inggris (`base.en`). Kata bahasa Indonesia yang terselip di kalimat anak akan ditranskripsi
  kacau.
- Skenario 3 ("data Grammar Correction Kaggle tanpa diolah"): file `Grammar Correction.csv.xls` yang
  diunduh 6 Oktober 2026 identik byte demi byte (MD5 `136568b4…`) dengan `FP/Grammar Correction.csv`.
  File itu sudah dipakai melatih v2 lewat `--kaggle`, jadi tidak perlu dilatih ulang. Untuk data baru,
  pakai `--extra`.
