# Pemeriksaan sebelum testing

Diperiksa 1–2 November 2026 di laptop pengembang, pada kode di commit yang memuat laporan ini.
Semua angka di sini berasal dari laporan lain di folder `reports/`.

## Kesimpulan

SpeakFix **siap masuk tahap testing dengan suara orang dewasa**, lewat demo publik dari laptop atau server
lokal. Semua pengujian otomatis lulus, dan hasil evaluasi konsisten dengan yang tertulis di README.

Yang **belum** teruji dan perlu dicoba tim: perekaman dari HP sungguhan. Yang **belum boleh** dilakukan: memakai
suara anak (belum ada evaluasinya, dan belum ada proses persetujuan orang tua). Image Docker juga belum pernah
di-build.

## Sebelum mulai

**Kosongkan drive C: laptop yang menjalankan server, lalu restart.** Saat pemeriksaan, C: tersisa 0,2 GB dari
292 GB, dan page file Windows sudah membesar sampai 35 GB. Akibatnya memori sempat habis saat uji beban di
CPU, sehingga sebagian request gagal (503 dan 500). Restart biasanya mengecilkan page file. Yang paling banyak memakai
C: saat diperiksa: distro WSL Ubuntu 53 GB, `pagefile.sys` 35 GB, dan cache Hugging Face 7 GB (model BGE, bukan milik
project ini). Tutup juga aplikasi berat (game, Steam, tab browser yang tidak perlu) selama demo. Skrip demo memberi
peringatan jika memori yang tersedia di bawah 6 GB.

## Cara menjalankan

Demo untuk banyak orang (link HTTPS sementara, mikrofon bisa dipakai di HP):

```powershell
cd speakfix
powershell -ExecutionPolicy Bypass -File scripts\run_public_demo.ps1
```

Bagikan link `https://....trycloudflare.com` yang muncul di terminal. Tekan **Ctrl+C** untuk menutup; setelah itu
link mati. Jangan biarkan link menyala tanpa diawasi.

Uji sendiri di laptop:

```powershell
cd speakfix
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m speakfix        # buka http://localhost:8000
```

## Skenario uji untuk tim

Hasil yang diharapkan pada nomor 1–10 dan 13 sudah dicek otomatis di komputer. Nomor 11, 12, dan 14 belum pernah
dicoba, jadi di situlah temuan baru paling mungkin muncul.

| # | Langkah | Yang diharapkan |
|---|---|---|
| 1 | Buka link, baca halaman | Banner "Mode demo", teks persetujuan orang dewasa |
| 2 | Ketik `My mother cook rice every morning.` | Saran "cooks", kata "cook" disorot, penjelasan Bahasa Indonesia |
| 3 | Ketik `She goes to school every day.` | "Bagus! Tidak ada kesalahan yang terdeteksi." |
| 4 | Ketik `I am agree with your opinion.` | Tidak ada pujian, catatan "sistem belum cukup yakin" |
| 5 | Tekan "Mulai rekam" tanpa mencentang persetujuan | Diminta mencentang persetujuan |
| 6 | Centang, rekam `They is here today.`, tekan "Berhenti" | Rekaman bisa diputar, transkrip dan saran "are" muncul |
| 7 | Rekam sampai timer habis (59 detik) | Berhenti sendiri dan tetap diproses |
| 8 | Tolak izin mikrofon di browser | Pesan "Izin mikrofon ditolak..." |
| 9 | Unggah mp3 atau m4a kalimat salah | Transkrip dan saran muncul |
| 10 | Unggah file bukan audio, lalu file di atas 10 MB | Pesan kesalahan yang jelas, bukan error teknis |
| 11 | **HP**: ulangi langkah 6 di iPhone (Safari) dan Android (Chrome) | Sama seperti langkah 6 |
| 12 | **HP**: kunci layar saat merekam, lalu buka lagi | Tidak macet: rekaman berhenti sendiri atau bisa dihentikan dengan tombol |
| 13 | 20–30 orang menekan tombol bersamaan | Semua dapat jawaban, paling lambat sekitar 6 detik |
| 14 | Bicara dengan aksen dan kecepatan berbeda, di ruangan bising | Catat kalimat yang salah dengar |

Catat setiap temuan: kalimat yang diucapkan, hasil yang muncul, perangkat dan browser, serta jamnya. Saran yang keliru
pada kalimat yang sebenarnya benar paling penting untuk dilaporkan.

## Yang sudah diverifikasi

| Area | Cara | Hasil |
|---|---|---|
| Kode | ruff, mypy strict, pytest dengan model asli | 0 masalah, 173 test lulus, coverage 90% |
| Lingkungan bersih | venv baru dari `constraints.txt`, seperti CI | 166 test lulus, coverage 84% |
| Isi image Docker | venv tanpa PyTorch, CPU, tanpa editable install | Semua endpoint dan format audio berjalan; `reports/e2e_recordings.md` |
| Demo publik | `run_public_demo.ps1`, API lewat link HTTPS | 0 pengecekan gagal; link mati setelah ditutup |
| UI | Edge, Firefox, WebKit lewat link HTTPS | 52/52, 53/53, 43/43 lulus |
| Beban | 30–60 pengguna virtual, lonjakan 30 orang | `reports/load_test.md` |
| Angka evaluasi | Dijalankan ulang dengan kode final, dibandingkan dengan laporan | Semua metrik filter 2,5 sama persis, baik sebelum maupun sesudah perbaikan terakhir |
| Kalibrasi filter | Dijalankan ulang di data validasi | 20 dari 20 baris identik |
| Privasi | Isi log server selama semua uji | Tidak ada isi kalimat, tidak ada IP |
| Dependensi | `pip-audit` pada versi terkunci | Lihat catatan di README (*Kualitas kode*) |
| Checksum model | Dibandingkan dengan `models/MANIFEST.json` | Cocok; zip backup terverifikasi |

## Temuan pemeriksaan ini (sudah diperbaiki)

| Temuan | Akibat | Perbaikan |
|---|---|---|
| Decoder audio membuat satu thread per core untuk setiap file | Saat ramai server mencapai 944 thread lalu kehabisan memori (HTTP 500) | Decoder memakai 1 thread; sampel audio identik pada 23 file; sekarang paling banyak 31 thread |
| Timer rekam berhenti tepat di 60 detik, padahal server menolak di atas 60 detik | Rekaman penuh satu menit bisa ditolak | UI berhenti di 59 detik dan mengikuti batas server |
| Rekaman yang diakhiri browser (mis. layar HP terkunci) | Timer terus berjalan, tombol tetap "Berhenti" | Tombol dan timer direset saat rekaman berakhir |
| Pujian "Bagus!" muncul saat saran ditahan atau tidak ada suara | Pengguna mengira kalimatnya benar | Pujian tidak muncul pada kasus itu |
| Kalimat benar yang diketik tanpa titik selalu mendapat catatan "belum cukup yakin" | Catatan menyesatkan, tanpa pujian | Titik yang hilang langsung disarankan, tanpa catatan |
| Evaluasi Kaggle kehilangan 2 dari 302 kalimat tanpa pesan | Angka sedikit melenceng | Pemuat data diperbaiki, script berhenti jika ada yang hilang |
| Whisper base.en di GPU lebih buruk dari yang tertulis di README | Kesalahan lebih sering hilang dari transkrip | `SPEAKFIX_WHISPER_MODEL=auto`: small.en di GPU, base.en di CPU |
| File terlalu besar memutus koneksi | Pesan "Failed to fetch" | Dicek di browser, server menjawab 413 dengan benar |
| Error memori dilaporkan sebagai error umum | HTTP 500, dan pesan tidak jelas | Semua error memori menjadi 503 "server sedang penuh" |
| OpenBLAS membuat 15 thread dan memakan 0,5 GB tanpa manfaat | Memori boros | Dibatasi 1 thread |
| Banyak orang satu Wi-Fi kena rate limit, lonjakan ditolak antrean | Demo kelas gagal sebagian | Demo memakai 600/menit dan antrean 32 |
| Angka README lama keliru atau usang | Tabel utama tidak mencerminkan konfigurasi default | README ditulis ulang dari file hasil |

## Batasan yang perlu diketahui penguji

- Filter keyakinan sengaja menahan banyak saran: sekitar separuh kalimat salah di data uji tidak mendapat saran.
- Whisper kadang "membetulkan" kesalahan. Di GPU, `She can sings well.` tertulis `She can sing well.`, dan
  `He readed the book` tertulis `He read the book.`. Akibatnya tidak ada saran untuk kalimat itu.
- Di CPU, transkrip bisa sedikit berbeda antar-konfigurasi, walaupun rekamannya sama (47–67% rekaman tertulis apa
  adanya). Di GPU hasilnya stabil, jadi lakukan testing yang membandingkan hasil di GPU.
- Kalimat campur bahasa Indonesia-Inggris bisa rusak. Whisper juga hanya mengenali bahasa Inggris.
- Kapasitas tanpa GPU kecil (lihat `reports/load_test.md`).
- Link demo berganti setiap kali dijalankan, dan laptop harus tetap menyala.

## Yang belum diuji

- Perekaman dari HP sungguhan.
- `docker build` (Docker tidak terpasang; akan berjalan di CI setelah repo di-push).
- Suara anak.
