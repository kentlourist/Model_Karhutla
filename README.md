# Karhutla Intelligence — OKI prototype

Dashboard geospasial untuk demonstrasi penelitian karhutla di Kabupaten Ogan Komering Ilir (OKI), Sumatera Selatan. Berjalan sebagai situs statis GitHub Pages; JavaScript berjalan di browser.

## Isi
- `index.html`: dashboard multi-halaman
- `style.css`: desain responsif
- `app.js`: data sintetis, model demonstratif, Leaflet, hotspot, fire-spread dan plume-smoke overlay

## Publikasi GitHub Pages
1. Buat repository publik di https://github.com/new.
2. Unggah tiga berkas utama (`index.html`, `style.css`, `app.js`) dan README ke root repository.
3. Buka Settings → Pages.
4. Pilih **Deploy from a branch**, branch `main`, folder `/(root)`, lalu Save.
5. Buka URL yang ditampilkan GitHub Pages, umumnya `https://USERNAME.github.io/NAMA-REPO/`.

## Peta dan data
Peta Leaflet menggunakan tile publik OpenStreetMap; tersedia pilihan OpenTopoMap. Pusat peta dan cakupan koordinat adalah cakupan kajian perkiraan untuk OKI, bukan poligon batas administrasi resmi. Titik hotspot dihasilkan secara sintetis dan diberi label SIM-xxx. Jangan menyebut titik tersebut sebagai deteksi FIRMS aktual.

## Keamanan NASA FIRMS MAP_KEY
Jangan meletakkan MAP_KEY dalam `app.js`, HTML, atau repository publik. Kunci di frontend bisa dilihat siapa pun melalui browser developer tools atau source code. Versi ini sengaja tidak menyertakan key apa pun dan belum memanggil API FIRMS. Untuk data aktual di GitHub Pages:
1. Unduh data hotspot dari NASA FIRMS sebagai CSV menggunakan akun/kunci Anda di sisi pribadi.
2. Periksa metadata, periode, confidence, FRP, sumber sensor, lisensi dan aturan penggunaan.
3. Simpan data yang memang boleh dipublikasikan sebagai file data publik terpisah, misalnya `data/hotspots.csv`, setelah memastikan tidak ada rahasia di dalamnya.
4. Tambahkan parser CSV dan ubah generator sintetis agar membaca berkas itu. Perlu dicatat bahwa file data publik dapat diunduh siapa pun.
Untuk pembaruan API otomatis dengan kunci rahasia, diperlukan backend atau workflow terjadwal yang menyimpan secret; itu bukan fungsi website statis GitHub Pages semata.

## Keterbatasan ilmiah
- Indikator, label, confidence, dan titik saat ini sintetis.
- Implementasi model JavaScript merupakan demonstrasi algoritma ringkas, bukan pengganti pustaka ML tervalidasi. Gradient Boosting bukan XGBoost penuh; SVM adalah linear.
- Batas peta hanya extent perkiraan, bukan batas resmi Kabupaten OKI.
- Cellular automata dan smoke overlay disederhanakan. Plume asap bersifat ilustratif, bukan prakiraan konsentrasi PM2.5, transport atmosfer terkalibrasi, atau paparan kesehatan.
- Jangan gunakan prototipe untuk peringatan dini, penentuan operasi pemadaman, atau keputusan darurat.

## Roadmap
1. Tambahkan GeoJSON batas kabupaten/desa resmi dan layer data geospasial berlisensi.
2. Integrasikan hotspot FIRMS melalui CSV publik yang tervalidasi.
3. Integrasikan data area terbakar dan faktor lingkungan temporal.
4. Lakukan validasi spasial-temporal, kalibrasi probabilitas, dan analisis ketidakpastian.
5. Untuk asap, gunakan model transport/dispersi tervalidasi dan observasi kualitas udara.
