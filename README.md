# Karhutla Intelligence – WebGIS (Leaflet)

Dashboard simulasi disertasi karhutla untuk AOI **KHG Sungai Sugihan – Sungai Lumpur (Air Sugihan), Kab. OKI**.

- Satu file `index.html` (siap GitHub Pages: Settings → Pages → main / root). Batas AOI sudah tertanam dari `Air_Sugihan.kml` (disederhanakan ±20 m); KML lain bisa dimuat lewat menu **Unggah KML AOI** (grid, peta kerawanan, dan simulasi menyesuaikan otomatis).
- Peta dasar Esri (Satelit/Gelap/Jalan/Topografi), tanpa API key.
- Hotspot: isi MAP_KEY FIRMS di panel kontrol (disimpan hanya di localStorage). **Jangan menulis key di kode/repo publik.** Alternatif: unggah CSV hasil unduhan FIRMS.
- Yang masih simulasi: parameter fisis (data acak), PM2.5 (estimasi dari kepadatan partikel), koordinat kota (perkiraan).
