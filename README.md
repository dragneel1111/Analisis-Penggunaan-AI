````markdown
# 📊 Proyek Analisis Data Penggunaan LLM

Proyek ini menganalisis data dari dataset **LMSYS Arena Human Preference** untuk memahami pola penggunaan, preferensi, dan performa berbagai model LLM (Large Language Models).  
Analisis dilakukan dalam dua bentuk:
- **Notebook Jupyter (`Proyek_Analisis_Data.ipynb`)** → eksplorasi dan visualisasi.
- **Dashboard Streamlit (`dashboard.py`)** → interaktif untuk tim bisnis/eksekutif.

---

## 🎯 Pertanyaan Bisnis

Analisis ini difokuskan untuk menjawab **5 pertanyaan utama**:

1. **Popularitas Model**  
   - Model mana yang paling sering digunakan/dipilih oleh user?  
   - Visualisasi: *Countplot Popularitas Model*.

2. **Topik yang Dibahas**  
   - Apa kata kunci/topik utama dalam percakapan user dengan LLM?  
   - Visualisasi: *Bar chart kata kunci teratas*.

3. **Pola Interaksi & Preferensi (Win-Rate vs Panjang Percakapan)**  
   - Apakah model dengan win-rate tinggi juga cenderung menghasilkan percakapan lebih panjang/pendek?  
   - Visualisasi: *Scatterplot Win-Rate vs Avg Turns*.

4. **Turns-to-Solve (TTS)**  
   - Rata-rata/median berapa **giliran (turn)** dibutuhkan sampai percakapan dianggap **“beres”**?  
   - Proxy “beres” = sinyal bahasa dari user (*thanks, terima kasih, berhasil, works, resolved*).  
   - Visualisasi: *Tabel ringkasan & bar chart median TTS per model*.

5. **Fit-for-Purpose (Model × Topik)**  
   - Model mana yang unggul pada kategori tugas tertentu: **Coding, Penulisan, Analisis Data, Terjemahan**?  
   - Metrik: **Solved Rate (proxy)** = proporsi percakapan yang berujung “beres” pada kombinasi *model × topik*.  
   - Visualisasi: *Heatmap Model × Topik + tabel juara per topik*.

---

## 🛠️ Metodologi Singkat

- Dataset: `lmsys/lmsys-arena-human-preference-55k` dari HuggingFace.  
- Preprocessing:
  - Normalisasi teks user.  
  - Ekstraksi kata kunci untuk topik.  
  - Labelisasi **status “beres”** berdasarkan pola bahasa.  
  - Kategorisasi topik rule-based (Coding, Analisis Data, Terjemahan, Penulisan).  
- Analisis:
  - Popularitas model (frekuensi).  
  - Topik percakapan (word frequency).  
  - Win-rate dihitung dari proporsi kemenangan per model.  
  - **TTS** dihitung dari median jumlah giliran pada percakapan yang berujung “beres”.  
  - **Solved Rate per Model × Topik** dihitung sebagai indikator kecocokan model per kategori tugas.  

> **Catatan:**  
> - Status “beres” adalah **proxy** sederhana berbasis pola bahasa; bukan ground-truth.  
> - Kategori topik masih berbasis rule sederhana → bisa dikembangkan ke BERTopic / embedding-based clustering.  
> - Untuk reliabilitas, hasil heatmap & juara per topik hanya ditampilkan jika **N ≥ 30**.

---

## 💻 Teknologi yang Dipakai

- Python 3.x  
- [pandas](https://pandas.pydata.org/)  
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/)  
- [Streamlit](https://streamlit.io/)  
- [datasets (HuggingFace)](https://huggingface.co/docs/datasets)

---

## 🚀 Menjalankan Dashboard

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
````

2. Jalankan aplikasi Streamlit:

   ```bash
   streamlit run dashboard.py
   ```
3. Buka link yang muncul (biasanya `http://localhost:8501`) di browser.

---

## 📌 Kesimpulan Umum

* Model populer didominasi oleh beberapa nama besar, dengan distribusi penggunaan yang tidak merata.
* Topik percakapan menegaskan tren coding, penulisan, dan analisis data.
* Win-rate tidak selalu sejalan dengan panjang percakapan → efisiensi berbeda-beda.
* **TTS** memberi gambaran model mana yang lebih cepat membantu user menyelesaikan tugas.
* **Fit-for-Purpose heatmap** membuka peluang *routing otomatis* & bundling produk (misalnya: paket Pro-Coding vs Pro-Writing).

```

