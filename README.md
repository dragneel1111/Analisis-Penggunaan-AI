# 📊 Analisis Data Penggunaan LLM

Proyek ini menganalisis penggunaan **Large Language Models (LLM)** menggunakan dataset publik **[lmsys/lmsys-arena-human-preference-55k](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k)**.  
Analisis dilakukan dalam bentuk **Jupyter Notebook** dan **Streamlit Dashboard** interaktif.

---

## 🚀 Fitur Analisis
1. **Popularitas Model** – model mana yang paling sering digunakan.  
2. **Topik Utama (n-gram)** – kata kunci/tema terbanyak dari pesan pengguna.  
3. **Win-Rate (dengan Wilson CI)** – model mana yang lebih disukai pengguna.  
4. **Turns-to-Solve (TTS)** – efisiensi model dalam menyelesaikan percakapan.  
5. **Fit-for-Purpose (Model × Topik)** – model unggul per kategori tugas (Coding, Penulisan, Analisis Data, Terjemahan).  

---

## 🛠️ Teknologi
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- Streamlit
- Jupyter Notebook


## ▶️ Cara Menjalankan

### 1) Clone repo & install requirements
```bash
git clone <repo-anda>
cd <repo-anda>
pip install -r requirements.txt
````

### 2) Siapkan Hugging Face Token (wajib, karena dataset gated)

* Daftar/login di [Hugging Face](https://huggingface.co/)
* Minta akses ke dataset `lmsys/lmsys-arena-human-preference-55k`
* Buat **Access Token** (format: `hf_xxx`)

#### Opsi A — via environment variable

```bash
export HF_TOKEN="hf_xxx"   # Linux / MacOS
setx HF_TOKEN "hf_xxx"     # Windows PowerShell
```

#### Opsi B — via Streamlit secrets

Buat file `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_xxx"
```

### 3) Jalankan Streamlit Dashboard

```bash
streamlit run dashboard.py
```

### 4) Jalankan Jupyter Notebook

```bash
jupyter notebook Proyek_Analisis_Data.ipynb
```

---

## ✅ Hasil & Insight

* **Model dominan** digunakan untuk coding, penulisan, dan analisis data.
* **Win-Rate** antar model bervariasi, perlu diperhatikan CI untuk interpretasi.
* **TTS** menunjukkan model tertentu lebih efisien menyelesaikan percakapan.
* **Fit-for-Purpose** menegaskan keunggulan tiap model berbeda per kategori → peluang untuk **routing otomatis** & **bundling produk**.

