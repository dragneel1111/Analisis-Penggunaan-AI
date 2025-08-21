# Proyek Analisis Data: Analisis Penggunaan Model Bahasa (LLM)

Proyek ini merupakan submission untuk kelas **Belajar Analisis Data dengan Python** di Dicoding. Fokus analisis adalah untuk memahami tren penggunaan, pola interaksi, dan tingkat kepuasan pengguna terhadap berbagai *Large Language Models* (LLM).

## Latar Belakang

Dengan semakin populernya teknologi AI generatif, muncul berbagai model bahasa dari banyak perusahaan. Memahami model mana yang paling populer, bagaimana pengguna berinteraksi dengannya, dan mana yang paling disukai menjadi krusial bagi pengembang, peneliti, dan bisnis. Analisis ini mencoba menjawab pertanyaan-pertanyaan tersebut dengan menggunakan dataset publik dari **LMSYS Org**.

## Pertanyaan Bisnis

1.  **Popularitas Model:** Model AI mana yang paling populer digunakan oleh pengguna?
2.  **Pola Interaksi:** Bagaimana pola interaksi pengguna, seperti panjang percakapan dan bahasa yang digunakan, pada berbagai model AI?
3.  **Kepuasan Pengguna:** Model AI mana yang memiliki tingkat kepuasan atau preferensi tertinggi dari pengguna saat diadu satu sama lain?

## Sumber Data

Analisis ini menggunakan dua dataset dari **Hugging Face**:
1.  **[LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m):** Berisi 1 juta percakapan anonim dengan 25 model AI. Digunakan untuk menganalisis popularitas dan pola interaksi.
2.  **[Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations):** Berisi data percakapan dengan sistem *voting* preferensi pengguna. Digunakan untuk menganalisis tingkat kepuasan.

## File & Direktori

-   **`Proyek_Analisis_Data.ipynb`**: Notebook utama yang berisi seluruh proses analisis data, mulai dari pemuatan, pembersihan, hingga visualisasi data.
-   **`dashboard/`**: Folder yang berisi aplikasi web interaktif Streamlit.
    -   **`dashboard.py`**: File utama untuk menjalankan dasbor Streamlit.
-   **`requirements.txt`**: Daftar pustaka (library) Python yang dibutuhkan untuk menjalankan proyek.
-   **`README.md`**: Dokumen ini.

## Cara Menjalankan Proyek

1.  **Clone Repositori (Opsional)**
    Jika Anda mengunduh repositori ini, pastikan Anda berada di direktori yang benar.

2.  **Buat Lingkungan Virtual**
    Sangat disarankan untuk membuat lingkungan virtual untuk menghindari konflik dependensi.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk macOS/Linux
    venv\Scripts\activate  # Untuk Windows
    ```

3.  **Instal Dependensi**
    Instal semua pustaka yang dibutuhkan dengan menjalankan perintah berikut di terminal:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Notebook Analisis**
    Buka dan jalankan file `Proyek_Analisis_Data.ipynb` menggunakan Jupyter Notebook atau lingkungan serupa seperti VS Code. Pastikan untuk melakukan autentikasi Hugging Face sesuai petunjuk di dalam notebook untuk mengakses dataset.

5.  **Jalankan Dasbor Streamlit**
    Untuk melihat dasbor interaktif, navigasikan ke folder `dashboard/` dan jalankan perintah berikut:
    ```bash
    cd dashboard
    streamlit run dashboard.py
    ```

## Hasil Analisis (Kesimpulan)

Berdasarkan analisis yang telah dilakukan, ditemukan beberapa wawasan utama:
-   **Popularitas & Kepuasan:** Terdapat korelasi kuat antara popularitas, engagement (panjang percakapan), dan tingkat kepuasan pengguna. Model seperti **`gpt-4`** dan **`claude-1`** secara konsisten mendominasi di ketiga metrik tersebut, menunjukkan kepemimpinan pasar yang jelas.
-   **Pola Interaksi:** Model yang lebih canggih cenderung digunakan untuk percakapan yang lebih panjang dan kompleks, sementara bahasa yang dominan digunakan adalah Bahasa Inggris, menandakan audiens global.
-   **Tingkat Kemenangan:** Analisis *win rate* dari Chatbot Arena mengonfirmasi bahwa model-model teratas tidak hanya sering digunakan, tetapi juga lebih disukai kualitasnya saat dibandingkan secara langsung oleh pengguna.