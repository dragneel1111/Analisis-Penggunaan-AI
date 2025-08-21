# Proyek Analisis Data: Analisis Penggunaan Large Language Models (LLM)

## 🚀 Latar Belakang Proyek

Seiring dengan pesatnya perkembangan *Large Language Models* (LLM), analisis mengenai bagaimana model-model ini digunakan dan dipersepsikan oleh pengguna menjadi sangat penting. Proyek ini bertujuan untuk menganalisis data percakapan dari berbagai model AI untuk memahami tren popularitas, pola interaksi, dan tingkat kepuasan pengguna. Analisis ini krusial bagi pengembang, peneliti, dan bisnis untuk memahami lanskap kompetitif dan preferensi pengguna di dunia AI generatif.

## ❓ Pertanyaan Bisnis

Analisis ini dirancang untuk menjawab tiga pertanyaan bisnis utama:

1.  **Popularitas Model:** Model AI mana yang paling sering digunakan atau paling populer di antara pengguna?
2.  **Analisis Topik Percakapan:** Apa saja topik atau jenis tugas yang paling sering diminta oleh pengguna kepada model-model AI tersebut (misalnya: pemrograman, penulisan kreatif, tanya jawab umum)?
3.  **Perbandingan Kualitas & Kepuasan:** Apakah ada perbedaan dalam panjang percakapan atau pola interaksi yang menunjukkan kepuasan pengguna? Dan, model mana yang memiliki tingkat kepuasan (win rate) tertinggi?

## 💾 Sumber Data

Analisis ini menggunakan dua dataset publik yang disediakan oleh **Large Model Systems Organization (LMSYS Org)** melalui Hugging Face:

1.  **[LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m):** Berisi 1 juta percakapan anonim dari 25 model AI terkemuka. Dataset ini digunakan untuk menganalisis popularitas dan pola interaksi.
2.  **[Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations):** Berisi percakapan dengan sistem pemungutan suara (voting), di mana pengguna memilih respons AI yang lebih baik. Dataset ini digunakan sebagai proksi untuk mengukur kepuasan dan preferensi pengguna.

## 🛠️ Teknologi & Pustaka yang Digunakan

-   **Bahasa:** Python 3
-   **Analisis Data:** Pandas, NumPy
-   **Visualisasi Data:** Matplotlib, Seaborn
-   **Dasbor Interaktif:** Streamlit
-   **Akses Data:** Datasets (dari Hugging Face)
-   **Lingkungan:** Jupyter Notebook
