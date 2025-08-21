import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Mengatur style seaborn untuk plot yang lebih menarik
sns.set(style='darkgrid')

# --- FUNGSI UNTUK MEMUAT DATA DENGAN CACHING ---

@st.cache_data
def load_conversation_data(sample_size=5000):
    """Memuat dataset lmsys-chat-1m."""
    dataset = load_dataset("lmsys/lmsys-chat-1m", split=f'train[:{sample_size}]')
    df = pd.DataFrame(dataset)
    return df

@st.cache_data
def load_arena_data():
    """Memuat dataset lmsys/chatbot_arena_conversations dan menghitung win rate."""
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split='train')
    df_arena = pd.DataFrame(dataset)
    
    # Menghitung total kemenangan
    wins_a = df_arena[df_arena['winner'] == 'model_a']['model_a'].value_counts()
    wins_b = df_arena[df_arena['winner'] == 'model_b']['model_b'].value_counts()
    total_wins = wins_a.add(wins_b, fill_value=0)
    
    # Menghitung total pertandingan
    appearances_a = df_arena['model_a'].value_counts()
    appearances_b = df_arena['model_b'].value_counts()
    total_appearances = appearances_a.add(appearances_b, fill_value=0)
    
    # Menghitung win rate dan mengurutkan
    win_rate = (total_wins / total_appearances).sort_values(ascending=False)
    
    return win_rate

# --- MEMUAT DATA ---
df_conv = load_conversation_data()
win_rate_df = load_arena_data()

# --- HEADER ---
st.title('📊 Dasbor Analisis Penggunaan Model AI (LLM)')
st.write(
    """
    Dasbor ini menganalisis data percakapan dari dataset **LMSYS-Chat-1M** dan preferensi pengguna 
    dari **Chatbot Arena**. Gunakan filter di sidebar untuk menjelajahi data.
    """
)

# --- SIDEBAR ---
st.sidebar.header("Filter Data Percakapan")
# Ambil 10 model terpopuler dari data percakapan untuk dijadikan opsi default
top_10_popular = df_conv['model'].value_counts().nlargest(10).index.tolist()
selected_models = st.sidebar.multiselect(
    'Pilih Model AI:',
    options=df_conv['model'].unique(),
    default=top_10_popular
)

# Filter dataframe utama berdasarkan model yang dipilih
if selected_models:
    df_filtered = df_conv[df_conv['model'].isin(selected_models)]
else:
    df_filtered = df_conv.copy()

# --- MAIN PAGE ---

# 1. Visualisasi Tingkat Kemenangan (Win Rate)
st.header("🏆 1. Tingkat Kemenangan Model (Proxy Kepuasan Pengguna)")
st.write(
    "Grafik ini menunjukkan model mana yang paling sering dipilih sebagai pemenang oleh pengguna "
    "saat diadu satu sama lain. *Win rate* yang tinggi adalah indikator kuat kepuasan pengguna."
)

fig_win, ax_win = plt.subplots(figsize=(12, 8))
top_15_win_rate = win_rate_df.nlargest(15)
sns.barplot(
    x=top_15_win_rate.values,
    y=top_15_win_rate.index,
    palette='rocket',
    ax=ax_win
)
ax_win.set_title('Tingkat Kemenangan (Win Rate) Model AI Teratas')
ax_win.set_xlabel('Win Rate (Menang / Total Pertandingan)')
ax_win.set_ylabel('Model AI')
st.pyplot(fig_win)

# 2. Visualisasi Popularitas Model
st.header("💬 2. Popularitas Model Berdasarkan Jumlah Percakapan")
st.write("Jumlah percakapan untuk setiap model AI yang dipilih. Grafik ini menunjukkan frekuensi penggunaan.")

fig_pop, ax_pop = plt.subplots(figsize=(12, 8))
sns.countplot(
    y=df_filtered['model'],
    order=df_filtered['model'].value_counts().index,
    palette='viridis',
    ax=ax_pop
)
ax_pop.set_title('Jumlah Percakapan per Model AI')
ax_pop.set_xlabel('Jumlah Percakapan')
ax_pop.set_ylabel('Model AI')
ax_pop.tick_params(axis='y', labelsize=10)
st.pyplot(fig_pop)

# 3. Analisis Bahasa dan Panjang Percakapan
st.header("🌐 3. Analisis Detail Percakapan")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Bahasa")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(
        y=df_filtered['language'],
        order=df_filtered['language'].value_counts().nlargest(10).index,
        palette='crest',
        ax=ax
    )
    ax.set_title('Top 10 Bahasa yang Digunakan')
    ax.set_xlabel('Jumlah Percakapan')
    ax.set_ylabel('Bahasa')
    st.pyplot(fig)

with col2:
    st.subheader("Distribusi Panjang Percakapan")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(
        x='turn',
        y='model',
        data=df_filtered,
        order=df_filtered['model'].value_counts().index,
        palette='flare',
        ax=ax
    )
    ax.set_title('Distribusi Jumlah Giliran per Model')
    ax.set_xlabel('Jumlah Giliran')
    ax.set_ylabel('Model AI')
    ax.set_xlim(0, df_filtered['turn'].quantile(0.95)) # Batasi sumbu x agar boxplot lebih mudah dibaca
    st.pyplot(fig)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("Proyek Analisis Data Penggunaan Model AI.")