import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter
import re

# Mengatur style seaborn untuk plot yang lebih menarik
sns.set(style='darkgrid')

# --- FUNGSI-FUNGSI UNTUK MEMUAT DAN MENGOLAH DATA DENGAN CACHING ---

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
    
    wins_a = df_arena[df_arena['winner'] == 'model_a']['model_a'].value_counts()
    wins_b = df_arena[df_arena['winner'] == 'model_b']['model_b'].value_counts()
    total_wins = wins_a.add(wins_b, fill_value=0)
    
    appearances_a = df_arena['model_a'].value_counts()
    appearances_b = df_arena['model_b'].value_counts()
    total_appearances = appearances_a.add(appearances_b, fill_value=0)
    
    win_rate = (total_wins / total_appearances).sort_values(ascending=False)
    return win_rate

@st.cache_data
def analyze_user_topics(_df):
    """Menganalisis dan mengembalikan DataFrame berisi topik/kata kunci teratas dari percakapan pengguna."""
    user_texts = []
    for conv in _df['conversation']:
        for message in conv:
            if message['role'] == 'user':
                user_texts.append(message['content'])

    full_text = ' '.join(user_texts).lower()
    full_text = re.sub(r'[^\w\s]', '', full_text)

    stopwords = set(['the', 'a', 'an', 'in', 'is', 'it', 'of', 'and', 'to', 'for', 'on', 'with', 'that', 'this', 'i', 'you', 'me', 'my', 'what', 'who', 'when', 'where', 'why', 'how', 'can', 'please', 'tell', 'give', 'about', 'some', 'do', 'does', 'did', 'are', 'was', 'were', 'be', 'been'])
    words = [word for word in full_text.split() if word not in stopwords and len(word) > 2]

    word_counts = Counter(words)
    top_20_words = word_counts.most_common(20)
    
    df_top_words = pd.DataFrame(top_20_words, columns=['Kata Kunci', 'Frekuensi'])
    return df_top_words

# --- MEMUAT DATA ---
df_conv = load_conversation_data()
win_rate_df = load_arena_data()
df_top_words = analyze_user_topics(df_conv) # Menganalisis topik

# --- HEADER ---
st.title('📊 Dasbor Analisis Penggunaan Model AI (LLM)')

# --- SIDEBAR ---
st.sidebar.header("Filter Data")
top_10_popular = df_conv['model'].value_counts().nlargest(10).index.tolist()
selected_models = st.sidebar.multiselect(
    'Pilih Model AI:',
    options=df_conv['model'].unique(),
    default=top_10_popular
)
df_filtered = df_conv[df_conv['model'].isin(selected_models)] if selected_models else df_conv.copy()

# --- MAIN PAGE ---

# 1. Tingkat Kemenangan
st.header("🏆 1. Tingkat Kemenangan Model (Proxy Kepuasan Pengguna)")
fig_win, ax_win = plt.subplots(figsize=(12, 8))
top_15_win_rate = win_rate_df.nlargest(15)
sns.barplot(x=top_15_win_rate.values, y=top_15_win_rate.index, palette='rocket', ax=ax_win)
ax_win.set_title('Tingkat Kemenangan (Win Rate) Model AI Teratas')
ax_win.set_xlabel('Win Rate')
ax_win.set_ylabel('Model AI')
st.pyplot(fig_win)

# 2. Analisis Topik
st.header("💬 2. Topik Utama Percakapan Pengguna")
st.write("Kata kunci yang paling sering muncul dari permintaan pengguna. Ini menunjukkan topik dan tugas yang paling diminati.")
df_top_words_filtered = analyze_user_topics(df_filtered) # Analisis ulang topik berdasarkan filter
fig_topic, ax_topic = plt.subplots(figsize=(12, 8))
sns.barplot(x='Frekuensi', y='Kata Kunci', data=df_top_words_filtered, palette='inferno', ax=ax_topic)
ax_topic.set_title('Top 20 Topik/Tugas yang Paling Sering Diminta')
ax_topic.set_xlabel('Frekuensi')
ax_topic.set_ylabel('Kata Kunci')
st.pyplot(fig_topic)


# 3. Analisis Popularitas dan Panjang Percakapan
st.header("📊 3. Analisis Detail Interaksi")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Popularitas Model")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(y=df_filtered['model'], order=df_filtered['model'].value_counts().index, palette='viridis', ax=ax)
    ax.set_title('Jumlah Percakapan per Model')
    ax.set_xlabel('Jumlah Percakapan')
    ax.set_ylabel('Model AI')
    st.pyplot(fig)

with col2:
    st.subheader("Panjang Percakapan")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x='turn', y='model', data=df_filtered, order=df_filtered['model'].value_counts().index, palette='flare', ax=ax)
    ax.set_title('Distribusi Jumlah Giliran per Model')
    ax.set_xlabel('Jumlah Giliran')
    ax.set_ylabel('')
    ax.set_xlim(0, df_filtered['turn'].quantile(0.95))
    st.pyplot(fig)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("Proyek Analisis Data Penggunaan Model AI.")