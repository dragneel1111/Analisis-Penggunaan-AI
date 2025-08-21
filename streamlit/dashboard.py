import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import re
import warnings

# Bungkam FutureWarning default dari pandas
warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated and will be changed to True in a future version of pandas",
    category=FutureWarning
)

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="Dashboard Analisis Data Penggunaan LLM",
    page_icon="🤖",
    layout="wide"
)

st.title("📊 Dashboard Analisis Data Penggunaan LLM")

# --- FUNGSI-FUNGSI UNTUK MEMUAT DAN MENGOLAH DATA ---
@st.cache_data(show_spinner=True, ttl=3600)
def load_conversation_data(sample_size=5000):
    dataset = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
    df_raw = pd.DataFrame(dataset)
    if sample_size and sample_size < len(df_raw):
        df_raw = df_raw.sample(sample_size, random_state=42).reset_index(drop=True)

    # 🔑 Normalisasi: gabungkan (model, conversation) dari A & B
    df_long_a = df_raw[["winner_model", "model_a", "conversation_a"]].rename(
        columns={"model_a": "model", "conversation_a": "conversation"}
    )
    df_long_b = df_raw[["winner_model", "model_b", "conversation_b"]].rename(
        columns={"model_b": "model", "conversation_b": "conversation"}
    )

    df_long = pd.concat([df_long_a, df_long_b], ignore_index=True)
    return df_long

@st.cache_data(show_spinner=True, ttl=3600)
def compute_win_rate(df):
    appearances = df['model'].value_counts()
    wins = df.groupby('winner_model').size()
    win_rate = (wins / appearances).dropna()
    return win_rate

@st.cache_data(show_spinner=True, ttl=3600)
def compute_avg_turns(df):
    avg_turns = df.groupby('model')['conversation'].apply(lambda g: g.apply(len).mean())
    return avg_turns

# --- HELPER: proxy "beres" & kategori topik (untuk TTS & Fit-for-Purpose) ---
OK_PAT = re.compile(
    r"(thanks|thank you|terima kasih|berhasil|works|solved|mantap|fixed?|oke+|ok\s*(udah|siap)?|sip|resolved)",
    re.IGNORECASE
)

TOPIC_RULES = {
    "Coding": re.compile(r"\b(code|coding|bug|function|class|method|api|regex|python|javascript|java|ts|typescript|cpp|golang|php|html|css|framework|compile|error)\b", re.I),
    "Analisis Data": re.compile(r"\b(data|dataset|pandas|numpy|stat(istik|s)?|regression|cluster|model(ing)?|visualisasi|plot|chart|csv|etl|eda)\b", re.I),
    "Terjemahan": re.compile(r"\b(translate|translat(e|ion)|terjemah|alih ?bahasa|english to indonesian|indonesian to english|b\.?inggris|b\.?indonesia)\b", re.I),
    "Penulisan": re.compile(r"\b(tulis|menulis|writing|essay|artikel|copy|caption|paragraf|ringkas|rangkuman|summary|email|surat|konten)\b", re.I),
}

def _user_text_from_conv(conv):
    parts = []
    for msg in conv:
        if msg.get("role") == "user":
            parts.append((msg.get("content") or "").strip())
    return " ".join(parts)

def is_solved(conv):
    for msg in reversed(conv):
        if msg.get("role") == "user":
            return bool(OK_PAT.search((msg.get("content") or "").lower()))
    return False

def topic_category_from_text(text):
    for label in ["Coding", "Analisis Data", "Terjemahan", "Penulisan"]:
        if TOPIC_RULES[label].search(text):
            return label
    return "Lainnya"

def add_derived_columns(df):
    df = df.copy()
    df["user_text"] = df["conversation"].apply(_user_text_from_conv)
    df["is_solved"] = df["conversation"].apply(is_solved)
    if "topic_category" not in df.columns:
        df["topic_category"] = df["user_text"].apply(topic_category_from_text)
    if "turn" not in df.columns:
        df["turn"] = df["conversation"].apply(lambda conv: len(conv) if isinstance(conv, (list, tuple)) else None)
    return df

# --- LOAD DATA ---
sample_size = st.sidebar.slider("Sample Size", min_value=1000, max_value=50000, value=5000, step=1000)
df = load_conversation_data(sample_size=sample_size)

# --- FILTER MODEL POPULER ---
top_n = st.sidebar.slider("Top-N Model Populer", min_value=5, max_value=30, value=10, step=1)
top_models = df['model'].value_counts().head(top_n).index
df_filtered = df[df['model'].isin(top_models)]

# --- VISUALISASI 1: POPULARITAS MODEL ---
st.header("1. Popularitas Model")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.countplot(data=df_filtered, x="model", order=df_filtered['model'].value_counts().index, ax=ax1)
ax1.set_title("Popularitas Model (Top-N)")
ax1.set_xlabel("Model")
ax1.set_ylabel("Jumlah Percakapan")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig1)

# --- VISUALISASI 2: TOPIK ---
st.header("2. Topik yang Dibahas")
user_texts = []
for conv in df_filtered["conversation"]:
    for msg in conv:
        if msg["role"] == "user":
            user_texts.append(msg["content"].lower())
text = " ".join(user_texts)
words = re.findall(r"\b\w+\b", text)
stopwords = set(["the","and","to","a","i","of","in","for","is","it","that","you","on","this","with","as","are","be","can","my","me"])
filtered_words = [w for w in words if len(w)>3 and w not in stopwords]
word_freq = pd.Series(filtered_words).value_counts().head(20)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=word_freq.values, y=word_freq.index, ax=ax2)
ax2.set_title("20 Kata Teratas dari User")
ax2.set_xlabel("Frekuensi")
ax2.set_ylabel("Kata")
st.pyplot(fig2)

# --- VISUALISASI 3: WIN-RATE VS PANJANG PERCAKAPAN ---
st.header("3. Win-Rate vs Panjang Percakapan")
win_rate = compute_win_rate(df_filtered)
avg_turns = compute_avg_turns(df_filtered)
comparison = pd.DataFrame({"Win-Rate": win_rate, "Avg Turns": avg_turns})
comparison = comparison.dropna()

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x="Avg Turns", y="Win-Rate", data=comparison, s=80, ax=ax3)
for model, row in comparison.iterrows():
    ax3.text(row["Avg Turns"], row["Win-Rate"], model, fontsize=9)
ax3.set_title("Win-Rate vs Panjang Percakapan")
ax3.set_xlabel("Rata-rata Turns")
ax3.set_ylabel("Win-Rate")
st.pyplot(fig3)

# === Derivasi metrik baru: TTS & Fit-for-Purpose ===
df_feat = add_derived_columns(df_filtered)

# 1) TTS
df_solved = df_feat[df_feat["is_solved"]]
tts_stats = (
    df_solved
    .groupby("model", observed=True)["turn"]
    .agg(n_solved="count", mean="mean", median="median", p75=lambda s: s.quantile(0.75))
    .sort_values("median")
)

# 2) Heatmap Model × Topik
TOP_N_HEAT = st.sidebar.number_input("Top-N model untuk heatmap", min_value=4, max_value=20, value=8, step=1)
top_models_for_heat = df_feat["model"].value_counts().head(int(TOP_N_HEAT)).index

perf = (
    df_feat[df_feat["model"].isin(top_models_for_heat)]
    .groupby(["topic_category", "model"], observed=True)
    .agg(n=("model", "size"), solved_rate=("is_solved", "mean"))
    .reset_index()
)
heat = perf.pivot(index="topic_category", columns="model", values="solved_rate")

# --- SECTION 4: TTS ---
st.header("⏱️ 4. Turns-to-Solve (TTS)")
st.write("""
TTS = rata-rata jumlah giliran (turn) pada percakapan yang **berujung 'beres'** 
(proxy bahasa: thanks/terima kasih/berhasil/works/dll). Lebih rendah = lebih efisien.
""")

if tts_stats is None or tts_stats.empty:
    st.info("Belum ada percakapan 'beres' pada filter saat ini.")
else:
    st.subheader("Ringkasan TTS per Model")
    st.dataframe(tts_stats.round(2))

    fig_tts, ax_tts = plt.subplots(figsize=(10, 4.8))
    order = tts_stats.index.tolist()
    sns.barplot(x=tts_stats.index, y=tts_stats["median"], order=order, ax=ax_tts, palette="viridis")
    ax_tts.set_xlabel("")
    ax_tts.set_ylabel("Median TTS (turn)")
    ax_tts.set_title("Median TTS per Model")
    plt.xticks(rotation=20, ha="right")
    for i, model in enumerate(order):
        n = int(tts_stats.loc[model, "n_solved"])
        ax_tts.text(i, tts_stats.loc[model, "median"] + 0.1, f"n={n}", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig_tts)

# --- SECTION 5: Fit-for-Purpose ---
st.header("🧭 5. Fit-for-Purpose: Model × Topik")
st.write("""
Nilai sel = **Solved Rate (proxy)**: proporsi percakapan yang berujung 'beres' untuk kombinasi *model × topik*.  
Gunakan untuk *routing otomatis* dan ide paket bundling produk.
""")

if heat is None or heat.empty:
    st.info("Data tidak cukup untuk heatmap.")
else:
    fig_hm, ax_hm = plt.subplots(figsize=(min(12, 2 + 1.2*len(heat.columns)), 6))
    sns.heatmap(heat.fillna(0), cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".0%")
    ax_hm.set_xlabel("Model")
    ax_hm.set_ylabel("Kategori Topik")
    ax_hm.set_title("Solved Rate (Proxy) — Model × Topik")
    st.pyplot(fig_hm)

    MIN_N = st.sidebar.number_input("Ambang N juara per topik", min_value=10, max_value=200, value=30, step=5)
    leaders = (
        perf[perf["n"] >= int(MIN_N)]
        .sort_values(["topic_category", "solved_rate"], ascending=[True, False])
        .groupby("topic_category", observed=True)
        .head(1)
        .reset_index(drop=True)
    )

    st.caption(f"Hanya sel dengan N ≥ {int(MIN_N)} dipakai untuk 'juara per topik'.")
    if leaders.empty:
        st.info("Belum ada topik yang memenuhi ambang N.")
    else:
        st.subheader("🏆 Juara per Topik")
        st.dataframe(
            leaders.assign(solved_rate=(leaders["solved_rate"]*100).round(1)).rename(
                columns={"topic_category":"Topik", "model":"Model", "n":"N", "solved_rate":"Solved Rate (%)"}
            )[["Topik","Model","N","Solved Rate (%)"]]
        )

# --- FOOTER ---
st.markdown("---")
st.caption("Analisis Data Penggunaan LLM — Dashboard Streamlit")
