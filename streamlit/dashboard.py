import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import re
import warnings
from huggingface_hub import login
import os


# ---------- CONFIG & GLOBALS ----------
st.set_page_config(page_title="Dashboard Analisis Data LLM", page_icon="🤖", layout="wide")
sns.set(style="darkgrid")
warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated and will be changed to True in a future version of pandas",
    category=FutureWarning,
)

# --- Hugging Face auth (ambil dari Streamlit Secrets / env var) ---
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))

if HF_TOKEN:
    try:
        # Login sesi ini (tidak menulis ke disk)
        login(token=HF_TOKEN, add_to_git_credential=False)
        st.sidebar.success("Authenticated to Hugging Face Hub.")
    except Exception as e:
        st.sidebar.warning(f"Gagal login HF: {e}")
else:
    st.sidebar.info("HF_TOKEN tidak ditemukan di Secrets/env. Dataset gated mungkin gagal dimuat.")


OK_PAT = re.compile(
    r"(thanks|thank you|terima kasih|berhasil|works|solved|mantap|fixed?|oke+|ok|done|clear|yes|sip|resolved)",
    re.IGNORECASE,
)

TOPIC_RULES = {
    "Coding": re.compile(r"\b(code|coding|bug|function|class|method|api|regex|python|javascript|java|ts|typescript|cpp|golang|php|html|css|framework|compile|error)\b", re.I),
    "Analisis Data": re.compile(r"\b(data|dataset|pandas|numpy|stat(istik|s)?|regression|cluster|model(ing)?|visualisasi|plot|chart|csv|etl|eda)\b", re.I),
    "Terjemahan": re.compile(r"\b(translate|translat(e|ion)|terjemah|alih ?bahasa|english to indonesian|indonesian to english|b\.?inggris|b\.?indonesia)\b", re.I),
    "Penulisan": re.compile(r"\b(tulis|menulis|writing|essay|artikel|copy|caption|paragraf|ringkas|rangkuman|summary|email|surat|konten)\b", re.I),
}

# ---------- HELPERS ----------
def _user_text_from_conv(conv):
    if not isinstance(conv, (list, tuple)):
        return ""
    parts = []
    for msg in conv:
        if isinstance(msg, dict) and msg.get("role") == "user":
            parts.append((msg.get("content") or "").strip())
    return " ".join(parts)

def is_solved(conv):
    if not isinstance(conv, (list, tuple)):
        return False
    for msg in reversed(conv):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return bool(OK_PAT.search((msg.get("content") or "").lower()))
    return False

def topic_category_from_text(text: str) -> str:
    if not isinstance(text, str):
        return "Lainnya"
    for label in ["Coding", "Analisis Data", "Terjemahan", "Penulisan"]:
        if TOPIC_RULES[label].search(text):
            return label
    return "Lainnya"

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "user_text" not in df.columns:
        df["user_text"] = df["conversation"].apply(_user_text_from_conv)
    if "is_solved" not in df.columns:
        df["is_solved"] = df["conversation"].apply(is_solved)
    if "topic_category" not in df.columns:
        df["topic_category"] = df["user_text"].apply(topic_category_from_text)
    if "turn" not in df.columns:
        df["turn"] = df["conversation"].apply(lambda conv: len(conv) if isinstance(conv, (list, tuple)) else np.nan)
    return df

# ---------- DATA LOADERS ----------
@st.cache_data(show_spinner=True, ttl=3600)
def load_chat1m(sample_rows: int = 20000) -> pd.DataFrame:
    # Perhatikan: argumen yang dianjurkan adalah `token=...`
    ds = load_dataset("lmsys/lmsys-chat-1m", split=f"train[:{sample_rows}]", token=HF_TOKEN if HF_TOKEN else None)
    df = pd.DataFrame(ds)
    keep_cols = [c for c in ["model", "conversation"] if c in df.columns]
    df = df[keep_cols].dropna(subset=keep_cols)
    return add_derived_columns(df)


@st.cache_data(show_spinner=True, ttl=3600)
def load_arena(sample_rows: int = 40000):
    """
    lmsys/chatbot_arena_conversations → gunakan untuk Win-Rate, juga bisa untuk TTS & Heatmap.
    Kembalikan:
      - df_long: kolom (model, conversation) + turunan,
      - win_rate: Series per model.
    """
    ds = load_dataset("lmsys/chatbot_arena_conversations", split=f"train[:{sample_rows}]")
    df_raw = pd.DataFrame(ds)

    # Pastikan kolom tersedia
    need = {"model_a", "model_b", "conversation_a", "conversation_b", "winner"}
    missing = need - set(df_raw.columns)
    if missing:
        # Jika skema berubah, kembalikan kosong agar UI menampilkan info
        return pd.DataFrame(columns=["model", "conversation"]), pd.Series(dtype=float)

    # Hitung wins per model dari raw (tanpa duplikasi)
    wins_a = df_raw.loc[df_raw["winner"] == "model_a", "model_a"].value_counts()
    wins_b = df_raw.loc[df_raw["winner"] == "model_b", "model_b"].value_counts()
    wins = wins_a.add(wins_b, fill_value=0)

    # Appearances per model
    appearances_a = df_raw["model_a"].value_counts()
    appearances_b = df_raw["model_b"].value_counts()
    appearances = appearances_a.add(appearances_b, fill_value=0)

    win_rate = (wins / appearances).sort_values(ascending=False)

    # Long format untuk analitik lain
    df_a = df_raw[["model_a", "conversation_a"]].rename(columns={"model_a": "model", "conversation_a": "conversation"})
    df_b = df_raw[["model_b", "conversation_b"]].rename(columns={"model_b": "model", "conversation_b": "conversation"})
    df_long = pd.concat([df_a, df_b], ignore_index=True)
    df_long = df_long.dropna(subset=["model", "conversation"])
    df_long = add_derived_columns(df_long)
    return df_long, win_rate

@st.cache_data(show_spinner=True, ttl=3600)
def analyze_user_topics(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Kembalikan DF kata kunci teratas dari pesan user."""
    if df.empty:
        return pd.DataFrame(columns=["Kata Kunci", "Frekuensi"])
    user_texts = []
    for conv in df["conversation"]:
        if not isinstance(conv, (list, tuple)):
            continue
        for msg in conv:
            if isinstance(msg, dict) and msg.get("role") == "user":
                txt = (msg.get("content") or "").lower()
                user_texts.append(txt)
    all_text = " ".join(user_texts)
    # quick cleanup
    words = re.findall(r"[a-zA-Z]{3,}", all_text)
    stop = {
        "the","and","for","with","that","this","from","your","have","you","will","just",
        "does","did","can","could","would","there","here","into","them","then","than",
        "what","when","where","which","some","about","like","been","were","they","their",
        "kami","anda","yang","dengan","untuk","atau","dari","pada","dalam","akan","saya"
    }
    words = [w for w in words if w not in stop]
    vc = pd.Series(words).value_counts().head(top_k)
    return pd.DataFrame({"Kata Kunci": vc.index, "Frekuensi": vc.values})

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Pengaturan Data")
sample_chat = st.sidebar.slider("Sample rows Chat-1M", 5_000, 100_000, 20_000, step=5_000)
sample_arena = st.sidebar.slider("Sample rows Arena", 5_000, 100_000, 40_000, step=5_000)

# Load data
with st.spinner("Memuat data..."):
    try:
        df_chat = load_chat1m(sample_chat)
    except Exception as e:
        st.sidebar.warning(f"Gagal memuat Chat-1M: {e}")
        df_chat = pd.DataFrame(columns=["model", "conversation", "user_text", "is_solved", "topic_category", "turn"])
    try:
        df_arena, win_rate = load_arena(sample_arena)
    except Exception as e:
        st.sidebar.warning(f"Gagal memuat Arena: {e}")
        df_arena = pd.DataFrame(columns=["model", "conversation", "user_text", "is_solved", "topic_category", "turn"])
        win_rate = pd.Series(dtype=float)

# Filter model global (gabungan daftar model dari keduanya)
all_models = pd.Index(df_chat.get("model", pd.Series(dtype=object))).append(
    pd.Index(df_arena.get("model", pd.Series(dtype=object)))
).dropna().unique()

st.sidebar.markdown("---")
selected_models = st.sidebar.multiselect(
    "Filter Model (terapkan ke semua grafik)",
    options=sorted(all_models.tolist()),
    default=sorted(pd.Series(all_models).value_counts().head(10).index.tolist())
)

def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not selected_models:
        return df
    if "model" in df.columns:
        return df[df["model"].isin(selected_models)]
    return df

df_chat_f = apply_filter(df_chat)
df_arena_f = apply_filter(df_arena)

# Pilih sumber untuk TTS & Heatmap
st.sidebar.markdown("---")
src_choice = st.sidebar.radio("Sumber untuk TTS & Heatmap", ["Chat-1M", "Arena"], index=1 if not df_chat_f.empty else 0)
src_df = df_chat_f if src_choice == "Chat-1M" else df_arena_f

# ---------- HEADER ----------
st.title("📊 Dashboard Analisis Penggunaan LLM")

# ---------- SECTION 1: Popularitas Model (Chat-1M / fallback Arena) ----------
st.header("1) Popularitas Model")
base_pop = df_chat_f if not df_chat_f.empty else df_arena_f
if base_pop.empty:
    st.info("Data untuk popularitas model belum tersedia (cek koneksi/limit sampel).")
else:
    order = base_pop["model"].value_counts().index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=base_pop, x="model", order=order, ax=ax)
    ax.set_title("Popularitas Model (berdasar jumlah percakapan)")
    ax.set_xlabel("Model"); ax.set_ylabel("Jumlah Percakapan")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig)

# ---------- SECTION 2: Topik dari User (Chat-1M / fallback Arena) ----------
st.header("2) Topik Utama dari Pengguna")
base_topic = df_chat_f if not df_chat_f.empty else df_arena_f
if base_topic.empty:
    st.info("Data untuk analisis topik belum tersedia.")
else:
    df_top = analyze_user_topics(base_topic, top_k=20)
    if df_top.empty:
        st.info("Tidak ada kata kunci yang cukup untuk ditampilkan.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_top, x="Frekuensi", y="Kata Kunci", ax=ax, palette="inferno")
        ax.set_title("Top 20 Kata Kunci dari Pesan Pengguna")
        st.pyplot(fig)

# ---------- SECTION 3: Win‑Rate vs Rata-rata Turns (Arena) ----------
st.header("3) Win‑Rate vs Rata‑rata Turns (Arena)")
if df_arena_f.empty or win_rate.empty:
    st.info("Bagian ini membutuhkan data Arena. Coba perbesar sample rows Arena di sidebar.")
else:
    # Rata-rata turns per model (Arena)
    avg_turns = df_arena_f.groupby("model", observed=True)["turn"].mean()
    comp_idx = win_rate.index.intersection(avg_turns.index)
    comp = pd.DataFrame({
        "Win-Rate": win_rate.loc[comp_idx],
        "Avg Turns": avg_turns.loc[comp_idx],
    }).dropna()

    if comp.empty:
        st.info("Tidak ada irisan model antara Win-Rate dan Avg Turns.")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=comp, x="Avg Turns", y="Win-Rate", s=80, ax=ax)
        for model_name, row in comp.iterrows():
            ax.text(row["Avg Turns"], row["Win-Rate"], model_name, fontsize=8)
        ax.set_title("Win‑Rate vs Avg Turns (Arena)")
        st.pyplot(fig)

# ---------- SECTION 4: TTS (Turns‑to‑Solve) ----------
st.header("4) Turns‑to‑Solve (TTS)")
if src_df.empty:
    st.info(f"Data sumber '{src_choice}' kosong.")
else:
    df_solved = src_df[src_df["is_solved"]]
    if df_solved.empty:
        st.info("Belum ada percakapan yang terdeteksi 'beres' (proxy bahasa).")
        tts_stats = pd.DataFrame(columns=["n_solved", "mean", "median", "p75"])
    else:
        tts_stats = (
            df_solved.groupby("model", observed=True)["turn"]
            .agg(n_solved="count", mean="mean", median="median", p75=lambda s: s.quantile(0.75))
            .sort_values("median")
        )

    st.subheader("Ringkasan TTS per Model")
    st.dataframe(tts_stats.round(2))

    if not tts_stats.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = tts_stats.index.tolist()
        sns.barplot(x=tts_stats.index, y=tts_stats["median"], order=order, ax=ax, palette="viridis")
        ax.set_xlabel(""); ax.set_ylabel("Median TTS (turn)")
        ax.set_title(f"Median TTS per Model (sumber: {src_choice})")
        plt.xticks(rotation=20, ha="right")
        for i, model_name in enumerate(order):
            n = int(tts_stats.loc[model_name, "n_solved"])
            ax.text(i, float(tts_stats.loc[model_name, "median"]) + 0.1, f"n={n}", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig)

# ---------- SECTION 5: Fit‑for‑Purpose (Model × Topik) ----------
st.header("5) Fit‑for‑Purpose: Model × Topik")
if src_df.empty:
    st.info(f"Data sumber '{src_choice}' kosong.")
else:
    top_n_heat = st.sidebar.number_input("Top‑N model untuk heatmap", 4, 20, 8, 1)
    top_models_hm = src_df["model"].value_counts().head(int(top_n_heat)).index
    perf = (
        src_df[src_df["model"].isin(top_models_hm)]
        .groupby(["topic_category", "model"], observed=True)
        .agg(n=("model", "size"), solved_rate=("is_solved", "mean"))
        .reset_index()
    )
    heat = perf.pivot(index="topic_category", columns="model", values="solved_rate")

    if heat.empty:
        st.info("Data tidak mencukupi untuk membuat heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(min(12, 2 + 1.2 * len(heat.columns)), 6))
        sns.heatmap(heat.fillna(0), cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".0%")
        ax.set_xlabel("Model"); ax.set_ylabel("Kategori Topik")
        ax.set_title(f"Solved Rate (Proxy) — Model × Topik (sumber: {src_choice})")
        st.pyplot(fig)

        min_n = st.sidebar.number_input("Ambang N juara per topik", 10, 200, 30, 5)
        leaders = (
            perf[perf["n"] >= int(min_n)]
            .sort_values(["topic_category", "solved_rate"], ascending=[True, False])
            .groupby("topic_category", observed=True)
            .head(1)
            .reset_index(drop=True)
        )
        st.caption(f"Ambang keandalan: hanya sel dengan N ≥ {int(min_n)} dipertimbangkan.")
        if leaders.empty:
            st.info("Belum ada topik yang memenuhi ambang N.")
        else:
            st.subheader("🏆 Juara per Topik")
            st.dataframe(
                leaders.assign(solved_rate=(leaders["solved_rate"] * 100).round(1)).rename(
                    columns={"topic_category": "Topik", "model": "Model", "n": "N", "solved_rate": "Solved Rate (%)"}
                )[["Topik", "Model", "N", "Solved Rate (%)"]]
            )

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Analisis Data Penggunaan LLM — Streamlit Dashboard")
