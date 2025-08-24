# dashboard.py (versi auto-adapt skema)
import os
import re
import warnings
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from datasets import load_dataset

# -------------------- CONFIG & GLOBALS --------------------
st.set_page_config(page_title="Dashboard Analisis Data LLM", page_icon="🤖", layout="wide")
sns.set(style="darkgrid")
warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated and will be changed to True in a future version of pandas",
    category=FutureWarning,
)

OK_PAT = re.compile(
    r"(thanks|thank you|terima kasih|berhasil|works|solved|mantap|fixed?|oke+|ok|done|clear|yes|sip|resolved|great|perfect)",
    re.IGNORECASE,
)

TOPIC_RULES = {
    "Coding": re.compile(r"\b(code|coding|bug|function|class|method|api|regex|python|javascript|java|ts|typescript|cpp|golang|php|html|css|framework|compile|error)\b", re.I),
    "Analisis Data": re.compile(r"\b(data|dataset|pandas|numpy|stat(istik|s)?|regression|cluster|model(ing)?|visualisasi|plot|chart|csv|etl|eda)\b", re.I),
    "Terjemahan": re.compile(r"\b(translate|translat(e|ion)|terjemah|alih ?bahasa|english to indonesian|indonesian to english|b\.?inggris|b\.?indonesia)\b", re.I),
    "Penulisan": re.compile(r"\b(tulis|menulis|writing|essay|artikel|copy|caption|paragraf|ringkas|rangkuman|summary|email|surat|konten)\b", re.I),
}

STOP = {
    "the","and","for","with","that","this","from","your","have","you","will","just","does","did","can","could",
    "would","there","here","into","them","then","than","what","when","where","which","some","about","like",
    "been","were","they","their","ours","ourselves",
    "kami","kita","kamu","anda","yang","dengan","untuk","atau","dari","pada","dalam","akan","saya","dia",
    "itu","ini","bisa","tidak","iya","dan","atau","jadi","agar","karena","kalau","sehingga"
}

# -------------------- HELPERS --------------------
def normalize_model_name(m: str) -> str:
    if m is None:
        return ""
    m = str(m).strip()
    m = m.replace(" - ", "-")
    m = re.sub(r"\s+", " ", m)
    return m

def _user_text_from_conv(conv):
    if not isinstance(conv, (list, tuple)):
        return ""
    parts = []
    for msg in conv:
        if isinstance(msg, dict) and msg.get("role") == "user":
            parts.append((msg.get("content") or "").strip())
    return " ".join(parts)

def is_solved_from_conv(conv) -> bool:
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

def wilson_ci(k: float, n: float, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return lo, hi

# -------------------- DATA LOADER (robust + cache lokal) --------------------
@st.cache_data(show_spinner=True, ttl=3600)
def load_arena55k(sample_rows: int = 20000, local_path: str = "data/arena55k_sample.parquet"):
    """
    Muat dataset lmsys/lmsys-arena-human-preference-55k dengan dukungan 2 skema:
    A) 'conversation_a/b' + 'winner_model' (klasik)
    B) 'prompt' + 'response_a/b' + 'winner_model_a/b' + 'winner_tie' (pairwise)
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df_raw = None

    # 1) coba cache lokal
    if os.path.exists(local_path):
        try:
            df_raw = pd.read_parquet(local_path)
            st.sidebar.success(f"Memakai cache lokal: {local_path}")
        except Exception as e:
            st.sidebar.warning(f"Gagal baca cache lokal: {e}")

    # 2) unduh jika perlu
    if df_raw is None:
        try:
            ds = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
            df_raw = pd.DataFrame(ds)
            # simpan subset ke cache lokal
            df_save = df_raw.sample(sample_rows, random_state=42).reset_index(drop=True) if sample_rows and sample_rows < len(df_raw) else df_raw
            df_save.to_parquet(local_path, index=False)
            st.sidebar.success("Dataset berhasil diunduh & disimpan sebagai cache lokal.")
        except Exception as e:
            st.error("Gagal memuat dataset dari internet dan tidak ada cache lokal.")
            st.exception(e)
            return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), {"schema": "none"}

    # 3) subsample
    if sample_rows and sample_rows < len(df_raw):
        df_raw = df_raw.sample(sample_rows, random_state=42).reset_index(drop=True)

    # 4) deteksi skema
    has_conv = {"model_a","model_b","conversation_a","conversation_b"}.issubset(df_raw.columns)
    has_pair = {"model_a","model_b","prompt","response_a","response_b","winner_model_a","winner_model_b","winner_tie"}.issubset(df_raw.columns)

    meta = {}
    if has_conv:
        meta["schema"] = "conversation"
        # long conversations
        df_a = df_raw[["model_a", "conversation_a"]].rename(columns={"model_a":"model","conversation_a":"conversation"})
        df_b = df_raw[["model_b", "conversation_b"]].rename(columns={"model_b":"model","conversation_b":"conversation"})
        df_long = pd.concat([df_a, df_b], ignore_index=True)
        df_long.dropna(subset=["model","conversation"], inplace=True)

        # win-rate
        if "winner_model" in df_raw.columns:
            wins = df_raw["winner_model"].value_counts()
        elif "winner" in df_raw.columns:
            wins_a = df_raw.loc[df_raw["winner"]=="model_a","model_a"].value_counts()
            wins_b = df_raw.loc[df_raw["winner"]=="model_b","model_b"].value_counts()
            wins = wins_a.add(wins_b, fill_value=0)
        else:
            wins = pd.Series(dtype=float)

        apps = df_raw["model_a"].value_counts().add(df_raw["model_b"].value_counts(), fill_value=0)

        # turunan
        df_long["model_norm"] = df_long["model"].apply(normalize_model_name)
        df_long["user_text"] = df_long["conversation"].apply(_user_text_from_conv)
        df_long["is_solved"] = df_long["conversation"].apply(is_solved_from_conv)
        df_long["topic_category"] = df_long["user_text"].apply(topic_category_from_text)
        df_long["turn"] = df_long["conversation"].apply(lambda conv: len(conv) if isinstance(conv,(list,tuple)) else np.nan)

    elif has_pair:
        meta["schema"] = "pairwise"
        # bentuk percakapan sintetis: [user: prompt, assistant: response_x]
        df_a = df_raw[["model_a","prompt","response_a","winner_model_a","winner_tie"]].copy()
        df_b = df_raw[["model_b","prompt","response_b","winner_model_b","winner_tie"]].copy()
        df_a.rename(columns={"model_a":"model","response_a":"response","winner_model_a":"won"}, inplace=True)
        df_b.rename(columns={"model_b":"model","response_b":"response","winner_model_b":"won"}, inplace=True)

        df_a["conversation"] = df_a.apply(lambda r: [{"role":"user","content":r["prompt"]},{"role":"assistant","content":r["response"]}], axis=1)
        df_b["conversation"] = df_b.apply(lambda r: [{"role":"user","content":r["prompt"]},{"role":"assistant","content":r["response"]}], axis=1)

        # jika tie, set won=False untuk keduanya
        df_a.loc[df_a["winner_tie"]==1, "won"] = 0
        df_b.loc[df_b["winner_tie"]==1, "won"] = 0

        df_long = pd.concat([df_a[["model","conversation","won"]], df_b[["model","conversation","won"]]], ignore_index=True)
        df_long.dropna(subset=["model","conversation"], inplace=True)

        # win-rate & appearances
        wins = df_long.groupby("model")["won"].sum(min_count=1)
        apps = df_long["model"].value_counts()

        # turunan
        df_long["model_norm"] = df_long["model"].apply(normalize_model_name)
        df_long["user_text"] = df_long["conversation"].apply(_user_text_from_conv)
        # gunakan 'won' sebagai proxy solved (tidak ideal, tapi konsisten untuk heatmap)
        df_long["is_solved"] = df_long["won"].fillna(0).astype(int) == 1
        df_long["topic_category"] = df_long["user_text"].apply(topic_category_from_text)
        df_long["turn"] = 2  # prompt + satu balasan (pairwise)

        meta["tts_note"] = "Skema pairwise hanya 1 balasan per model → TTS ≈ 2 (kurang informatif)."

    else:
        st.error("Skema dataset tidak dikenali. Kolom kunci tidak ditemukan.")
        st.caption(f"Kolom tersedia: {sorted(df_raw.columns.tolist())[:40]} …")
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), {"schema":"unknown"}

    # normalisasi index untuk win-rate
    wins.index = wins.index.astype(str).map(normalize_model_name)
    apps.index = apps.index.astype(str).map(normalize_model_name)
    win_rate = (wins / apps).dropna().sort_values(ascending=False)

    # tabel win-rate detail + Wilson CI
    wr_df = pd.DataFrame({"wins": wins, "apps": apps}).fillna(0)
    wr_df["win_rate"] = wr_df.apply(lambda r: (r["wins"]/r["apps"]) if r["apps"]>0 else np.nan, axis=1)
    wr_df[["wr_lo","wr_hi"]] = wr_df.apply(lambda r: pd.Series(wilson_ci(r["wins"], r["apps"])), axis=1)
    wr_df.index.name = "model_norm"

    return df_long, win_rate, wr_df, meta

# -------------------- COMPUTATION (cached) --------------------
@st.cache_data(show_spinner=True, ttl=1200)
def compute_topics_ngram(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Kata Kunci", "Frekuensi"])
    texts = df["user_text"].astype(str).tolist()
    all_text = " ".join(texts).lower()
    tokens = re.findall(r"[a-zA-Z]{3,}", all_text)
    tokens = [w for w in tokens if w not in STOP]
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    merged = tokens + bigrams
    vc = pd.Series(merged).value_counts().head(top_k)
    return pd.DataFrame({"Kata Kunci": vc.index, "Frekuensi": vc.values})

@st.cache_data(show_spinner=True, ttl=1200)
def compute_tts(df_in: pd.DataFrame, min_turn: int = 3) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame(columns=["n_solved","mean","median","p75"])
    df_use = df_in[(df_in["is_solved"]) & (df_in["turn"].fillna(0) >= min_turn)]
    if df_use.empty:
        return pd.DataFrame(columns=["n_solved","mean","median","p75"])
    tts = (
        df_use.groupby("model_norm", observed=True)["turn"]
        .agg(n_solved="count", mean="mean", median="median", p75=lambda s: s.quantile(0.75))
        .sort_values("median")
    )
    return tts

@st.cache_data(show_spinner=True, ttl=1200)
def compute_fit_for_purpose(df_in: pd.DataFrame, top_n_models: int = 8):
    if df_in.empty:
        return pd.DataFrame(), pd.DataFrame()
    top_models = df_in["model_norm"].value_counts().head(int(top_n_models)).index
    perf = (
        df_in[df_in["model_norm"].isin(top_models)]
        .groupby(["topic_category","model_norm"], observed=True)
        .agg(n=("model_norm","size"), solved_rate=("is_solved","mean"))
        .reset_index()
    )
    heat = perf.pivot(index="topic_category", columns="model_norm", values="solved_rate")
    return perf, heat

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Pengaturan")
sample_rows = st.sidebar.slider("Sample rows", 5_000, 50_000, 20_000, step=5_000)
min_turn = st.sidebar.number_input("Minimal turn untuk TTS", 1, 10, 3, 1)
top_n_pop = st.sidebar.slider("Top‑N Model (popularitas)", 5, 30, 12, 1)
top_n_heat = st.sidebar.slider("Top‑N Model (heatmap)", 4, 20, 8, 1)
min_n_leader = st.sidebar.slider("Ambang N juara per topik", 10, 200, 30, 5)

# -------------------- LOAD DATA --------------------
with st.spinner("Memuat & menyiapkan data..."):
    try:
        df_long, win_rate_series, wr_df, meta = load_arena55k(sample_rows)
    except Exception as e:
        st.error("Gagal memuat dataset (error tak terduga).")
        st.exception(e)
        df_long, win_rate_series, wr_df, meta = pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), {"schema":"none"}

schema = meta.get("schema","none")

# Filter global model
all_models = sorted(df_long["model_norm"].dropna().unique().tolist()) if not df_long.empty else []
selected_models = st.sidebar.multiselect(
    "Filter Model (berlaku global)",
    options=all_models,
    default=all_models[:min(len(all_models), top_n_pop)]
)
def apply_model_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not selected_models:
        return df
    return df[df["model_norm"].isin(selected_models)]

df_f = apply_model_filter(df_long)

# -------------------- SECTION 1: Popularitas --------------------
st.header("1) Popularitas Model")
if df_f.empty:
    st.info("Data kosong. Perbesar sample rows / periksa koneksi.")
else:
    order = df_f["model_norm"].value_counts().head(top_n_pop).index
    df_pop = df_f[df_f["model_norm"].isin(order)]
    fig, ax = plt.subplots(figsize=(10,5))
    sns.countplot(data=df_pop, x="model_norm", order=order, ax=ax, palette="cividis")
    ax.set_title("Popularitas Model (berdasar jumlah percakapan)")
    ax.set_xlabel("Model"); ax.set_ylabel("Jumlah Percakapan")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig)
    st.caption(f"N efektif: {len(df_pop):,} percakapan | Model unik: {len(order)}")

# -------------------- SECTION 2: Topik n‑gram --------------------
st.header("2) Topik Utama Pengguna (Unigram + Bigram)")
if df_f.empty:
    st.info("Data kosong. Tidak dapat menganalisis topik.")
else:
    df_topics = compute_topics_ngram(df_f, top_k=20)
    if df_topics.empty:
        st.info("Tidak ada kata kunci yang cukup untuk ditampilkan.")
    else:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=df_topics, x="Frekuensi", y="Kata Kunci", ax=ax, palette="cividis")
        ax.set_title("Top 20 N‑gram dari Pesan Pengguna")
        st.pyplot(fig)
        st.caption(f"N efektif: {len(df_f):,} percakapan | Token unik terpilih: {len(df_topics)}")

# -------------------- SECTION 3: Win‑Rate (Wilson CI) --------------------
st.header("3) Win‑Rate per Model (dengan Wilson CI)")
if wr_df.empty:
    st.info("Informasi win‑rate tidak tersedia pada sample saat ini.")
else:
    wr_view = wr_df.copy()
    if selected_models:
        wr_view = wr_view.loc[wr_view.index.intersection(selected_models)]
    wr_view = wr_view.sort_values("win_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(wr_view))
    ax.errorbar(
        x, wr_view["win_rate"].values,
        yerr=[wr_view["win_rate"].values - wr_view["wr_lo"].values,
              wr_view["wr_hi"].values - wr_view["win_rate"].values],
        fmt="o", capsize=3
    )
    ax.set_xticks(x)
    ax.set_xticklabels(wr_view.index, rotation=25, ha="right")
    ax.set_ylabel("Win‑Rate")
    ax.set_title("Win‑Rate per Model (error bars = Wilson 95% CI)")
    st.pyplot(fig)
    st.caption(f"N pasangan kompetisi (apps) total: {int(wr_view['apps'].sum()):,}")

# -------------------- SECTION 4: TTS --------------------
st.header("4) Turns‑to‑Solve (TTS)")
if schema == "pairwise":
    st.info("Catatan: skema pairwise hanya 1 balasan per model → TTS ≈ 2 turn (kurang informatif).")
if df_f.empty:
    st.info("Data kosong. Tidak bisa menghitung TTS.")
else:
    tts_stats = compute_tts(df_f, min_turn=min_turn)
    st.subheader("Ringkasan TTS per Model")
    st.dataframe(tts_stats.round(2))
    st.download_button("⬇️ Unduh TTS (CSV)", tts_stats.to_csv().encode(), "tts_stats.csv", "text/csv")

    if not tts_stats.empty:
        fig, ax = plt.subplots(figsize=(10,5))
        order = tts_stats.index.tolist()
        sns.barplot(x=tts_stats.index, y=tts_stats["median"], order=order, ax=ax, palette="cividis")
        ax.set_xlabel(""); ax.set_ylabel("Median TTS (turn)")
        ax.set_title("Median TTS per Model (lebih kecil lebih baik)")
        plt.xticks(rotation=20, ha="right")
        for i, model_name in enumerate(order):
            n = int(tts_stats.loc[model_name, "n_solved"])
            ax.text(i, float(tts_stats.loc[model_name, "median"]) + 0.1, f"n={n}",
                    ha="center", va="bottom", fontsize=9)
        st.pyplot(fig)
        st.caption(f"N efektif (percakapan 'beres' & turn ≥ {min_turn}): {int(tts_stats['n_solved'].sum()):,}")

# -------------------- SECTION 5: Fit‑for‑Purpose --------------------
st.header("5) Fit‑for‑Purpose: Model × Topik")
if df_f.empty:
    st.info("Data kosong. Tidak bisa membuat heatmap.")
else:
    perf, heat = compute_fit_for_purpose(df_f, top_n_models=top_n_heat)
    if heat.empty:
        st.info("Data tidak mencukupi untuk heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(min(12, 2 + 1.2*len(heat.columns)), 6))
        sns.heatmap(heat.fillna(0), cmap="cividis", vmin=0, vmax=1, annot=True, fmt=".0%")
        ax.set_xlabel("Model"); ax.set_ylabel("Kategori Topik")
        ax.set_title("Solved Rate (Proxy) — Model × Topik")
        st.pyplot(fig)

        leaders = (
            perf[perf["n"] >= int(min_n_leader)]
            .sort_values(["topic_category","solved_rate"], ascending=[True, False])
            .groupby("topic_category", observed=True).head(1).reset_index(drop=True)
        )
        st.caption(f"Ambang keandalan: hanya sel dengan N ≥ {int(min_n_leader)} dipertimbangkan.")
        if leaders.empty:
            st.info("Belum ada topik yang memenuhi ambang N.")
        else:
            leaders_show = leaders.assign(solved_rate=(leaders["solved_rate"]*100).round(1)).rename(
                columns={"topic_category":"Topik","model_norm":"Model","n":"N","solved_rate":"Solved Rate (%)"}
            )[["Topik","Model","N","Solved Rate (%)"]]
            st.subheader("🏆 Juara per Topik")
            st.dataframe(leaders_show)
            st.download_button("⬇️ Unduh Perf Heatmap (CSV)", perf.to_csv(index=False).encode(),
                               "fit_for_purpose.csv", "text/csv")
            st.download_button("⬇️ Unduh Juara per Topik (CSV)", leaders_show.to_csv(index=False).encode(),
                               "leaders_per_topic.csv", "text/csv")

# -------------------- FOOTER --------------------
st.markdown("---")
schema_note = "Skema: Pairwise (prompt+1 balasan/model)" if schema=="pairwise" else ("Skema: Conversation multi-turn" if schema=="conversation" else "Skema: Unknown")
st.caption(f"Analisis Data Penggunaan LLM — Streamlit Dashboard · {schema_note} · "
           "Metric proxy: 'Solved' berbasis sinyal bahasa/pemenang; gunakan ambang N untuk reliabilitas.")
