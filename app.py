import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0c10;
    --card: #13161e;
    --border: #1f2333;
    --accent: #7c6af7;
    --accent2: #f7c26a;
    --accent3: #6af7c2;
    --accent4: #f76a6a;
    --text: #e2e4f0;
    --muted: #5a5f7a;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.main { background-color: var(--bg); }
.block-container { padding: 2rem 2.5rem; }
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; }

.hero {
    background: linear-gradient(135deg, #0a0c10 0%, #13101e 60%, #0a100c 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.hero h1 { color: var(--accent); font-size: 1.8rem; margin: 0 0 0.4rem; }
.hero p  { color: var(--muted); margin: 0; font-size: 0.95rem; }

.seg-card {
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    border: 1px solid var(--border);
    margin-bottom: 0.8rem;
}
.seg-val  { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; }
.seg-lbl  { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }
.seg-sub  { font-size: 0.75rem; margin-top: 0.5rem; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

.insight-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}
.insight-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; margin-bottom: 0.4rem; }

section[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }

.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
label { color: var(--text) !important; font-size: 0.88rem !important; }
.stSlider > div > div > div { background: var(--accent) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Segment colors & labels ───────────────────────────────────────────────────
SEG_COLORS = ["#7c6af7", "#f7c26a", "#6af7c2", "#f76a6a", "#6ab4f7"]
SEG_NAMES  = {
    0: ("👑 Champions",        "High value, frequent, recent buyers",        "#7c6af7"),
    1: ("💛 Loyal Customers",  "Regular buyers with good frequency",         "#f7c26a"),
    2: ("🌱 New Customers",    "Recent first-time or low-frequency buyers",  "#6af7c2"),
    3: ("⚠️ At Risk",          "Previously active, now going quiet",         "#f76a6a"),
    4: ("😴 Lost Customers",   "Low recency, frequency & monetary value",    "#6ab4f7"),
}
SEG_ACTIONS = {
    0: ["🎁 Reward with loyalty program", "🤝 Ask for referrals", "🔒 Offer early access to new products"],
    1: ["📧 Send personalized offers",    "⭐ Upsell premium products",        "🎉 Appreciate with thank-you notes"],
    2: ["👋 Onboard with welcome offers", "📚 Educate about product range",    "🎯 Re-engage with targeted campaigns"],
    3: ["🚨 Win-back campaigns urgently", "💸 Offer special discounts",        "📞 Personal outreach from sales team"],
    4: ["💌 Last-chance reactivation",    "🔍 Survey to understand why lost",  "🗑️ Consider removing from main list"],
}

# ── Generate sample data ──────────────────────────────────────────────────────
@st.cache_data
def generate_sample_data(n=200):
    np.random.seed(42)
    data = []
    profiles = [
        dict(r_mean=5,  r_std=3,  f_mean=20, f_std=5,  m_mean=1500, m_std=300,  n=40),
        dict(r_mean=20, r_std=8,  f_mean=12, f_std=4,  m_mean=800,  m_std=200,  n=50),
        dict(r_mean=10, r_std=5,  f_mean=5,  f_std=2,  m_mean=300,  m_std=100,  n=40),
        dict(r_mean=60, r_std=15, f_mean=8,  f_std=3,  m_mean=600,  m_std=150,  n=40),
        dict(r_mean=90, r_std=20, f_mean=2,  f_std=1,  m_mean=150,  m_std=80,   n=30),
    ]
    cid = 1001
    for p in profiles:
        for _ in range(p["n"]):
            data.append({
                "CustomerID":      cid,
                "Recency":         max(1, int(np.random.normal(p["r_mean"], p["r_std"]))),
                "Frequency":       max(1, int(np.random.normal(p["f_mean"], p["f_std"]))),
                "MonetaryValue":   max(10, round(np.random.normal(p["m_mean"], p["m_std"]), 2)),
            })
            cid += 1
    return pd.DataFrame(data)

# ── Run KMeans segmentation ───────────────────────────────────────────────────
@st.cache_data
def run_segmentation(df, n_clusters):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df[["Recency", "Frequency", "MonetaryValue"]])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = km.fit_predict(rfm_scaled)

    # Map clusters to meaningful labels by avg monetary desc
    cluster_means = df.groupby("Cluster")["MonetaryValue"].mean().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(cluster_means.index)}
    df["Segment"] = df["Cluster"].map(mapping)
    return df

# ── Matplotlib dark style ─────────────────────────────────────────────────────
def dark_fig(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#13161e")
    ax.set_facecolor("#13161e")
    for spine in ax.spines.values(): spine.set_edgecolor("#1f2333")
    ax.tick_params(colors="#5a5f7a", labelsize=8)
    ax.xaxis.label.set_color("#5a5f7a")
    ax.yaxis.label.set_color("#5a5f7a")
    ax.title.set_color("#e2e4f0")
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio("Choose data", ["📊 Use Sample Data", "📁 Upload CSV"], label_visibility="collapsed")

    df_raw = None
    if data_source == "📁 Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df_raw)} rows")
            st.write("**Columns:**", list(df_raw.columns))
            st.markdown('<div class="section-label" style="margin-top:1rem">Map Columns</div>', unsafe_allow_html=True)
            cols = list(df_raw.columns)
            r_col = st.selectbox("Recency column",   cols)
            f_col = st.selectbox("Frequency column", cols)
            m_col = st.selectbox("Monetary column",  cols)
            if st.button("Apply Column Mapping"):
                df_raw = df_raw.rename(columns={r_col: "Recency", f_col: "Frequency", m_col: "MonetaryValue"})
                df_raw["CustomerID"] = range(1001, 1001 + len(df_raw))
                st.success("Columns mapped! ✅")
    else:
        df_raw = generate_sample_data()
        st.info("Using 200 synthetic customers with realistic RFM patterns.")

    st.markdown('<div class="section-label" style="margin-top:1.5rem">Segmentation Settings</div>',
                unsafe_allow_html=True)
    n_clusters = st.slider("Number of Segments", 2, 5, 5)
    run_btn    = st.button("🎯 Run Segmentation")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🎯 Customer Segmentation</h1>
  <p>RFM Analysis + K-Means Clustering · Built by Taskeen Hussain</p>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.info("👈 Choose a data source in the sidebar and click **Run Segmentation**.")
    st.stop()

# ── Always show raw data preview ─────────────────────────────────────────────
with st.expander("📋 Preview Raw Data", expanded=False):
    st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

# ── Run on button click (or auto on first load with sample data) ──────────────
if "seg_df" not in st.session_state or run_btn:
    needed = {"Recency", "Frequency", "MonetaryValue"}
    if not needed.issubset(df_raw.columns):
        st.error("❌ Your data must have Recency, Frequency, MonetaryValue columns. Use the column mapper in the sidebar.")
        st.stop()
    with st.spinner("Running K-Means clustering..."):
        st.session_state["seg_df"] = run_segmentation(df_raw, n_clusters)

seg_df = st.session_state["seg_df"]

# ── Top KPI row ───────────────────────────────────────────────────────────────
total   = len(seg_df)
n_segs  = seg_df["Segment"].nunique()
avg_val = seg_df["MonetaryValue"].mean()
top_pct = round(len(seg_df[seg_df["Segment"] == 0]) / total * 100, 1)

k1, k2, k3, k4 = st.columns(4)
for col, val, lbl, color in [
    (k1, total,          "Total Customers",    "#7c6af7"),
    (k2, n_segs,         "Segments Found",     "#f7c26a"),
    (k3, f"${avg_val:.0f}", "Avg Monetary Value","#6af7c2"),
    (k4, f"{top_pct}%",  "Champion Customers", "#f76a6a"),
]:
    col.markdown(
        f'<div class="seg-card" style="border-top:3px solid {color}">'
        f'<div class="seg-val" style="color:{color}">{val}</div>'
        f'<div class="seg-lbl">{lbl}</div></div>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Segment summary + charts ──────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Segment Breakdown</div>', unsafe_allow_html=True)
    summary = (seg_df.groupby("Segment")
               .agg(Count=("CustomerID","count"),
                    Avg_Recency=("Recency","mean"),
                    Avg_Frequency=("Frequency","mean"),
                    Avg_Monetary=("MonetaryValue","mean"))
               .round(1).reset_index())

    for _, row in summary.iterrows():
        seg_id = int(row["Segment"])
        if seg_id >= len(SEG_NAMES): continue
        name, desc, color = SEG_NAMES[seg_id]
        pct = round(row["Count"] / total * 100, 1)
        st.markdown(f"""
        <div class="insight-box" style="border-left:3px solid {color}">
          <div class="insight-title" style="color:{color}">{name}
            <span style="float:right;color:#5a5f7a;font-size:0.75rem">{int(row['Count'])} customers ({pct}%)</span>
          </div>
          <div style="font-size:0.78rem;color:#5a5f7a;margin-bottom:0.5rem">{desc}</div>
          <div style="font-size:0.78rem;display:flex;gap:1.5rem">
            <span>📅 Recency: <b style="color:{color}">{row['Avg_Recency']}d</b></span>
            <span>🔁 Freq: <b style="color:{color}">{row['Avg_Frequency']}</b></span>
            <span>💰 Value: <b style="color:{color}">${row['Avg_Monetary']}</b></span>
          </div>
        </div>""", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">Segment Distribution</div>', unsafe_allow_html=True)
    counts = summary["Count"].values
    labels = [SEG_NAMES[int(i)][0] if int(i) < len(SEG_NAMES) else f"Seg {i}"
              for i in summary["Segment"].values]
    colors = [SEG_NAMES[int(i)][2] if int(i) < len(SEG_NAMES) else SEG_COLORS[i % len(SEG_COLORS)]
              for i in summary["Segment"].values]

    fig, ax = dark_fig((5, 4))
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, colors=colors,
        autopct='%1.1f%%', startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor="#0a0c10", linewidth=2))
    for at in autotexts:
        at.set_color("#0a0c10"); at.set_fontsize(8); at.set_fontweight("bold")
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=7, framealpha=0,
              labelcolor="#e2e4f0")
    ax.set_title("Customer Segments", fontsize=10, pad=10)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # RFM scatter
    st.markdown('<div class="section-label" style="margin-top:1rem">Frequency vs Monetary</div>',
                unsafe_allow_html=True)
    fig2, ax2 = dark_fig((5, 3.2))
    for seg_id in seg_df["Segment"].unique():
        if seg_id >= len(SEG_NAMES): continue
        sub = seg_df[seg_df["Segment"] == seg_id]
        _, _, color = SEG_NAMES[seg_id]
        ax2.scatter(sub["Frequency"], sub["MonetaryValue"],
                    c=color, alpha=0.6, s=25, label=SEG_NAMES[seg_id][0])
    ax2.set_xlabel("Frequency", fontsize=8)
    ax2.set_ylabel("Monetary Value ($)", fontsize=8)
    ax2.legend(fontsize=6, framealpha=0, labelcolor="#e2e4f0")
    ax2.grid(True, alpha=0.08)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ── RFM Bar charts ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">RFM Profile by Segment</div>', unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)
seg_labels_short = [SEG_NAMES[int(i)][0].split(" ", 1)[1] if int(i) < len(SEG_NAMES) else f"Seg {i}"
                    for i in summary["Segment"].values]
bar_colors = [SEG_NAMES[int(i)][2] if int(i) < len(SEG_NAMES) else "#7c6af7"
              for i in summary["Segment"].values]

for col, metric, title in [(b1, "Avg_Recency", "Avg Recency (days) ↓ better"),
                            (b2, "Avg_Frequency","Avg Frequency ↑ better"),
                            (b3, "Avg_Monetary", "Avg Monetary Value ($)")]:
    fig, ax = dark_fig((4, 3))
    bars = ax.bar(seg_labels_short, summary[metric], color=bar_colors, edgecolor="#0a0c10", linewidth=0.5)
    for bar, val in zip(bars, summary[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=7, color="#e2e4f0")
    ax.set_title(title, fontsize=8, pad=8)
    ax.set_xticklabels(seg_labels_short, fontsize=6, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.08)
    col.pyplot(fig, use_container_width=True)
    plt.close()

# ── Marketing Actions ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">📣 Recommended Marketing Actions</div>', unsafe_allow_html=True)

act_cols = st.columns(min(n_clusters, 5))
for i, col in enumerate(act_cols):
    if i >= len(SEG_NAMES): break
    name, _, color = SEG_NAMES[i]
    actions = SEG_ACTIONS[i]
    col.markdown(f"""
    <div class="insight-box" style="border-top:3px solid {color};height:100%">
      <div class="insight-title" style="color:{color};font-size:0.8rem">{name}</div>
      {"".join(f'<div style="font-size:0.75rem;color:#c0c4d8;margin:0.3rem 0">{a}</div>' for a in actions)}
    </div>""", unsafe_allow_html=True)

# ── Download segmented data ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">⬇️ Download Results</div>', unsafe_allow_html=True)

export_df = seg_df.copy()
export_df["SegmentName"] = export_df["Segment"].apply(
    lambda x: SEG_NAMES[x][0] if x < len(SEG_NAMES) else f"Segment {x}")
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Segmented Customer Data (CSV)",
                   data=csv, file_name="customer_segments.csv", mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#5a5f7a;font-size:0.78rem;font-family:'IBM Plex Mono',monospace">
  Built by <span style="color:#7c6af7">Taskeen Hussain</span> · Data Analyst & Python Developer ·
  <a href="https://github.com/TaskeenHussain" style="color:#7c6af7">GitHub</a> ·
  <a href="https://kaggle.com/taskeenhkbbeechtree" style="color:#7c6af7">Kaggle</a> ·
  <a href="https://taskeenhussain.github.io/taskeen-portfolio" style="color:#7c6af7">Portfolio</a>
</div>
""", unsafe_allow_html=True)
