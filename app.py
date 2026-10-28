from __future__ import annotations

import json
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from plotly import graph_objects as go
from plotly import io as pio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


APP_TITLE = "SmartCart Customer Insights"
APP_TAGLINE = "A clean, resume-ready customer segmentation dashboard built from the SmartCart notebook."
DATA_FILE = Path(__file__).with_name("smartcart_customers.csv")


app = Flask(__name__)


def _demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 120
    base_date = pd.Timestamp("2024-12-31")

    df = pd.DataFrame(
        {
            "ID": np.arange(1000, 1000 + size),
            "Year_Birth": rng.integers(1948, 2001, size=size),
            "Education": rng.choice(["Graduation", "Master", "PhD", "2n Cycle", "Basic"], size=size),
            "Marital_Status": rng.choice(["Married", "Together", "Single", "Divorced", "Widow"], size=size),
            "Income": np.clip(rng.normal(62000, 22000, size=size), 12000, 180000),
            "Kidhome": rng.integers(0, 3, size=size),
            "Teenhome": rng.integers(0, 3, size=size),
            "Dt_Customer": pd.to_datetime(base_date - pd.to_timedelta(rng.integers(30, 2500, size=size), unit="D")),
            "Recency": rng.integers(0, 100, size=size),
            "Response": rng.integers(0, 2, size=size),
            "MntWines": rng.integers(0, 1500, size=size),
            "MntFruits": rng.integers(0, 250, size=size),
            "MntMeatProducts": rng.integers(0, 1200, size=size),
            "MntFishProducts": rng.integers(0, 350, size=size),
            "MntSweetProducts": rng.integers(0, 280, size=size),
            "MntGoldProds": rng.integers(0, 350, size=size),
        }
    )
    return df


def load_data(uploaded_file=None) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read().decode("utf-8", errors="ignore")
            frame = pd.read_csv(StringIO(raw))
            if not frame.empty:
                return frame, "Uploaded CSV"
        except Exception:
            pass

        return _demo_data(), "Built-in demo dataset"

    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE), DATA_FILE.name

    return _demo_data(), "Built-in demo dataset"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_defaults = {
        "Year_Birth": 1970,
        "Education": "Graduation",
        "Marital_Status": "Single",
        "Income": float(df["Income"].median()) if "Income" in df and df["Income"].notna().any() else 0.0,
        "Kidhome": 0,
        "Teenhome": 0,
        "Dt_Customer": pd.Timestamp(datetime.utcnow().date()),
        "Recency": 0,
        "Response": 0,
        "MntWines": 0,
        "MntFruits": 0,
        "MntMeatProducts": 0,
        "MntFishProducts": 0,
        "MntSweetProducts": 0,
        "MntGoldProds": 0,
    }

    for column, default in required_defaults.items():
        if column not in df.columns:
            df[column] = default

    df["Income"] = pd.to_numeric(df["Income"], errors="coerce").fillna(df["Income"].median())
    df["Year_Birth"] = pd.to_numeric(df["Year_Birth"], errors="coerce").fillna(1970).astype(int)

    current_year = datetime.utcnow().year
    df["Age"] = current_year - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)
    reference_date = df["Dt_Customer"].max()
    if pd.isna(reference_date):
        reference_date = pd.Timestamp.utcnow()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days.fillna(0)

    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    for column in spend_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["Total_Spending"] = df[spend_cols].sum(axis=1)
    df["Total_Children"] = pd.to_numeric(df["Kidhome"], errors="coerce").fillna(0) + pd.to_numeric(df["Teenhome"], errors="coerce").fillna(0)

    df["Education"] = df["Education"].replace(
        {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate",
        }
    )
    df["Living_With"] = df["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Widow": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone",
        }
    )

    df = df[(df["Age"] < 90) & (df["Income"] < 600_000)].copy()

    return df


def cluster_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_frame = df[
        [
            "Income",
            "Recency",
            "Age",
            "Customer_Tenure_Days",
            "Total_Spending",
            "Total_Children",
        ]
    ].copy()

    encoded = pd.get_dummies(
        df[["Education", "Living_With"]],
        columns=["Education", "Living_With"],
        drop_first=False,
    )
    feature_frame = pd.concat([feature_frame, encoded], axis=1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)

    cluster_count = min(4, max(1, len(df)))
    if cluster_count == 1:
        clustered = df.copy()
        clustered["cluster"] = 0
        metadata = {"cluster_count": 1, "silhouette": None, "explained_variance": None}
        return clustered, metadata

    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    pca = PCA(n_components=min(3, scaled.shape[1], len(df)))
    pca_values = pca.fit_transform(scaled)

    clustered = df.copy()
    clustered["cluster"] = labels
    clustered["PCA1"] = pca_values[:, 0]
    clustered["PCA2"] = pca_values[:, 1] if pca_values.shape[1] > 1 else 0.0
    clustered["PCA3"] = pca_values[:, 2] if pca_values.shape[1] > 2 else 0.0

    silhouette = None
    if len(np.unique(labels)) > 1 and len(clustered) > len(np.unique(labels)):
        silhouette = float(silhouette_score(scaled, labels))

    metadata = {
        "cluster_count": cluster_count,
        "silhouette": silhouette,
        "explained_variance": [float(value) for value in pca.explained_variance_ratio_],
    }
    return clustered, metadata


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def make_kpi(title: str, value: str, subtitle: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """


def figure_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build_figures(df: pd.DataFrame, clustered: pd.DataFrame) -> dict[str, str]:
    figures: dict[str, str] = {}

    education_counts = df["Education"].value_counts().reset_index()
    education_counts.columns = ["Education", "Count"]
    fig_education = go.Figure(
        data=[go.Bar(x=education_counts["Education"], y=education_counts["Count"], marker_color="#2b6cb0")]
    )
    fig_education.update_layout(title="Education Mix", template="plotly_white", height=360, margin=dict(l=20, r=20, t=50, b=20))
    figures["education"] = figure_to_div(fig_education)

    cluster_counts = clustered["cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    fig_clusters = go.Figure(
        data=[go.Pie(labels=[f"Cluster {value}" for value in cluster_counts["Cluster"]], values=cluster_counts["Count"], hole=0.5)]
    )
    fig_clusters.update_layout(title="Cluster Distribution", template="plotly_white", height=360, margin=dict(l=20, r=20, t=50, b=20))
    figures["clusters"] = figure_to_div(fig_clusters)

    fig_scatter = go.Figure(
        data=[
            go.Scatter(
                x=clustered["Income"],
                y=clustered["Total_Spending"],
                mode="markers",
                marker=dict(color=clustered["cluster"], colorscale="Viridis", size=9, opacity=0.85),
                text=[f"Cluster {value}" for value in clustered["cluster"]],
            )
        ]
    )
    fig_scatter.update_layout(
        title="Income vs Spending",
        xaxis_title="Income",
        yaxis_title="Total Spending",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    figures["scatter"] = figure_to_div(fig_scatter)

    age_bins = pd.cut(df["Age"], bins=[18, 30, 40, 50, 60, 70, 90], right=False, include_lowest=True)
    age_counts = age_bins.value_counts().sort_index().reset_index()
    age_counts.columns = ["Age Band", "Count"]
    fig_age = go.Figure(data=[go.Bar(x=age_counts["Age Band"].astype(str), y=age_counts["Count"], marker_color="#d97706")])
    fig_age.update_layout(title="Age Distribution", template="plotly_white", height=360, margin=dict(l=20, r=20, t=50, b=20))
    figures["age"] = figure_to_div(fig_age)

    return figures


def summarize_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    summary = clustered.groupby("cluster").agg(
        customers=("cluster", "size"),
        avg_income=("Income", "mean"),
        avg_spending=("Total_Spending", "mean"),
        avg_age=("Age", "mean"),
        avg_children=("Total_Children", "mean"),
    )
    return summary.reset_index().sort_values("cluster")


def render_dashboard(df: pd.DataFrame, clustered: pd.DataFrame, source_name: str, uploaded_file_name: str | None = None) -> str:
    cluster_summary = summarize_clusters(clustered)
    figures = build_figures(df, clustered)
    cluster_preview = clustered[["Income", "Age", "Total_Spending", "cluster"]].head(8).to_html(index=False, classes="table table-sm table-striped align-middle", border=0)

    avg_income = format_currency(df["Income"].mean())
    avg_spending = format_currency(df["Total_Spending"].mean())
    avg_age = f"{df['Age'].mean():.1f}"
    segment_count = clustered["cluster"].nunique()
    silhouette = clustered.attrs.get("silhouette")

    if silhouette is None:
        silhouette_display = "N/A"
    else:
        silhouette_display = f"{silhouette:.2f}"

    kpi_html = "".join(
        [
            make_kpi("Customers", f"{len(df):,}", f"Source: {source_name}"),
            make_kpi("Avg Income", avg_income, "After cleaning outliers"),
            make_kpi("Avg Spending", avg_spending, "Aggregated from product categories"),
            make_kpi("Avg Age", avg_age, f"{segment_count} modeled segments"),
            make_kpi("Silhouette", silhouette_display, "Cluster quality snapshot"),
        ]
    )

    uploaded_notice = f"<span class='upload-pill'>Loaded: {uploaded_file_name}</span>" if uploaded_file_name else ""

    cluster_rows = "".join(
        f"""
        <tr>
            <td>Cluster {int(row.cluster)}</td>
            <td>{int(row.customers)}</td>
            <td>{format_currency(row.avg_income)}</td>
            <td>{format_currency(row.avg_spending)}</td>
            <td>{row.avg_age:.1f}</td>
            <td>{row.avg_children:.1f}</td>
        </tr>
        """
        for row in cluster_summary.itertuples(index=False)
    )

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{APP_TITLE}</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            :root {{
                --bg: #07111f;
                --surface: rgba(12, 20, 35, 0.84);
                --surface-strong: rgba(16, 26, 46, 0.96);
                --text: #ffffff;
                --muted: #d0d8e8;
                --accent: #4dd9b8;
                --accent-2: #64b5ff;
                --accent-3: #ffb86c;
                --border: rgba(255, 255, 255, 0.15);
            }}

            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                color: var(--text);
                font-family: 'DM Sans', sans-serif;
                background:
                    radial-gradient(circle at top left, rgba(102, 227, 196, 0.18), transparent 24%),
                    radial-gradient(circle at top right, rgba(124, 183, 255, 0.18), transparent 26%),
                    linear-gradient(160deg, #050a13 0%, #091423 44%, #101f35 100%);
                min-height: 100vh;
            }}

            .app-shell {{ max-width: 1440px; margin: 0 auto; padding: 28px 18px 42px; }}
            .hero {{
                position: relative;
                overflow: hidden;
                border: 1px solid var(--border);
                background: linear-gradient(135deg, rgba(10, 18, 32, 0.95), rgba(16, 30, 49, 0.88));
                border-radius: 30px;
                padding: 34px;
                box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            }}
            .hero::after {{
                content: '';
                position: absolute;
                inset: auto -40px -40px auto;
                width: 220px;
                height: 220px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(102, 227, 196, 0.22), transparent 70%);
                filter: blur(12px);
            }}
            .eyebrow {{
                display: inline-flex;
                gap: 10px;
                align-items: center;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                color: var(--accent);
                font-size: 0.74rem;
            }}
            .hero h1 {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: clamp(2.2rem, 5vw, 4.6rem);
                line-height: 0.96;
                margin: 14px 0 18px;
                max-width: 12ch;
            }}
            .hero p {{ color: var(--muted); font-size: 1.08rem; max-width: 70ch; }}
            .hero-meta {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 22px; }}
            .chip {{
                border: 1px solid var(--border);
                background: rgba(255, 255, 255, 0.05);
                padding: 10px 14px;
                border-radius: 999px;
                color: var(--text);
                font-size: 0.92rem;
            }}
            .upload-pill {{
                display: inline-block;
                margin-left: 10px;
                padding: 8px 12px;
                border-radius: 999px;
                background: rgba(102, 227, 196, 0.14);
                border: 1px solid rgba(102, 227, 196, 0.26);
                color: #bdf8ea;
                font-size: 0.88rem;
            }}
            .section-block {{ margin-top: 22px; }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 16px;
                margin-top: 20px;
            }}
            .kpi-card {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 20px;
                backdrop-filter: blur(18px);
                min-height: 130px;
            }}
            .kpi-title {{ color: var(--muted); font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.08em; }}
            .kpi-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 2rem; margin-top: 8px; }}
            .kpi-subtitle {{ color: var(--muted); margin-top: 8px; font-size: 0.95rem; }}
            .panel {{
                background: var(--surface-strong);
                border: 1px solid var(--border);
                border-radius: 26px;
                padding: 22px;
                box-shadow: 0 16px 45px rgba(0, 0, 0, 0.22);
            }}
            .panel-title {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.18rem;
                margin-bottom: 14px;
            }}
            .table {{
                color: var(--text);
                margin-bottom: 0;
                --bs-table-bg: transparent;
                --bs-table-striped-bg: rgba(255,255,255,0.03);
                --bs-table-border-color: rgba(255,255,255,0.08);
            }}
            .table td {{ color: #ffffff !important; }}
            .table thead th {{ color: var(--muted); border-bottom: 1px solid var(--border); }}
            .footer-note {{ color: var(--muted); font-size: 0.9rem; margin-top: 18px; }}
            .section-heading {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.4rem;
                margin: 24px 0 14px;
            }}
            .layout-grid {{ display: grid; grid-template-columns: 1.35fr 0.65fr; gap: 18px; }}
            .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
            .upload-row {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-top: 18px; }}
            .upload-form {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
            .form-control {{
                background: rgba(255,255,255,0.07);
                border: 1px solid var(--border);
                color: var(--text);
                min-width: 280px;
            }}
            .form-control:focus {{
                background: rgba(255,255,255,0.08);
                border-color: rgba(102, 227, 196, 0.55);
                color: var(--text);
                box-shadow: 0 0 0 0.2rem rgba(102, 227, 196, 0.15);
            }}
            .btn-accent {{
                background: linear-gradient(135deg, #66e3c4, #7cb7ff);
                border: none;
                color: #07111f;
                font-weight: 700;
                padding: 10px 18px;
                border-radius: 14px;
            }}
            .btn-accent:hover {{ color: #07111f; opacity: 0.94; }}
            .note-box {{
                border: 1px solid rgba(255,255,255,0.09);
                background: rgba(255,255,255,0.04);
                border-radius: 20px;
                padding: 18px;
            }}
            .note-box h3 {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; }}
            .note-box p {{ color: var(--muted); margin-bottom: 0; }}
            @media (max-width: 1200px) {{
                .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
                .layout-grid, .chart-grid {{ grid-template-columns: 1fr; }}
            }}
            @media (max-width: 768px) {{
                .app-shell {{ padding: 14px 12px 30px; }}
                .hero {{ padding: 24px; border-radius: 22px; }}
                .kpi-grid {{ grid-template-columns: 1fr; }}
                .form-control {{ min-width: 100%; }}
                .panel {{ border-radius: 20px; padding: 16px; }}
            }}
        </style>
    </head>
    <body>
        <div class="app-shell">
            <section class="hero">
                <div class="eyebrow">SmartCart • Customer Segmentation</div>
                <h1>{APP_TITLE}</h1>
            </section>

            <div class="upload-row">
                <form class="upload-form" method="post" enctype="multipart/form-data">
                    <input class="form-control" type="file" name="dataset" accept=".csv">
                    <button class="btn btn-accent" type="submit">Analyze CSV</button>
                </form>
            </div>

            <div class="kpi-grid section-block">
                {kpi_html}
            </div>

            <div class="layout-grid section-block">
                <div class="panel">
                    <div class="panel-title">Cluster Summary</div>
                    <div class="table-responsive">
                        <table class="table table-sm align-middle">
                            <thead>
                                <tr>
                                    <th>Cluster</th>
                                    <th>Customers</th>
                                    <th>Avg Income</th>
                                    <th>Avg Spending</th>
                                    <th>Avg Age</th>
                                    <th>Avg Children</th>
                                </tr>
                            </thead>
                            <tbody>
                                {cluster_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-title">Data Snapshot</div>
                    {cluster_preview}
                </div>
            </div>

            <div class="chart-grid section-block">
                <div class="panel">{figures["clusters"]}</div>
                <div class="panel">{figures["scatter"]}</div>
                <div class="panel">{figures["education"]}</div>
                <div class="panel">{figures["age"]}</div>
            </div>

            <div class="footer-note">Built from the SmartCart notebook analysis. Use this dashboard in your README and resume as a polished portfolio piece.</div>
        </div>
    </body>
    </html>
    """
    return html


@app.route("/", methods=["GET", "POST"])
def home():
    uploaded_file = request.files.get("dataset") if request.method == "POST" else None
    df_raw, source_name = load_data(uploaded_file)
    uploaded_name = uploaded_file.filename if uploaded_file and uploaded_file.filename else None
    cleaned = prepare_data(df_raw)
    if cleaned.empty:
        cleaned = prepare_data(_demo_data())
        source_name = "Built-in demo dataset"
        uploaded_name = None
    clustered, metadata = cluster_customers(cleaned)
    clustered.attrs.update(metadata)
    return render_dashboard(cleaned, clustered, source_name, uploaded_name)


@app.route("/health")
def health():
    return {"status": "ok", "app": "SmartCart Customer Insights"}


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)