from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from segmentation_pipeline import (
    MODEL_SPECS,
    add_cluster_labels,
    clean_customer_data,
    clustering_metrics,
    compute_inertia_curve,
    feature_ranges,
    load_customer_data,
    train_all_models,
)


COLORS = {
    "background": "#0E1117",
    "sidebar": "#161B22",
    "card": "#1C212B",
    "input": "#262D38",
    "text": "#F3F4F6",
    "muted": "#AAB2BF",
    "border": "#303744",
    "accent": "#FF4B5C",
}

px.defaults.template = "plotly_dark"


st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --app-bg: #0E1117;
        --sidebar-bg: #161B22;
        --surface: #1C212B;
        --input-bg: #262D38;
        --text-main: #F3F4F6;
        --text-muted: #AAB2BF;
        --border: #303744;
        --accent: #FF4B5C;
    }

    .stApp {
        background: var(--app-bg);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        min-height: 96px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        color: var(--text-muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--text-main);
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: var(--sidebar-bg);
        color: var(--text-main);
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-main);
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] small {
        color: var(--text-muted);
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border);
    }

    [data-testid="stSidebar"] [data-testid="stCodeBlock"] {
        background: var(--input-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] textarea {
        background-color: var(--input-bg);
        border-color: var(--border);
        color: var(--text-main);
        border-radius: 8px;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="input"] input {
        color: var(--text-main);
    }

    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: rgba(255, 75, 92, 0.16);
        border: 1px solid rgba(255, 75, 92, 0.35);
        color: var(--text-main);
    }

    [data-testid="stSidebar"] [role="slider"] {
        background-color: var(--accent);
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(255, 75, 92, 0.18);
    }

    [data-testid="stSidebar"] [data-baseweb="checkbox"] span {
        color: var(--text-main);
    }

    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [data-baseweb="menu"] {
        background: var(--input-bg);
        border: 1px solid var(--border);
        color: var(--text-main);
    }

    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] li {
        background: var(--input-bg);
        color: var(--text-main);
    }

    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background: rgba(255, 75, 92, 0.18);
        color: var(--text-main);
    }

    .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 0.85rem;
        border: 1px solid var(--border);
        color: var(--text-muted);
        background: var(--surface);
    }

    .stTabs [aria-selected="true"] {
        color: var(--text-main);
        border-color: var(--accent);
    }

    .section-note {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_customer_data()
    cleaned = clean_customer_data(raw)
    return raw, cleaned


@st.cache_resource
def get_models(cleaned: pd.DataFrame):
    return train_all_models(cleaned)


@st.cache_data
def get_labeled_data(cleaned: pd.DataFrame) -> pd.DataFrame:
    return add_cluster_labels(cleaned, get_models(cleaned))


@st.cache_data
def get_metrics(cleaned: pd.DataFrame) -> pd.DataFrame:
    models = get_models(cleaned)
    rows = []
    for name, spec in MODEL_SPECS.items():
        row = {"Model": name, **clustering_metrics(cleaned, spec, models[name])}
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def get_inertia(cleaned: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return compute_inertia_curve(cleaned, MODEL_SPECS[model_name])


def format_metric(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.3f}"


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    return fig


def profile_cluster(labeled: pd.DataFrame, cluster_column: str, cluster_id: int) -> pd.DataFrame:
    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    segment = labeled[labeled[cluster_column] == cluster_id]
    return segment[numeric_cols].mean().to_frame("Cluster average")


def cluster_profile_table(labeled: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    profile = (
        labeled.groupby(cluster_column)
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Income=("Annual Income (k$)", "mean"),
            Avg_Spending=("Spending Score (1-100)", "mean"),
            Female_Share=("Gender", lambda s: (s == "Female").mean()),
        )
        .reset_index()
        .rename(columns={cluster_column: "Cluster"})
    )
    profile["Share"] = profile["Customers"] / profile["Customers"].sum()
    return profile


def filter_customers(
    df: pd.DataFrame,
    genders: list[str],
    age_range: tuple[int, int],
    income_range: tuple[int, int],
    spending_range: tuple[int, int],
) -> pd.DataFrame:
    filtered = df[df["Gender"].isin(genders)].copy()
    filtered = filtered[filtered["Age"].between(age_range[0], age_range[1])]
    filtered = filtered[filtered["Annual Income (k$)"].between(income_range[0], income_range[1])]
    filtered = filtered[filtered["Spending Score (1-100)"].between(spending_range[0], spending_range[1])]
    return filtered


def nearest_customers(
    labeled: pd.DataFrame,
    features: tuple[str, ...],
    values: dict[str, float],
    cluster_column: str,
    cluster_id: int,
    top_n: int = 5,
) -> pd.DataFrame:
    comparison = labeled[labeled[cluster_column] == cluster_id].copy()
    input_array = np.array([values[feature] for feature in features])
    comparison["Distance"] = np.linalg.norm(comparison.loc[:, features].values - input_array, axis=1)
    columns = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Distance"]
    return comparison.sort_values("Distance").loc[:, columns].head(top_n)


def make_2d_cluster_chart(labeled: pd.DataFrame, model_name: str, model, show_gender: bool) -> go.Figure:
    spec = MODEL_SPECS[model_name]
    cluster_column = f"{model_name} Cluster"
    labels = {cluster_column: "Cluster"}
    hover_data = ["CustomerID", "Gender"] if "CustomerID" in labeled.columns else ["Gender"]
    symbol = "Gender" if show_gender else None
    fig = px.scatter(
        labeled,
        x=spec.features[0],
        y=spec.features[1],
        color=cluster_column,
        symbol=symbol,
        hover_data=hover_data,
        color_continuous_scale="Viridis",
        labels=labels,
        height=520,
    )
    centroids = model.cluster_centers_
    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers+text",
            marker=dict(color="#dc2626", size=16, symbol="x"),
            text=[f"C{i}" for i in range(len(centroids))],
            textposition="top center",
            name="Centroids",
        )
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
    return apply_chart_theme(fig)


def make_3d_cluster_chart(labeled: pd.DataFrame, model_name: str) -> go.Figure:
    cluster_column = f"{model_name} Cluster"
    fig = px.scatter_3d(
        labeled,
        x="Age",
        y="Spending Score (1-100)",
        z="Annual Income (k$)",
        color=cluster_column,
        hover_data=["CustomerID", "Gender"],
        color_continuous_scale="Viridis",
        height=620,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.82))
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
    return apply_chart_theme(fig)


raw_df, clean_df = get_data()
models = get_models(clean_df)
labeled_df = get_labeled_data(clean_df)
metrics_df = get_metrics(clean_df)

st.sidebar.title("Customer Segmentation")
st.sidebar.caption("Explore notebook-based K-Means customer clusters.")
selected_model_name = st.sidebar.selectbox(
    "Model",
    list(MODEL_SPECS.keys()),
    index=1,
)
selected_spec = MODEL_SPECS[selected_model_name]
selected_model = models[selected_model_name]
show_gender = st.sidebar.checkbox("Show gender markers", value=True)

st.sidebar.divider()
st.sidebar.subheader("Dataset Filters")
gender_options = sorted(clean_df["Gender"].unique().tolist())
selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
age_filter = st.sidebar.slider(
    "Age",
    int(clean_df["Age"].min()),
    int(clean_df["Age"].max()),
    (int(clean_df["Age"].min()), int(clean_df["Age"].max())),
)
income_filter = st.sidebar.slider(
    "Annual Income (k$)",
    int(clean_df["Annual Income (k$)"].min()),
    int(clean_df["Annual Income (k$)"].max()),
    (int(clean_df["Annual Income (k$)"].min()), int(clean_df["Annual Income (k$)"].max())),
)
spending_filter = st.sidebar.slider(
    "Spending Score",
    int(clean_df["Spending Score (1-100)"].min()),
    int(clean_df["Spending Score (1-100)"].max()),
    (int(clean_df["Spending Score (1-100)"].min()), int(clean_df["Spending Score (1-100)"].max())),
)
if not selected_genders:
    st.sidebar.warning("Select at least one gender to show filtered results.")
    selected_genders = gender_options

filtered_labeled_df = filter_customers(
    labeled_df,
    selected_genders,
    age_filter,
    income_filter,
    spending_filter,
)
selected_cluster_column = f"{selected_model_name} Cluster"
selected_profile = cluster_profile_table(labeled_df, selected_cluster_column)

st.sidebar.divider()
st.sidebar.caption("Notebook task")
st.sidebar.write("Unsupervised customer clustering with K-Means.")
st.sidebar.caption("Launch")
st.sidebar.code("streamlit run streamlit_app.py", language="bash")

st.title("Customer Segmentation and K-Means Analysis")
st.write(
    "Interactive Streamlit application built from the existing Mall Customers notebook. "
    "The app reuses the notebook's K-Means feature sets, cluster counts, and random state."
)

summary_cols = st.columns(4)
summary_cols[0].metric("Rows", f"{clean_df.shape[0]:,}")
summary_cols[1].metric("Filtered Rows", f"{filtered_labeled_df.shape[0]:,}")
summary_cols[2].metric("Missing Values", f"{int(clean_df.isna().sum().sum()):,}")
summary_cols[3].metric("Active Model Clusters", f"{selected_spec.n_clusters}")

tab_overview, tab_eda, tab_preprocess, tab_models, tab_predict = st.tabs(
    ["Overview", "Data Exploration", "Preprocessing", "Models and Results", "Predict Segment"]
)

with tab_overview:
    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Dataset")
        st.dataframe(filtered_labeled_df, width="stretch", hide_index=True)
        st.download_button(
            "Download filtered data",
            filtered_labeled_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customer_segments.csv",
            mime="text/csv",
        )
    with right:
        st.subheader("Notebook Models")
        model_cards = pd.DataFrame(
            [
                {
                    "Model": spec.name,
                    "Features": ", ".join(spec.features),
                    "Clusters": spec.n_clusters,
                }
                for spec in MODEL_SPECS.values()
            ]
        )
        st.dataframe(model_cards, width="stretch", hide_index=True)
        st.subheader("Current Selection")
        st.write(selected_spec.description)
        st.write(f"Features: {', '.join(selected_spec.features)}")
        st.write(f"Clusters: {selected_spec.n_clusters}")
        st.write(f"Filtered customers: {filtered_labeled_df.shape[0]} of {labeled_df.shape[0]}")

with tab_eda:
    st.subheader("Data Quality")
    q1, q2 = st.columns(2)
    q1.dataframe(
        raw_df.isna().sum().rename("Missing values").to_frame(),
        width="stretch",
    )
    q2.dataframe(
        pd.DataFrame(
            {
                "Column": raw_df.columns,
                "Type": [str(dtype) for dtype in raw_df.dtypes],
                "Unique values": [raw_df[col].nunique() for col in raw_df.columns],
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Descriptive Statistics")
    st.dataframe(raw_df.describe(include="all").T, width="stretch")

    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    hist_col, bar_col = st.columns([1.2, 0.8])
    with hist_col:
        selected_numeric = st.selectbox("Histogram feature", numeric_cols)
        fig = px.histogram(
            filtered_labeled_df,
            x=selected_numeric,
            color="Gender",
            marginal="box",
            nbins=20,
            barmode="overlay",
            height=430,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")
    with bar_col:
        gender_counts = filtered_labeled_df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig = px.bar(gender_counts, x="Gender", y="Count", color="Gender", height=430)
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    scatter_left, heatmap_right = st.columns([1.1, 0.9])
    with scatter_left:
        fig = px.scatter_matrix(
            filtered_labeled_df,
            dimensions=numeric_cols,
            color="Gender",
            height=560,
        )
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")
    with heatmap_right:
        corr = filtered_labeled_df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            height=560,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

with tab_preprocess:
    st.subheader("Preprocessing Consistent With the Notebook")
    steps = pd.DataFrame(
        [
            {
                "Step": "Load CSV",
                "Applied": "Yes",
                "Detail": "Reads Mall_Customers.csv.",
            },
            {
                "Step": "Missing values",
                "Applied": "Checked",
                "Detail": "No missing values in the supplied dataset; app includes safe median/mode filling.",
            },
            {
                "Step": "Duplicate rows",
                "Applied": "Checked",
                "Detail": "No duplicate rows detected; app drops duplicates before fitting.",
            },
            {
                "Step": "Encoding",
                "Applied": "No",
                "Detail": "Gender is used for exploration only; the notebook's K-Means models use numeric features.",
            },
            {
                "Step": "Scaling",
                "Applied": "No",
                "Detail": "The original notebook fits K-Means on raw numeric values, so the app does the same.",
            },
            {
                "Step": "CustomerID",
                "Applied": "Excluded",
                "Detail": "Identifier column is displayed but not used as a clustering feature.",
            },
        ]
    )
    st.dataframe(steps, width="stretch", hide_index=True)

    with st.expander("Cleaned dataset preview"):
        st.dataframe(clean_df, width="stretch", hide_index=True)

with tab_models:
    st.subheader("Model Evaluation")
    st.markdown(
        '<div class="section-note">Because this is clustering, the app reports clustering metrics rather than classification or regression scores.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        metrics_df.assign(
            **{
                "Clusters": metrics_df["Clusters"].astype(int),
                "Inertia": metrics_df["Inertia"].map(lambda value: f"{value:,.2f}"),
                "Silhouette Score": metrics_df["Silhouette Score"].map(lambda value: f"{value:.3f}"),
                "Calinski-Harabasz": metrics_df["Calinski-Harabasz"].map(lambda value: f"{value:,.2f}"),
                "Davies-Bouldin": metrics_df["Davies-Bouldin"].map(lambda value: f"{value:.3f}"),
            }
        ),
        width="stretch",
        hide_index=True,
    )

    selected_metrics = metrics_df[metrics_df["Model"] == selected_model_name].iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Clusters", str(int(selected_metrics["Clusters"])))
    metric_cols[1].metric("Inertia", format_metric(selected_metrics["Inertia"]))
    metric_cols[2].metric("Silhouette", format_metric(selected_metrics["Silhouette Score"]))
    metric_cols[3].metric("Davies-Bouldin", format_metric(selected_metrics["Davies-Bouldin"]))

    elbow_col, cluster_col = st.columns([0.9, 1.1])
    with elbow_col:
        inertia_df = get_inertia(clean_df, selected_model_name)
        fig = px.line(
            inertia_df,
            x="Clusters",
            y="Inertia",
            markers=True,
            title=f"Elbow Curve: {selected_model_name}",
            height=470,
        )
        fig.add_vline(
            x=selected_spec.n_clusters,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Notebook choice: {selected_spec.n_clusters}",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")
    with cluster_col:
        if len(selected_spec.features) == 3:
            st.plotly_chart(make_3d_cluster_chart(labeled_df, selected_model_name), width="stretch")
        else:
            st.plotly_chart(
                make_2d_cluster_chart(labeled_df, selected_model_name, selected_model, show_gender),
                width="stretch",
            )

    st.subheader("Cluster Profiles")
    display_profile = selected_profile.copy()
    display_profile["Share"] = display_profile["Share"].map(lambda value: f"{value:.1%}")
    display_profile["Female_Share"] = display_profile["Female_Share"].map(lambda value: f"{value:.1%}")
    st.dataframe(display_profile.round(2), width="stretch", hide_index=True)

    profile_fig = px.bar(
        selected_profile,
        x="Cluster",
        y="Customers",
        color="Avg_Spending",
        color_continuous_scale="Viridis",
        text="Customers",
        height=360,
        title="Cluster Size and Average Spending",
    )
    profile_fig.update_layout(coloraxis_colorbar_title="Avg spending", margin=dict(l=10, r=10, t=45, b=10))
    profile_fig = apply_chart_theme(profile_fig)
    st.plotly_chart(profile_fig, width="stretch")

with tab_predict:
    st.subheader("Interactive Segment Prediction")
    st.write(
        "Enter a customer profile and assign it to the nearest K-Means segment. "
        "K-Means does not provide class probabilities, so distances to centroids are shown instead."
    )

    ranges = feature_ranges(clean_df, selected_spec.features)
    with st.form("prediction_form"):
        input_cols = st.columns(len(selected_spec.features))
        values = {}
        for idx, feature in enumerate(selected_spec.features):
            min_value, max_value, median_value = ranges[feature]
            values[feature] = input_cols[idx].number_input(
                feature,
                min_value=min_value,
                max_value=max_value,
                value=median_value,
                step=1.0,
            )
        submitted = st.form_submit_button("Predict Segment", type="primary")

    if submitted:
        input_array = np.array([[values[feature] for feature in selected_spec.features]])
        cluster_id = int(selected_model.predict(input_array)[0])
        distances = selected_model.transform(input_array)[0]
        nearest_distance = float(distances[cluster_id])
        sorted_distances = np.sort(distances)
        separation = float(sorted_distances[1] - sorted_distances[0]) if len(sorted_distances) > 1 else 0.0
        distance_df = pd.DataFrame(
            {
                "Cluster": [f"Cluster {idx}" for idx in range(len(distances))],
                "Distance to centroid": distances,
            }
        ).sort_values("Distance to centroid")

        result_cols = st.columns([0.7, 1.3])
        result_cols[0].metric("Predicted Segment", f"Cluster {cluster_id}")
        result_cols[0].metric("Nearest Distance", format_metric(nearest_distance))
        result_cols[0].metric("Distance Margin", format_metric(separation))
        with result_cols[1]:
            fig = px.bar(
                distance_df,
                x="Cluster",
                y="Distance to centroid",
                color="Distance to centroid",
                color_continuous_scale="Tealrose",
                height=320,
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=10, t=25, b=10))
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Assigned Cluster Profile")
        st.dataframe(profile_cluster(labeled_df, f"{selected_model_name} Cluster", cluster_id).round(2))

        st.subheader("Nearest Existing Customers")
        st.dataframe(
            nearest_customers(
                labeled_df,
                selected_spec.features,
                values,
                selected_cluster_column,
                cluster_id,
            ).round(2),
            width="stretch",
            hide_index=True,
        )
