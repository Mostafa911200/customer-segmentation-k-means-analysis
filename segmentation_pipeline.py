from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


DATA_PATH = Path(__file__).resolve().parent / "Mall_Customers.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: tuple[str, ...]
    n_clusters: int
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "Age and Spending Score": ModelSpec(
        name="Age and Spending Score",
        features=("Age", "Spending Score (1-100)"),
        n_clusters=4,
        description="Segments customers by age and spending behavior.",
    ),
    "Income and Spending Score": ModelSpec(
        name="Income and Spending Score",
        features=("Annual Income (k$)", "Spending Score (1-100)"),
        n_clusters=5,
        description="Notebook's primary business segmentation using income and spending.",
    ),
    "Full Customer Profile": ModelSpec(
        name="Full Customer Profile",
        features=("Age", "Annual Income (k$)", "Spending Score (1-100)"),
        n_clusters=6,
        description="Three-dimensional segmentation using all numeric customer attributes.",
    ),
}


def load_customer_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    for column in cleaned.columns:
        if cleaned[column].isna().any():
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                cleaned[column] = cleaned[column].fillna(cleaned[column].mode(dropna=True).iloc[0])
    return cleaned


def make_kmeans(n_clusters: int) -> KMeans:
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        random_state=111,
        algorithm="lloyd" if n_clusters == 1 else "elkan",
    )


def train_model(df: pd.DataFrame, spec: ModelSpec) -> KMeans:
    model = make_kmeans(spec.n_clusters)
    model.fit(df.loc[:, spec.features].values)
    return model


def train_all_models(df: pd.DataFrame) -> dict[str, KMeans]:
    return {name: train_model(df, spec) for name, spec in MODEL_SPECS.items()}


def compute_inertia_curve(df: pd.DataFrame, spec: ModelSpec, max_clusters: int = 10) -> pd.DataFrame:
    rows = []
    x = df.loc[:, spec.features].values
    for n_clusters in range(1, max_clusters + 1):
        model = make_kmeans(n_clusters)
        model.fit(x)
        rows.append({"Clusters": n_clusters, "Inertia": model.inertia_})
    return pd.DataFrame(rows)


def add_cluster_labels(df: pd.DataFrame, models: dict[str, KMeans]) -> pd.DataFrame:
    labeled = df.copy()
    for name, model in models.items():
        spec = MODEL_SPECS[name]
        labeled[f"{name} Cluster"] = model.predict(labeled.loc[:, spec.features].values)
    return labeled


def clustering_metrics(df: pd.DataFrame, spec: ModelSpec, model: KMeans) -> dict[str, float]:
    x = df.loc[:, spec.features].values
    labels = model.labels_
    return {
        "Clusters": float(spec.n_clusters),
        "Inertia": float(model.inertia_),
        "Silhouette Score": float(silhouette_score(x, labels)),
        "Calinski-Harabasz": float(calinski_harabasz_score(x, labels)),
        "Davies-Bouldin": float(davies_bouldin_score(x, labels)),
    }


def feature_ranges(df: pd.DataFrame, features: tuple[str, ...]) -> dict[str, tuple[float, float, float]]:
    ranges = {}
    for feature in features:
        series = df[feature]
        ranges[feature] = (float(series.min()), float(series.max()), float(series.median()))
    return ranges
