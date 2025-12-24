import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def degree_vs_eigenvector_plot(df: pd.DataFrame, out_png: str, annotate_ids=None) -> pd.DataFrame:
    X = df[["degree"]].values
    y = df["eigenvector"].values

    model = LinearRegression().fit(X, y)
    yhat = model.predict(X)
    resid = y - yhat

    df = df.copy()
    df["eig_residual"] = resid
    df["eig_residual_z"] = (resid - resid.mean()) / resid.std(ddof=0)

    plt.figure()
    plt.scatter(df["degree"], df["eigenvector"], s=10)
    plt.plot(df["degree"], yhat, linewidth=2)
    plt.xlabel("Degree")
    plt.ylabel("Eigenvector Centrality")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Optional: annotate selected IDs
    if annotate_ids:
        sub = df[df["id"].isin(annotate_ids)]
        plt.figure()
        plt.scatter(df["degree"], df["eigenvector"], s=10)
        plt.plot(df["degree"], yhat, linewidth=2)
        for _, r in sub.iterrows():
            plt.annotate(r["name"], (r["degree"], r["eigenvector"]))
        plt.xlabel("Degree")
        plt.ylabel("Eigenvector Centrality")
        plt.savefig(out_png.replace(".png", "_annotated.png"), dpi=200, bbox_inches="tight")
        plt.close()

    return df




def degree_vs_closeness_plot(df, out_png, annotate_ids):
    plt.figure()
    plt.scatter(df["norm_degree"], df["closeness"], s=10)
    for _, r in df[df["id"].isin(annotate_ids)].iterrows():
        plt.annotate(r["name"], (r["norm_degree"], r["closeness"]))
    plt.xlabel("Normalized Degree")
    plt.ylabel("Closeness Centrality")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()



import networkx as nx
import matplotlib.pyplot as plt

def ego_network_plot(G, center_id, out_png, k=0.35):
    ego = nx.ego_graph(G, center_id, radius=1)
    deg = dict(ego.degree())
    sizes = [50 + 10 * deg[n] for n in ego.nodes()]

    pos = nx.spring_layout(ego, k=k, seed=42)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(ego, pos, node_size=sizes)
    nx.draw_networkx_edges(ego, pos, alpha=0.4)

    # label only the center node
    nx.draw_networkx_labels(ego, pos, labels={center_id: str(center_id)})
    plt.axis("off")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()





def bump_chart(rank_df: pd.DataFrame, out_png: str):
    # rank_df columns: ["id","name","neutral","supportive","suppressive"]
    plt.figure(figsize=(10, 6))
    regimes = ["suppressive", "neutral", "supportive"]
    x = range(len(regimes))

    for _, r in rank_df.iterrows():
        plt.plot(x, [r[c] for c in regimes], marker="o")
        # label at the right edge only
        plt.text(x[-1] + 0.02, r["supportive"], r["name"], fontsize=8)

    plt.gca().invert_yaxis()  # rank 1 at top
    plt.xticks(list(x), regimes)
    plt.ylabel("Rank (1 = best)")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()






def deg_vs_eig_plot_and_outliers(
    df: pd.DataFrame,
    out_png: str,
    out_csv: str,
    min_degree_rank: int = 100,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Fits a simple linear model: eigenvector ~ degree.
    Outliers are nodes with large positive residuals (higher eigenvector than predicted)
    while being outside the top-N degree ranks.
    """

    # ranks if not present
    if "degree_rank" not in df.columns:
        df = df.copy()
        df["degree_rank"] = df["degree"].rank(method="min", ascending=False).astype(int)

    X = df[["degree"]].values
    y = df["eigenvector"].values

    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    resid = y - y_hat

    df2 = df.copy()
    df2["eig_pred"] = y_hat
    df2["eig_residual"] = resid

    # Scatter + regression line
    plt.figure(figsize=(7, 5))
    plt.scatter(df2["degree"], df2["eigenvector"], s=10)
    plt.plot(df2["degree"], df2["eig_pred"])
    plt.xlabel("Degree")
    plt.ylabel("Eigenvector Centrality")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Outliers: low-degree side (rank worse than min_degree_rank) + high positive residual
    outliers = (
        df2[df2["degree_rank"] > min_degree_rank]
        .sort_values("eig_residual", ascending=False)
        .head(top_n)
        .loc[:, ["id", "name", "degree", "degree_rank", "eigenvector", "eig_residual"]]
        .copy()
    )

    outliers.to_csv(out_csv, index=False)
    return outliers
