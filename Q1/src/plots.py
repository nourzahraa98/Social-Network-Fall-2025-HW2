import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import networkx as nx


def deg_vs_eig_plot_and_outliers(
    df: pd.DataFrame,
    out_png: str,
    out_csv: str,
    min_degree_rank: int = 100,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Scatter plot: Degree (X) vs Eigenvector (Y), with a fitted linear trend line.
    Exports 'outliers' = nodes with degree_rank > min_degree_rank and largest positive residuals.
    """

    df2 = df.copy()

    # Ensure degree_rank exists
    if "degree_rank" not in df2.columns:
        df2["degree_rank"] = df2["degree"].rank(method="min", ascending=False).astype(int)

    X = df2[["degree"]].values
    y = df2["eigenvector"].values

    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    resid = y - y_hat

    df2["eig_pred"] = y_hat
    df2["eig_residual"] = resid

    # Plot scatter + trend line
    plt.figure(figsize=(7, 5))
    plt.scatter(df2["degree"], df2["eigenvector"], s=10)
    plt.plot(df2["degree"], df2["eig_pred"])
    plt.xlabel("Degree")
    plt.ylabel("Eigenvector Centrality")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Export top residual outliers among low-degree (by rank)
    outliers = (
        df2[df2["degree_rank"] > min_degree_rank]
        .sort_values("eig_residual", ascending=False)
        .head(top_n)
        .loc[:, ["id", "name", "degree", "degree_rank", "eigenvector", "eig_residual"]]
        .copy()
    )
    outliers.to_csv(out_csv, index=False)
    return outliers


def degree_vs_closeness_plot(
    df: pd.DataFrame,
    out_png: str,
    annotate_ids: list[int],
):
    """
    Scatter plot: Normalized Degree (X) vs Closeness (Y).
    Annotates only the nodes in annotate_ids.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(df["norm_degree"], df["closeness"], s=10)

    for _, r in df[df["id"].isin(annotate_ids)].iterrows():
        plt.annotate(
            r["name"],
            (r["norm_degree"], r["closeness"]),
            fontsize=9,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel("Normalized Degree")
    plt.ylabel("Closeness Centrality")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def ego_network_plot(
    G: nx.Graph,
    center_id: int,
    out_png: str,
    k: float = 0.35,
    seed: int = 42,
):
    """
    Ego network visualization (radius 1):
    - Spring layout with adjustable k (spacing)
    - Node size proportional to degree (within ego graph)
    - Label only the center node
    """
    ego = nx.ego_graph(G, center_id, radius=1)

    deg_ego = dict(ego.degree())
    sizes = [50 + 15 * deg_ego[n] for n in ego.nodes()]

    pos = nx.spring_layout(ego, k=k, seed=seed)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(ego, pos, node_size=sizes)
    nx.draw_networkx_edges(ego, pos, alpha=0.35)
    nx.draw_networkx_labels(ego, pos, labels={center_id: str(center_id)}, font_size=12)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def bump_chart(df_key: pd.DataFrame, out_png: str, label_ids: list[int] | None = None):
    """
    Bump chart for Bonacich rank trajectories.

    Expects df_key columns:
      - id
      - name
      - bonacich_suppressive_rank
      - bonacich_neutral_rank
      - bonacich_supportive_rank

    label_ids: if provided, label only those node IDs (prevents label clutter).
    """
    regimes = ["suppressive", "neutral", "supportive"]
    x = list(range(len(regimes)))

    plt.figure(figsize=(10, 6))

    label_set = set(label_ids) if label_ids else set()

    for _, r in df_key.iterrows():
        y = [
            r["bonacich_suppressive_rank"],
            r["bonacich_neutral_rank"],
            r["bonacich_supportive_rank"],
        ]
        plt.plot(x, y, marker="o", linewidth=1, alpha=0.7)

        # label only selected nodes (with small deterministic jitter to reduce overlap)
        if (not label_ids) or (int(r["id"]) in label_set):
            jitter = 20 * ((hash(int(r["id"])) % 3) - 1)  # -20, 0, +20
            plt.text(x[-1] + 0.03, y[-1] + jitter, r["name"], fontsize=9, va="center")

    plt.gca().invert_yaxis()
    plt.xticks(x, regimes)
    plt.ylabel("Rank (1 = highest)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
