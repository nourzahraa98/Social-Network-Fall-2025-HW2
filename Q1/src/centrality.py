import pandas as pd
import networkx as nx



def compute_centralities(G: nx.Graph, id2name: dict[int, str]) -> pd.DataFrame:
    n = G.number_of_nodes()

    deg = dict(G.degree())
    norm_deg = {u: d / (n - 1) for u, d in deg.items()}

    # Eigenvector can be slow; increase max_iter to be safe
    eig = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-8)

    clo = nx.closeness_centrality(G)

    df = pd.DataFrame({"id": list(G.nodes())})
    df["name"] = df["id"].map(id2name)
    df["degree"] = df["id"].map(deg)
    df["norm_degree"] = df["id"].map(norm_deg)
    df["eigenvector"] = df["id"].map(eig)
    df["closeness"] = df["id"].map(clo)

    return df


def top_k(df: pd.DataFrame, col: str, k: int = 10) -> pd.DataFrame:
    return df.sort_values(col, ascending=False).head(k).copy()


def add_ranks(df):
    df = df.copy()
    df["degree_rank"] = df["degree"].rank(method="min", ascending=False).astype(int)
    df["eigenvector_rank"] = df["eigenvector"].rank(method="min", ascending=False).astype(int)
    df["closeness_rank"] = df["closeness"].rank(method="min", ascending=False).astype(int)
    return df




def add_betweenness(df, G: nx.Graph, k: int = 800, seed: int = 42):
    """
    Approximate betweenness using k sampled sources.
    Increase k for higher accuracy (slower).
    """
    bet = nx.betweenness_centrality(G, k=k, normalized=True, seed=seed)
    df = df.copy()
    df["betweenness"] = df["id"].map(bet)
    df["betweenness_rank"] = df["betweenness"].rank(method="min", ascending=False).astype(int)
    return df



def betweenness_gap_table(df: pd.DataFrame, top_k_bet: int = 10) -> pd.DataFrame:
    """
    Returns top-k betweenness nodes with degree rank, plus a simple 'bridge_score'
    that highlights high betweenness despite low degree ranking.
    """
    t = (
        df.sort_values("betweenness", ascending=False)
          .head(top_k_bet)
          .loc[:, ["id", "name", "degree", "degree_rank", "betweenness", "betweenness_rank"]]
          .copy()
    )

    # Simple gap indicator: higher means "more bridge-like"
    # (high betweenness rank should be small; degree rank large indicates low degree)
    t["bridge_gap"] = t["degree_rank"]  # simple and interpretable for top-10 betweenness only
    return t.sort_values(["betweenness_rank", "bridge_gap"], ascending=[True, False])
