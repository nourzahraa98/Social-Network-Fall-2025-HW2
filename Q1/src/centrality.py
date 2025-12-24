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
