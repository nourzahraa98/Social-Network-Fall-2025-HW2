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
