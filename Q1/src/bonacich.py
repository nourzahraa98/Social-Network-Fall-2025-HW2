import networkx as nx
from scipy.sparse.linalg import eigsh


def largest_eigenvalue(G: nx.Graph) -> float:
    """Largest eigenvalue (spectral radius) of adjacency matrix."""
    A = nx.to_scipy_sparse_array(G, dtype=float, format="csr")
    vals = eigsh(A, k=1, which="LA", return_eigenvectors=False)
    return float(vals[0])


def bonacich_power(G: nx.Graph, beta: float) -> dict:
    """
    Bonacich-style power centrality implemented via Katz centrality:
      x = alpha * A x + b * 1
    Here, alpha corresponds to the assignment's beta parameter.
    """
    return nx.katz_centrality(
        G,
        alpha=beta,      # assignment's β
        beta=1.0,        # constant exogenous term
        normalized=True,
        max_iter=5000,
        tol=1e-8,
    )
