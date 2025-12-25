# Q2/src/run_q2.py
from __future__ import annotations

import argparse
from pathlib import Path   # ← MUST be here
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ADD THIS near the imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]






def load_wikivote_edges(path: Path) -> List[Tuple[int, int]]:
    """
    Wiki-Vote: each non-comment line: "src dst" meaning src -> dst (vote cast by src for dst).
    """
    edges: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            edges.append((int(a), int(b)))
    if not edges:
        raise ValueError(f"No edges loaded from {path}")
    return edges


def build_indexed_graph(edges: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], np.ndarray]:
    """
    Returns:
      src_idx, dst_idx: arrays of edge endpoints in [0..N-1]
      id2idx: mapping original node id -> compact index
      idx2id: array index -> original node id
    """
    nodes = sorted({u for u, v in edges} | {v for u, v in edges})
    idx2id = np.array(nodes, dtype=np.int64)
    id2idx = {nid: i for i, nid in enumerate(nodes)}

    src = np.array([id2idx[u] for u, _ in edges], dtype=np.int64)
    dst = np.array([id2idx[v] for _, v in edges], dtype=np.int64)
    return src, dst, id2idx, idx2id


def pagerank_power_iteration(
    src: np.ndarray,
    dst: np.ndarray,
    n: int,
    alpha: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    """
    PageRank with dangling handling, power iteration.
    Uses incoming edges to update.
    """
    outdeg = np.bincount(src, minlength=n).astype(np.float64)
    pr = np.full(n, 1.0 / n, dtype=np.float64)

    # group incoming by destination
    # We'll iterate over edges vectorized: contribution = pr[src] / outdeg[src]
    for _ in range(max_iter):
        pr_old = pr.copy()

        dangling_mass = pr_old[outdeg == 0].sum()
        base = (1.0 - alpha) / n
        pr = np.full(n, base, dtype=np.float64)

        # incoming contributions from non-dangling sources
        contrib = np.zeros(n, dtype=np.float64)
        valid = outdeg[src] > 0
        contrib_part = pr_old[src[valid]] / outdeg[src[valid]]
        np.add.at(contrib, dst[valid], contrib_part)

        pr += alpha * (contrib + dangling_mass / n)

        if np.linalg.norm(pr - pr_old, ord=1) < tol:
            break

    # normalize (numerical safety)
    pr /= pr.sum()
    return pr


def hits_authority(
    src: np.ndarray,
    dst: np.ndarray,
    n: int,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HITS via power iteration on adjacency using edge lists.
    a = A^T h ; h = A a
    """
    a = np.full(n, 1.0, dtype=np.float64)
    h = np.full(n, 1.0, dtype=np.float64)

    for _ in range(max_iter):
        a_old = a.copy()
        h_old = h.copy()

        # a_i = sum_{j -> i} h_j  (incoming)
        a = np.zeros(n, dtype=np.float64)
        np.add.at(a, dst, h_old[src])

        # h_i = sum_{i -> k} a_k  (outgoing)
        h = np.zeros(n, dtype=np.float64)
        np.add.at(h, src, a[dst])

        # normalize to avoid blow-up
        a_norm = np.linalg.norm(a)
        h_norm = np.linalg.norm(h)
        if a_norm > 0:
            a /= a_norm
        if h_norm > 0:
            h /= h_norm

        if np.linalg.norm(a - a_old, ord=1) < tol and np.linalg.norm(h - h_old, ord=1) < tol:
            break

    return h, a


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    Rank 1 is highest score. Ties broken by stable ordering of argsort.
    Returns rank array where rank[i] is rank of node i.
    """
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_rank_table(
    out_csv: Path,
    idx2id: np.ndarray,
    auth: np.ndarray,
    pr: np.ndarray,
    auth_rank: np.ndarray,
    pr_rank: np.ndarray,
) -> None:
    header = "node_id,authority_score,pagerank_score,authority_rank,pagerank_rank\n"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(header)
        for i in range(len(idx2id)):
            f.write(f"{int(idx2id[i])},{auth[i]:.12e},{pr[i]:.12e},{int(auth_rank[i])},{int(pr_rank[i])}\n")


def plot_loglog_scatter(
    out_png: Path,
    auth_rank: np.ndarray,
    pr_rank: np.ndarray,
) -> None:
    x = auth_rank.astype(np.float64)
    y = pr_rank.astype(np.float64)

    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("HITS Authority Rank (1 = best)")
    plt.ylabel("PageRank Rank (1 = best)")
    plt.title("Wiki-Vote: Authority Rank vs PageRank Rank (log-log)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def resolve_default_input(user_path: str | None) -> Path:
    if user_path is not None:
        return Path(user_path)

    # reasonable defaults
    candidates = [
    PROJECT_ROOT / "data" / "Wiki-Vote.txt",
    PROJECT_ROOT / "Wiki-Vote.txt",
]

    for c in candidates:
        if c.exists():
            return c
    # last resort: show expected
    return Path("Wiki-Vote.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="Path to Wiki-Vote.txt")
    ap.add_argument("--alpha", type=float, default=0.85, help="PageRank damping factor")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()

    in_path = resolve_default_input(args.input)
    edges = load_wikivote_edges(in_path)
    src, dst, _, idx2id = build_indexed_graph(edges)
    n = len(idx2id)

    pr = pagerank_power_iteration(src, dst, n, alpha=args.alpha, tol=args.tol, max_iter=args.max_iter)
    _, auth = hits_authority(src, dst, n, tol=args.tol, max_iter=args.max_iter)

    pr_rank = ranks_from_scores(pr)
    auth_rank = ranks_from_scores(auth)

    out_fig_dir = PROJECT_ROOT / "Q2" / "outputs" / "figures"
    out_tab_dir = PROJECT_ROOT / "Q2" / "outputs" / "tables"
    ensure_dirs(out_fig_dir, out_tab_dir)

    out_csv = out_tab_dir / "q2a_ranks.csv"
    out_png = out_fig_dir / "q2a_hits_authority_vs_pagerank_rank_loglog.png"

    save_rank_table(out_csv, idx2id, auth, pr, auth_rank, pr_rank)
    plot_loglog_scatter(out_png, auth_rank, pr_rank)

    print(f"[OK] Saved ranks: {out_csv}")
    print(f"[OK] Saved figure: {out_png}")


if __name__ == "__main__":
    main()
