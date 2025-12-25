# Q2/src/run_q2.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Repo root (…/Q2/src/run_q2.py -> parents[2] is repo root)
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

    for _ in range(max_iter):
        pr_old = pr.copy()

        dangling_mass = pr_old[outdeg == 0].sum()
        base = (1.0 - alpha) / n
        pr = np.full(n, base, dtype=np.float64)

        contrib = np.zeros(n, dtype=np.float64)
        valid = outdeg[src] > 0
        contrib_part = pr_old[src[valid]] / outdeg[src[valid]]
        np.add.at(contrib, dst[valid], contrib_part)

        pr += alpha * (contrib + dangling_mass / n)

        if np.linalg.norm(pr - pr_old, ord=1) < tol:
            break

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

        a = np.zeros(n, dtype=np.float64)
        np.add.at(a, dst, h_old[src])

        h = np.zeros(n, dtype=np.float64)
        np.add.at(h, src, a[dst])

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
            f.write(
                f"{int(idx2id[i])},{auth[i]:.12e},{pr[i]:.12e},{int(auth_rank[i])},{int(pr_rank[i])}\n"
            )


def plot_loglog_scatter(out_png: Path, auth_rank: np.ndarray, pr_rank: np.ndarray) -> None:
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


def compute_degrees(src: np.ndarray, dst: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    outdeg = np.bincount(src, minlength=n).astype(np.int64)
    indeg = np.bincount(dst, minlength=n).astype(np.int64)
    return indeg, outdeg


def in_neighbor_lists(src: np.ndarray, dst: np.ndarray, n: int) -> List[List[int]]:
    inn = [[] for _ in range(n)]
    for u, v in zip(src.tolist(), dst.tolist()):
        inn[v].append(u)
    return inn


def summarize_in_neighbors(
    inn: List[List[int]],
    outdeg: np.ndarray,
    hub: np.ndarray,
    pr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(inn)
    mean_in_outdeg = np.zeros(n, dtype=np.float64)
    mean_in_hub = np.zeros(n, dtype=np.float64)
    mean_in_pr = np.zeros(n, dtype=np.float64)

    for v in range(n):
        neigh = inn[v]
        if not neigh:
            continue
        arr = np.array(neigh, dtype=np.int64)
        mean_in_outdeg[v] = outdeg[arr].mean()
        mean_in_hub[v] = hub[arr].mean()
        mean_in_pr[v] = pr[arr].mean()

    return mean_in_outdeg, mean_in_hub, mean_in_pr


def select_representative_nodes(auth_rank: np.ndarray, pr_rank: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    """
    delta = log10(PR_rank) - log10(Auth_rank)
      +delta: PR rank worse than Authority rank
      -delta: PR rank better than Authority rank
    """
    delta = np.log10(pr_rank.astype(np.float64)) - np.log10(auth_rank.astype(np.float64))
    order_pos = np.argsort(-delta)
    order_neg = np.argsort(delta)
    order_abs = np.argsort(np.abs(delta))

    return {
        "auth_high_pr_low": order_pos[:k],
        "pr_high_auth_low": order_neg[:k],
        "near_diagonal": order_abs[:k],
        "delta": delta,
    }


def save_outlier_table(
    out_csv: Path,
    idx2id: np.ndarray,
    auth: np.ndarray,
    hub: np.ndarray,
    pr: np.ndarray,
    auth_rank: np.ndarray,
    pr_rank: np.ndarray,
    indeg: np.ndarray,
    outdeg: np.ndarray,
    mean_in_outdeg: np.ndarray,
    mean_in_hub: np.ndarray,
    mean_in_pr: np.ndarray,
    picks: Dict[str, np.ndarray],
) -> None:
    delta = picks["delta"]
    chosen = np.unique(
        np.concatenate([picks["auth_high_pr_low"], picks["pr_high_auth_low"], picks["near_diagonal"]])
    )

    header = (
        "node_id,group,authority_rank,pagerank_rank,delta_log10,"
        "indeg,outdeg,authority_score,hub_score,pagerank_score,"
        "mean_in_neighbor_outdeg,mean_in_neighbor_hub,mean_in_neighbor_pagerank\n"
    )

    auth_set = set(picks["auth_high_pr_low"].tolist())
    pr_set = set(picks["pr_high_auth_low"].tolist())
    diag_set = set(picks["near_diagonal"].tolist())

    def group_of(i: int) -> str:
        if i in auth_set:
            return "auth_high_pr_low"
        if i in pr_set:
            return "pr_high_auth_low"
        if i in diag_set:
            return "near_diagonal"
        return "selected"

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(header)
        for i in chosen.tolist():
            f.write(
                f"{int(idx2id[i])},{group_of(i)},"
                f"{int(auth_rank[i])},{int(pr_rank[i])},{delta[i]:.6f},"
                f"{int(indeg[i])},{int(outdeg[i])},"
                f"{auth[i]:.12e},{hub[i]:.12e},{pr[i]:.12e},"
                f"{mean_in_outdeg[i]:.6f},{mean_in_hub[i]:.12e},{mean_in_pr[i]:.12e}\n"
            )


def plot_annotated_outliers(
    out_png: Path,
    auth_rank: np.ndarray,
    pr_rank: np.ndarray,
    idx2id: np.ndarray,
    picks: Dict[str, np.ndarray],
) -> None:
    x = auth_rank.astype(np.float64)
    y = pr_rank.astype(np.float64)

    chosen = np.unique(
        np.concatenate([picks["auth_high_pr_low"], picks["pr_high_auth_low"], picks["near_diagonal"]])
    )

    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.35)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("HITS Authority Rank (1 = best)")
    plt.ylabel("PageRank Rank (1 = best)")
    plt.title("Annotated outliers (log-log)")

    plt.scatter(x[chosen], y[chosen], s=28, alpha=0.9)
    for i in chosen.tolist():
        plt.annotate(str(int(idx2id[i])), (x[i], y[i]), fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def resolve_default_input(user_path: str | None) -> Path:
    if user_path is not None:
        return Path(user_path)

    candidates = [
        PROJECT_ROOT / "data" / "Wiki-Vote.txt",
        PROJECT_ROOT / "Wiki-Vote.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("Wiki-Vote.txt")


# -------------------------
# Q2(b) helpers
# -------------------------
def compute_ranks_for_alphas(
    src: np.ndarray,
    dst: np.ndarray,
    n: int,
    alphas: np.ndarray,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """
    Returns ranks_matrix with shape (len(alphas), n), where ranks_matrix[i, v] is rank of node v at alpha_i.
    """
    ranks_matrix = np.zeros((len(alphas), n), dtype=np.int64)
    for i, a in enumerate(alphas):
        pr = pagerank_power_iteration(src, dst, n, alpha=float(a), tol=tol, max_iter=max_iter)
        ranks_matrix[i] = ranks_from_scores(pr)
    return ranks_matrix


def save_trajectory_table(
    out_csv: Path,
    alphas: np.ndarray,
    idx2id: np.ndarray,
    nodes_idx: List[int],
    ranks_matrix: np.ndarray,
) -> None:
    cols = [f"node_{int(idx2id[i])}" for i in nodes_idx]
    header = "alpha," + ",".join(cols) + "\n"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(header)
        for ai, a in enumerate(alphas):
            row = [f"{float(a):.4f}"]
            row += [str(int(ranks_matrix[ai, ni])) for ni in nodes_idx]
            f.write(",".join(row) + "\n")


def plot_rank_trajectories(
    out_png: Path,
    alphas: np.ndarray,
    idx2id: np.ndarray,
    nodes_idx: List[int],
    ranks_matrix: np.ndarray,
) -> None:
    plt.figure()
    for ni in nodes_idx:
        plt.plot(alphas, ranks_matrix[:, ni], label=str(int(idx2id[ni])))
    plt.xlabel("PageRank damping factor α")
    plt.ylabel("Rank (1 = best)")
    plt.title("Rank trajectories across α (lower is better)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def summarize_trajectories(
    out_csv: Path,
    alphas: np.ndarray,
    idx2id: np.ndarray,
    nodes_idx: List[int],
    ranks_matrix: np.ndarray,
) -> None:
    """
    Summarize each node by:
      - rank at alpha_min and alpha_max
      - best/worst rank across sweep
      - simple slope sign using linear fit (trend)
    """
    x = alphas.astype(np.float64)
    header = "node_id,rank_at_0.50,rank_at_0.99,best_rank,worst_rank,trend_slope\n"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(header)
        for ni in nodes_idx:
            y = ranks_matrix[:, ni].astype(np.float64)
            # linear trend
            slope = np.polyfit(x, y, 1)[0]
            f.write(
                f"{int(idx2id[ni])},"
                f"{int(y[0])},{int(y[-1])},"
                f"{int(y.min())},{int(y.max())},"
                f"{slope:.6f}\n"
            )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="Path to Wiki-Vote.txt")
    ap.add_argument("--alpha", type=float, default=0.85, help="PageRank damping factor (used in part a)")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-10)

    ap.add_argument("--part", type=str, default="a", choices=["a", "b"], help="Run Q2 part a or b")
    ap.add_argument("--alphas", type=int, default=25, help="Number of alpha points in [0.50,0.99] (part b)")
    ap.add_argument("--k", type=int, default=10, help="Top-k nodes to track (part b); also outlier k (part a)")

    args = ap.parse_args()

    in_path = resolve_default_input(args.input)
    edges = load_wikivote_edges(in_path)
    src, dst, id2idx, idx2id = build_indexed_graph(edges)
    n = len(idx2id)

    out_fig_dir = PROJECT_ROOT / "Q2" / "outputs" / "figures"
    out_tab_dir = PROJECT_ROOT / "Q2" / "outputs" / "tables"
    ensure_dirs(out_fig_dir, out_tab_dir)

    # -------------------------
    # Q2(a)
    # -------------------------
    if args.part == "a":
        pr = pagerank_power_iteration(src, dst, n, alpha=args.alpha, tol=args.tol, max_iter=args.max_iter)
        hub, auth = hits_authority(src, dst, n, tol=args.tol, max_iter=args.max_iter)

        pr_rank = ranks_from_scores(pr)
        auth_rank = ranks_from_scores(auth)

        indeg, outdeg = compute_degrees(src, dst, n)
        inn = in_neighbor_lists(src, dst, n)
        mean_in_outdeg, mean_in_hub, mean_in_pr = summarize_in_neighbors(inn, outdeg, hub, pr)

        picks = select_representative_nodes(auth_rank, pr_rank, k=args.k)

        out_csv = out_tab_dir / "q2a_ranks.csv"
        out_png = out_fig_dir / "q2a_hits_authority_vs_pagerank_rank_loglog.png"
        save_rank_table(out_csv, idx2id, auth, pr, auth_rank, pr_rank)
        plot_loglog_scatter(out_png, auth_rank, pr_rank)

        out_outliers_csv = out_tab_dir / "q2a_outliers.csv"
        out_outliers_png = out_fig_dir / "q2a_outliers_annotated.png"
        save_outlier_table(
            out_outliers_csv,
            idx2id,
            auth,
            hub,
            pr,
            auth_rank,
            pr_rank,
            indeg,
            outdeg,
            mean_in_outdeg,
            mean_in_hub,
            mean_in_pr,
            picks,
        )
        plot_annotated_outliers(out_outliers_png, auth_rank, pr_rank, idx2id, picks)

        print(f"[OK] Saved ranks: {out_csv}")
        print(f"[OK] Saved figure: {out_png}")
        print(f"[OK] Saved outliers: {out_outliers_csv}")
        print(f"[OK] Saved annotated figure: {out_outliers_png}")
        return

    # -------------------------
    # Q2(b)
    # -------------------------
    # Reference alpha=0.85 for "top-10"
    pr_ref = pagerank_power_iteration(src, dst, n, alpha=0.85, tol=args.tol, max_iter=args.max_iter)
    pr_ref_rank = ranks_from_scores(pr_ref)
    topk_idx = np.argsort(pr_ref_rank)[: args.k].tolist()

    # Use anomalies from Q2(a) if available; else compute them quickly here
    anomalies_idx: List[int] = []
    outliers_path = out_tab_dir / "q2a_outliers.csv"
    if outliers_path.exists():
        node_ids: List[int] = []
        with outliers_path.open("r", encoding="utf-8") as f:
            next(f)
            for line in f:
                node_ids.append(int(line.split(",", 1)[0]))
        anomalies_idx = [id2idx[nid] for nid in sorted(set(node_ids)) if nid in id2idx]
    else:
        # fallback: compute anomalies using HITS+PR at alpha=0.85
        hub, auth = hits_authority(src, dst, n, tol=args.tol, max_iter=args.max_iter)
        auth_rank = ranks_from_scores(auth)
        picks = select_representative_nodes(auth_rank, pr_ref_rank, k=max(5, min(args.k, 10)))
        anomalies_idx = np.unique(
            np.concatenate([picks["auth_high_pr_low"], picks["pr_high_auth_low"], picks["near_diagonal"]])
        ).tolist()

    # unique, preserve order
    nodes_to_track = list(dict.fromkeys(topk_idx + anomalies_idx))

    alphas = np.linspace(0.50, 0.99, int(args.alphas), dtype=np.float64)
    ranks_matrix = compute_ranks_for_alphas(src, dst, n, alphas, tol=args.tol, max_iter=args.max_iter)

    out_csv = out_tab_dir / "q2b_rank_trajectories.csv"
    out_png = out_fig_dir / "q2b_rank_trajectories.png"
    save_trajectory_table(out_csv, alphas, idx2id, nodes_to_track, ranks_matrix)
    plot_rank_trajectories(out_png, alphas, idx2id, nodes_to_track, ranks_matrix)
    
    out_sum = out_tab_dir / "q2b_trajectory_summary.csv"
    summarize_trajectories(out_sum, alphas, idx2id, nodes_to_track, ranks_matrix)
    print(f"[OK] Saved trajectory summary: {out_sum}")

    print(f"[OK] Tracked nodes: {len(nodes_to_track)} (top-{args.k} @ alpha=0.85 + anomalies)")
    print(f"[OK] Saved trajectories table: {out_csv}")
    print(f"[OK] Saved trajectories figure: {out_png}")


if __name__ == "__main__":
    main()
