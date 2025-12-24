import os
from Q1.src.io import load_data, build_graph, id_to_name_map
from Q1.src.centrality import compute_centralities, top_k


def main():
    # outputs
    out_tables = "Q1/outputs/tables"
    os.makedirs(out_tables, exist_ok=True)

    # data paths (your data folder is at repo root)
    edges_path = "data/politician_edges.csv"
    nodes_path = "data/politician_nodes.csv"

    edges, nodes = load_data(edges_path, nodes_path)
    G = build_graph(edges)
    id2name = id_to_name_map(nodes)

    df = compute_centralities(G, id2name)
    df.to_csv(f"{out_tables}/a1_all_centralities.csv", index=False)

    top_k(df, "norm_degree", 10).to_csv(f"{out_tables}/a1_top10_norm_degree.csv", index=False)
    top_k(df, "eigenvector", 10).to_csv(f"{out_tables}/a1_top10_eigenvector.csv", index=False)
    top_k(df, "closeness", 10).to_csv(f"{out_tables}/a1_top10_closeness.csv", index=False)

    print("Done: wrote Q1(a).1 tables to Q1/outputs/tables/")


if __name__ == "__main__":
    main()
