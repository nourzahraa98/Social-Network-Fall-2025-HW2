import os
from Q1.src.io import load_data, build_graph, id_to_name_map
from Q1.src.centrality import compute_centralities, top_k
from Q1.src.plots import deg_vs_eig_plot_and_outliers
from Q1.src.centrality import add_ranks


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
        # (a).2 Degree vs Eigenvector scatter + outlier table
    deg_vs_eig_plot_and_outliers(
        df=df,
        out_png="Q1/outputs/figures/a2_deg_vs_eig.png",
        out_csv="Q1/outputs/tables/a2_lowdeg_higheig_outliers.csv",
        min_degree_rank=100,
        top_n=30,
    )
    

        # add ranks
    df = add_ranks(df)

    # Q1(a).3 selection criteria
    candidates = df[
        (df["degree_rank"] > 100) &
        (df["eigenvector_rank"] <= 50)
    ].sort_values("eigenvector_rank")

    # select top 3 by eigenvector rank
    three_way = candidates.head(3)[
        [
            "id",
            "name",
            "degree",
            "degree_rank",
            "eigenvector",
            "eigenvector_rank",
            "closeness",
            "closeness_rank",
        ]
    ]

    three_way.to_csv(
        "Q1/outputs/tables/a3_three_way_case_study.csv",
        index=False
    )


    top_k(df, "norm_degree", 10).to_csv(f"{out_tables}/a1_top10_norm_degree.csv", index=False)
    top_k(df, "eigenvector", 10).to_csv(f"{out_tables}/a1_top10_eigenvector.csv", index=False)
    top_k(df, "closeness", 10).to_csv(f"{out_tables}/a1_top10_closeness.csv", index=False)

    print("Done: wrote Q1(a).1 tables to Q1/outputs/tables/")


if __name__ == "__main__":
    main()
