import os
import pandas as pd
from Q1.src.io import load_data, build_graph, id_to_name_map
from Q1.src.centrality import (
    compute_centralities, top_k, add_ranks,
    add_betweenness, betweenness_gap_table
)
from Q1.src.plots import (
    deg_vs_eig_plot_and_outliers,
    degree_vs_closeness_plot,
    ego_network_plot
)
from Q1.src.bonacich import largest_eigenvalue, bonacich_power


def main():
    out_tables = "Q1/outputs/tables"
    out_figures = "Q1/outputs/figures"
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_figures, exist_ok=True)

    edges_path = "data/politician_edges.csv"
    nodes_path = "data/politician_nodes.csv"

    edges, nodes = load_data(edges_path, nodes_path)
    G = build_graph(edges)
    id2name = id_to_name_map(nodes)

    # (a).1 centralities
    df = compute_centralities(G, id2name)
    df = add_ranks(df)
    df.to_csv(f"{out_tables}/a1_all_centralities.csv", index=False)

    top_k(df, "norm_degree", 10).to_csv(f"{out_tables}/a1_top10_norm_degree.csv", index=False)
    top_k(df, "eigenvector", 10).to_csv(f"{out_tables}/a1_top10_eigenvector.csv", index=False)
    top_k(df, "closeness", 10).to_csv(f"{out_tables}/a1_top10_closeness.csv", index=False)

    # (a).2 plot + outliers
    deg_vs_eig_plot_and_outliers(
        df=df,
        out_png=f"{out_figures}/a2_deg_vs_eig.png",
        out_csv=f"{out_tables}/a2_lowdeg_higheig_outliers.csv",
        min_degree_rank=100,
        top_n=30,
    )

    # (a).3 three-way case study
    candidates = df[
        (df["degree_rank"] > 100) &
        (df["eigenvector_rank"] <= 50)
    ].sort_values("eigenvector_rank")

    three_way = candidates.head(3)[[
        "id", "name",
        "degree", "degree_rank",
        "eigenvector", "eigenvector_rank",
        "closeness", "closeness_rank",
    ]]
    three_way.to_csv(f"{out_tables}/a3_three_way_case_study.csv", index=False)

    # (b).1 betweenness + top10
    df = add_betweenness(df, G, k=800, seed=42)
    df.to_csv(f"{out_tables}/b1_with_betweenness.csv", index=False)

    top10_bet = (
        df.sort_values("betweenness", ascending=False)
          .head(10)
          .loc[:, ["id", "name", "degree", "degree_rank", "betweenness", "betweenness_rank"]]
    )
    top10_bet.to_csv(f"{out_tables}/b1_top10_betweenness.csv", index=False)

    # (b).2 gap table
    gap = betweenness_gap_table(df, top_k_bet=10)
    gap.to_csv(f"{out_tables}/b2_betweenness_gap_table.csv", index=False)

    # (c).1 top10 closeness
    top10_closeness_c1 = (
        df.sort_values("closeness", ascending=False)
          .head(10)
          .loc[:, ["id", "name", "degree", "degree_rank", "closeness", "closeness_rank"]]
    )
    top10_closeness_c1.to_csv(f"{out_tables}/c1_top10_closeness.csv", index=False)

    # (c).2 efficient monitors: closeness top 20 + degree outside top 100
    efficient_monitors = df[
        (df["closeness_rank"] <= 20) &
        (df["degree_rank"] > 100)
    ].sort_values("closeness_rank")

    efficient_three = efficient_monitors.head(3)
    efficient_three.to_csv(f"{out_tables}/c2_efficient_monitors.csv", index=False)

    # Guard: if fewer than 1 found, don't crash
    if len(efficient_three) > 0:
        # plot first (c.2)
        degree_vs_closeness_plot(
            df=df,
            out_png=f"{out_figures}/c2_degree_vs_closeness.png",
            annotate_ids=efficient_three["id"].tolist(),
        )

        # ego network (c.3) using first
        center_id = int(efficient_three.iloc[0]["id"])
        ego_network_plot(
            G=G,
            center_id=center_id,
            out_png=f"{out_figures}/c3_ego_network_center_{center_id}.png",
            k=0.45,
        )

    # (d).1 bonacich regimes
    lambda_max = largest_eigenvalue(G)
    betas = {
        "neutral": 1e-4,
        "supportive": 0.9 / lambda_max,
        "suppressive": -0.9 / lambda_max,
    }

    bonacich_scores = {reg: bonacich_power(G, b) for reg, b in betas.items()}

    bon_df = pd.DataFrame({
        "id": list(G.nodes()),
        "bonacich_neutral": [bonacich_scores["neutral"].get(n) for n in G.nodes()],
        "bonacich_supportive": [bonacich_scores["supportive"].get(n) for n in G.nodes()],
        "bonacich_suppressive": [bonacich_scores["suppressive"].get(n) for n in G.nodes()],
    })
    bon_df["name"] = bon_df["id"].map(id2name)
    bon_df.to_csv(f"{out_tables}/d1_bonacich_scores.csv", index=False)

    print("Done: Q1(a)–(d1) outputs written.")


if __name__ == "__main__":
    main()
