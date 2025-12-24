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
    ego_network_plot,
    bump_chart
)
from Q1.src.bonacich import largest_eigenvalue, bonacich_power
from Q1.src.dynamics import add_rank_columns, classify_rank_shifts


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

    # -------------------------
    # Q1(a).1 Centralities + Top-10s
    # -------------------------
    df = compute_centralities(G, id2name)
    df = add_ranks(df)
    df.to_csv(f"{out_tables}/a1_all_centralities.csv", index=False)

    top_k(df, "norm_degree", 10).to_csv(f"{out_tables}/a1_top10_norm_degree.csv", index=False)
    top_k(df, "eigenvector", 10).to_csv(f"{out_tables}/a1_top10_eigenvector.csv", index=False)
    top_k(df, "closeness", 10).to_csv(f"{out_tables}/a1_top10_closeness.csv", index=False)

    # -------------------------
    # Q1(a).2 Degree vs Eigenvector plot + outliers
    # -------------------------
    deg_vs_eig_plot_and_outliers(
        df=df,
        out_png=f"{out_figures}/a2_deg_vs_eig.png",
        out_csv=f"{out_tables}/a2_lowdeg_higheig_outliers.csv",
        min_degree_rank=100,
        top_n=30,
    )

    # -------------------------
    # Q1(a).3 Three-way case study
    # -------------------------
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

    # -------------------------
    # Q1(b).1 Betweenness + Top-10
    # -------------------------
    df = add_betweenness(df, G, k=800, seed=42)
    df.to_csv(f"{out_tables}/b1_with_betweenness.csv", index=False)

    top10_bet = (
        df.sort_values("betweenness", ascending=False)
          .head(10)
          .loc[:, ["id", "name", "degree", "degree_rank", "betweenness", "betweenness_rank"]]
    )
    top10_bet.to_csv(f"{out_tables}/b1_top10_betweenness.csv", index=False)

    # -------------------------
    # Q1(b).2 Gap table (bridges vs hubs evidence)
    # -------------------------
    gap = betweenness_gap_table(df, top_k_bet=10)
    gap.to_csv(f"{out_tables}/b2_betweenness_gap_table.csv", index=False)

    # -------------------------
    # Q1(c).1 Top-10 closeness
    # -------------------------
    top10_closeness_c1 = (
        df.sort_values("closeness", ascending=False)
          .head(10)
          .loc[:, ["id", "name", "degree", "degree_rank", "closeness", "closeness_rank"]]
    )
    top10_closeness_c1.to_csv(f"{out_tables}/c1_top10_closeness.csv", index=False)

    # -------------------------
    # Q1(c).2 Efficient monitors + plot
    # -------------------------
    efficient_monitors = df[
        (df["closeness_rank"] <= 20) &
        (df["degree_rank"] > 100)
    ].sort_values("closeness_rank")

    efficient_three = efficient_monitors.head(3)
    efficient_three.to_csv(f"{out_tables}/c2_efficient_monitors.csv", index=False)

    if len(efficient_three) > 0:
        degree_vs_closeness_plot(
            df=df,
            out_png=f"{out_figures}/c2_degree_vs_closeness.png",
            annotate_ids=efficient_three["id"].tolist(),
        )

        # -------------------------
        # Q1(c).3 Ego network for one efficient monitor
        # -------------------------
        center_id = int(efficient_three.iloc[0]["id"])
        ego_network_plot(
            G=G,
            center_id=center_id,
            out_png=f"{out_figures}/c3_ego_network_center_{center_id}.png",
            k=0.45,
        )

    # -------------------------
    # Q1(d).1 Bonacich/Katz regimes + scores
    # -------------------------
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

    # -------------------------
    # Q1(d).2 Ranks + classes + readable bump chart (selective labels)
    # -------------------------
    bon_df = add_rank_columns(
        bon_df,
        cols=["bonacich_neutral", "bonacich_supportive", "bonacich_suppressive"]
    )

    bon_df2 = classify_rank_shifts(
        bon_df,
        supportive_rank_col="bonacich_supportive_rank",
        suppressive_rank_col="bonacich_suppressive_rank",
        stable_threshold=20,
        shift_threshold=200,
    )
    bon_df2.to_csv(f"{out_tables}/d2_bonacich_ranks_and_classes.csv", index=False)

    # key nodes = union of top 20 in each regime
    topN = 20
    key_ids = set(
        bon_df2.nsmallest(topN, "bonacich_suppressive_rank")["id"].tolist()
        + bon_df2.nsmallest(topN, "bonacich_neutral_rank")["id"].tolist()
        + bon_df2.nsmallest(topN, "bonacich_supportive_rank")["id"].tolist()
    )

    df_key = bon_df2[bon_df2["id"].isin(key_ids)].copy()
    df_key = df_key.sort_values("bonacich_supportive_rank")
    df_key.to_csv(f"{out_tables}/d2_keynode_rank_trajectories.csv", index=False)

    # label only 3 amplifiers + 3 inhibitors + 3 stable
    label_ids = (
        bon_df2[bon_df2["role_class"] == "Power Amplifier"]
            .sort_values("delta_rank")
            .head(3)["id"].tolist()
        +
        bon_df2[bon_df2["role_class"] == "Power Inhibitor"]
            .sort_values("delta_rank", ascending=False)
            .head(3)["id"].tolist()
        +
        bon_df2[bon_df2["role_class"] == "Stable Actor"]
            .sort_values("bonacich_supportive_rank")
            .head(3)["id"].tolist()
    )

    # IMPORTANT: pass id column + label_ids
    bump_chart(
        df_key=df_key[[
            "id",
            "name",
            "bonacich_suppressive_rank",
            "bonacich_neutral_rank",
            "bonacich_supportive_rank",
        ]],
        out_png=f"{out_figures}/d2_bump_chart_bonacich.png",
        label_ids=label_ids,
    )

    # export top examples per class
    bon_df2[bon_df2["role_class"] == "Power Amplifier"] \
        .sort_values("delta_rank") \
        .head(20) \
        .to_csv(f"{out_tables}/d2_power_amplifiers_top.csv", index=False)

    bon_df2[bon_df2["role_class"] == "Power Inhibitor"] \
        .sort_values("delta_rank", ascending=False) \
        .head(20) \
        .to_csv(f"{out_tables}/d2_power_inhibitors_top.csv", index=False)

    bon_df2[bon_df2["role_class"] == "Stable Actor"] \
        .sort_values("bonacich_supportive_rank") \
        .head(20) \
        .to_csv(f"{out_tables}/d2_stable_actors_top.csv", index=False)

    print("Done: Q1(a)–(d2) outputs written.")


if __name__ == "__main__":
    main()
