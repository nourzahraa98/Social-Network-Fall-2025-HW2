import pandas as pd
import networkx as nx


def load_data(edges_path: str, nodes_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = pd.read_csv(edges_path)
    nodes = pd.read_csv(nodes_path)
    return edges, nodes


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    # mutual likes => undirected graph
    return nx.from_pandas_edgelist(edges, "id_1", "id_2", create_using=nx.Graph())


def id_to_name_map(nodes: pd.DataFrame) -> dict[int, str]:
    return dict(zip(nodes["id"], nodes["page_name"]))
