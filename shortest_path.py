"""
Compute all-pairs shortest-path distances on a user-user graph.
"""
import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def _load_edges(social_file: str) -> pd.DataFrame:
    dataframe = pd.read_csv(social_file, index_col=[])
    if not {"user_id_1", "user_id_2"}.issubset(dataframe.columns):
        raise ValueError("trustnetwork.csv must contain user_id_1 and user_id_2 columns")
    return dataframe[["user_id_1", "user_id_2"]]


def _build_adjacency(edge_df: pd.DataFrame, num_nodes: int) -> csr_matrix:
    src = edge_df["user_id_1"].to_numpy(dtype=np.int64)-1
    dst = edge_df["user_id_2"].to_numpy(dtype=np.int64)-1

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.shape[0], dtype=np.int8)

    return csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def compute_shortest_path_distance(
    dataset: str,
    social_file: str = "trustnetwork.csv",
    output_file: str = "shortest_path_result.npy",
    threshold: Optional[int] = None,
) -> np.ndarray:
    """
    Compute all-pairs shortest-path distances using a user-user graph.

    Args:
        dataset: dataset directory containing trustnetwork.csv
        social_file: social edge list filename
        output_file: output .npy filename
        threshold: value for unreachable nodes (defaults to n + 1)
    """
    output_path = os.path.join('./dataset', dataset, output_file)
    if os.path.isfile(output_path):
        return np.load(output_path)

    social_path = os.path.join('./dataset', dataset, social_file)
    edge_df = _load_edges(social_path)

    max_user_id = int(edge_df[["user_id_1", "user_id_2"]].to_numpy().max())
    num_nodes = max_user_id
    unreachable = num_nodes + 1 if threshold is None else int(threshold)

    adjacency = _build_adjacency(edge_df, num_nodes)

    distances = shortest_path(adjacency, directed=False, unweighted=True)
    distances = np.where(np.isinf(distances), unreachable, distances).astype(np.int32)

    np.save(output_path, distances)
    return distances


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute user-user shortest-path distances.")
    parser.add_argument("--dataset", help="Dataset path containing trustnetwork.csv")
    parser.add_argument("--social-file", default="trustnetwork.csv")
    parser.add_argument("--output-file", default="shortest_path_result.npy")
    parser.add_argument("--threshold", type=int, default=None)
    args = parser.parse_args()

    compute_shortest_path_distance(
        dataset=args.dataset,
        social_file=args.social_file,
        output_file=args.output_file,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
