import json
import os
import pickle
import random
from collections import Counter
from copy import deepcopy
from glob import glob
from itertools import combinations, permutations
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track
from tqdm import tqdm

from tos.tos_dataset import Document, ToSDataset


def is_isomorphic_multiple(graphs, candidate_graph):
    for motif in graphs:
        DiGM = nx.algorithms.isomorphism.DiGraphMatcher(
            motif,
            candidate_graph,
            edge_match=lambda e1, e2: e1["label_0"] == e2["label_0"],
        )
        if DiGM.is_isomorphic():
            return True
    return False


def load_dataset(file_paths: str, max_per_file: int = None):
    dataset = []
    for file_path in file_paths:
        print(file_path)
        samples = tos_dataset.load_document_corpus(file_path)
        print(f"loaded: {len(samples)} samples")
        if max_per_file is not None:
            random.shuffle(samples)
            samples = samples[:max_per_file]
        dataset.extend(samples)
    print(f"all loaded: {len(dataset)} samples")
    return dataset


if __name__ == "__main__":
    tos_dataset = ToSDataset(
        dmrst_parser_dir="/Users/zaemyung/Development/DMRST_Parser",
        batch_size=1024,
        gpu_id=None,
    )

    motif_paths = [
        "/Users/zaemyung/Development/threads-of-subtlety/data/motifs/hc3-mage_M9_motifs.json",
        "/Users/zaemyung/Development/threads-of-subtlety/data/motifs/old_version/M_9-triangular_HC3-DeepfakeTextDetect.pkl",
    ]
    motif_graphs = []
    for path in motif_paths:
        if path.endswith(".json"):
            with open(path, "r") as f:
                _motifs = json.load(f)
                motif_graphs.extend(
                    [nx.json_graph.node_link_graph(v) for v in _motifs.values()]
                )
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                _motifs = pickle.load(f)
                motif_graphs.extend(_motifs)
    print("[info] amount of before motif graphs: ", len(motif_graphs))

    iso_graphs = []
    for graph in track(
        motif_graphs,
        description="Checking isomorphism",
        total=len(motif_graphs),
        disable=False,
    ):
        if not is_isomorphic_multiple(iso_graphs, graph):
            iso_graphs.append(graph)

    iso_dict = {}
    for G in track(
        iso_graphs,
        description="Converting to dict",
        total=len(iso_graphs),
        disable=False,
    ):
        iso_hash = nx.weisfeiler_lehman_graph_hash(G, edge_attr="label_0")
        G_dict = nx.json_graph.node_link_data(G)
        iso_dict[iso_hash] = G_dict

    print("[info] amount of after motif graphs: ", len(iso_graphs))
    print("[info] amount of after motif graphs: ", len(iso_dict))

    output_path = "data/motifs/hc3-mage_M9_motifs-new.json"
    if os.path.exists(output_path):
        raise ValueError("File already exists")
    with open(output_path, "w") as f:
        json.dump(iso_dict, f, indent=2)
