import json
import random
from glob import glob
from itertools import combinations
from multiprocessing import Manager, Pool

import networkx as nx
from tqdm import tqdm

from tos.tos_dataset import ToSDataset
from tos.tos_utils import is_isomorphic_multiple, save_graph_motifs

shared_list = None


def init_globals(manager_list):
    """Initializer for each child process to set the global shared_list."""
    global shared_list
    shared_list = manager_list


def worker_function(G):
    """
    Checks if 'item' is in the shared_list.
    If not found, appends it.
    Returns a message for demonstration.
    """
    for SG in (G.subgraph(s).copy() for s in combinations(G, 3)):
        if len(list(nx.isolates(SG))):
            continue
        if is_isomorphic_multiple(shared_list, SG):
            continue
        shared_list.append(SG)
    return "P"


if __name__ == "__main__":
    hc3_file_paths = glob("data/hc3/*.graph_added.jsonl")
    hc3_dataset = ToSDataset.load_datasets(hc3_file_paths)

    mage_file_paths = glob("data/mage/*.graph_added.jsonl")
    mage_dataset = ToSDataset.load_datasets(mage_file_paths, max_per_file=15000)

    dataset = hc3_dataset + mage_dataset

    motif_size = 3

    all_graphs = []
    for document in dataset:
        for tree in document.scene_discourse_trees.values():
            all_graphs.append(tree.graph_networkx)
    print(f"no. of all_graphs: {len(all_graphs)}")

    with Manager() as manager:
        manager_list = manager.list()
        with Pool(initializer=init_globals, initargs=(manager_list,)) as pool:
            results = list(
                tqdm(pool.imap(worker_function, all_graphs), total=len(all_graphs))
            )
        motifs = list(manager_list)
        print("Shared list:", motifs)
        print("len:", len(motifs))

    save_graph_motifs(motif_size, motifs, "hc3-mage")
