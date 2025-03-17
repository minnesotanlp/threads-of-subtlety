from glob import glob
from itertools import combinations

import networkx as nx
from tqdm.contrib.concurrent import process_map

from tos.tos_dataset import ToSDataset
from tos.tos_utils import is_motif_present, load_graph_motifs, save_graph_motifs

single_triads_path = "data/motifs/hc3-mage_M3_motifs.json"
motifs_three = load_graph_motifs(single_triads_path).values()
motifs_three = [nx.convert_node_labels_to_integers(m) for m in motifs_three]
print(f"no. of motifs_three: {len(motifs_three)}")


def extract_double_motifs(sample):
    G = sample

    present_motifs = []
    for motif in motifs_three:
        if is_motif_present(G, motif):
            present_motifs.append(motif)

    present_double_motifs = []
    relabels = [
        {"a": 0},
        {"b": 0},
        {"c": 0},
        {"a": 1},
        {"b": 1},
        {"c": 1},
        {"a": 2},
        {"b": 2},
        {"c": 2},
    ]
    for motif_a, motif_b in combinations(present_motifs, 2):
        nx.relabel_nodes(motif_b, {0: "a", 1: "b", 2: "c"}, copy=False)
        for label in relabels:
            re_motif_b = nx.relabel_nodes(motif_b, label, copy=True)
            double_motif = nx.compose(motif_a, re_motif_b)
            if is_motif_present(G, double_motif):
                present_double_motifs.append(double_motif)

    unique_double_motifs = {}
    for motif in present_double_motifs:
        sg_hash = nx.weisfeiler_lehman_graph_hash(motif, edge_attr="label_0")
        unique_double_motifs[sg_hash] = motif
    return unique_double_motifs


if __name__ == "__main__":
    # hc3_file_paths = glob("data/hc3/*.graph_added.jsonl")
    # hc3_dataset = tos_dataset.load_datasets(hc3_file_paths)

    mage_file_paths = glob("data/mage/*.graph_added.jsonl")
    mage_dataset = ToSDataset.load_datasets(mage_file_paths, max_per_file=15000)

    # dataset = hc3_dataset + mage_dataset
    dataset = mage_dataset

    all_graphs = []
    for document in dataset:
        for tree in document.scene_discourse_trees.values():
            all_graphs.append(tree.graph_networkx)
    print(f"no. of all_graphs: {len(all_graphs)}")

    results = process_map(
        extract_double_motifs, all_graphs, max_workers=12, chunksize=1
    )

    double_motifs = {}
    for res in results:
        for sg_hash, motif in res.items():
            double_motifs[sg_hash] = motif
    print(len(double_motifs))

    save_graph_motifs(6, double_motifs.values(), "mage")
