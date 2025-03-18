import json
import math
import os
import random
from glob import glob
from typing import Any, Dict, List

import networkx as nx
from tqdm.contrib.concurrent import process_map

from tos.tos_dataset import DiscourseMotifDists, Document, ToSDataset
from tos.tos_utils import load_json

random.seed(42)


selected_motif_hashes = load_json("data/motifs/hc3-mage_selected-motif-hashes.json")

m3_motifs = ToSDataset.load_motifs(
    "data/motifs/hc3-mage_M3_motifs.json", selected_motif_hashes["m3"]
)
m6_motifs = ToSDataset.load_motifs(
    "data/motifs/hc3-mage_M6_motifs.json", selected_motif_hashes["m6"]
)
m9_motifs = ToSDataset.load_motifs(
    "data/motifs/hc3-mage_M9_motifs.json", selected_motif_hashes["m9"]
)


def add_motif_dist_to_document(document: Document) -> Document:
    for tree_idx, tree in document.scene_discourse_trees.items():
        graph = tree.graph_networkx
        root_label = f"span_1-{len(tree.edus)}"
        m3_dists = ToSDataset.calculate_motif_distribution(
            graph, m3_motifs, root_label=root_label
        )
        m6_dists = ToSDataset.calculate_motif_distribution(
            graph, m6_motifs, root_label=root_label
        )
        m9_dists = ToSDataset.calculate_motif_distribution(
            graph, m9_motifs, root_label=root_label
        )
        tree.motif_dists = {
            "m3": DiscourseMotifDists(**m3_dists),
            "m6": DiscourseMotifDists(**m6_dists),
            "m9": DiscourseMotifDists(**m9_dists),
        }
    return document


def sample_by_domains(dataset: List[Document], num_samples: int) -> List[Document]:
    domain_to_documents: Dict[str, List[Document]] = {}
    for document in dataset:
        domain = document.source.split("_")[0].strip()
        if domain not in domain_to_documents:
            domain_to_documents[domain] = []
        domain_to_documents[domain].append(document)

    sampled_dataset = []
    for domain, documents in domain_to_documents.items():
        random.shuffle(documents)
        sampled_dataset.extend(
            documents[: int(num_samples // len(domain_to_documents))]
        )

    random.shuffle(sampled_dataset)
    print(f"Sampled {len(sampled_dataset)} documents.")
    return sampled_dataset


if __name__ == "__main__":
    hc3_file_paths = glob("data/hc3/*.graph_added.jsonl")
    # mage_file_paths = glob("data/mage/*.graph_added.jsonl")
    mage_file_paths = glob(
        "data/mage/mage_train_machine.discourse_parsed.graph_added.jsonl"
    )

    # all_file_paths = hc3_file_paths + mage_file_paths
    all_file_paths = mage_file_paths

    for f_path in all_file_paths:
        print(f_path)
        dataset = ToSDataset.load_document_corpus(f_path)
        if "mage_train_machine.discourse_parsed.graph_added.jsonl" in f_path:
            # Since the machine training set contains an excessive number of samples,
            # we will limit its size to 33% more than the human samples (92581 * 1.33).
            dataset = sample_by_domains(dataset, 92581 * 1.33)

        dataset = process_map(
            add_motif_dist_to_document, dataset, max_workers=14, chunksize=1
        )
        ToSDataset.save_dataset_as_jsonl(dataset, f"{f_path[:-6]}.motif_dists.jsonl")
