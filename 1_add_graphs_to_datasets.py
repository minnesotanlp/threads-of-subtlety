import os
from glob import glob
import networkx as nx
from typing import Dict, Any, List

from tos.tos_dataset import Document, ToSDataset


def filter_empty_trees(sample: Dict[str, Any]) -> Dict[str, Any]:
    _scenes = []
    _scene_discourse_trees = {}
    for scene_idx, tree in sample["scene_discourse_trees"].items():
        if tree["parsed"] == "NONE" or tree["parsed"] == "":
            continue
        _scenes.append(sample["scenes"][int(scene_idx)])
        _scene_discourse_trees[scene_idx] = tree
    sample["scenes"] = _scenes
    sample["scene_discourse_trees"] = _scene_discourse_trees
    return sample


def add_graphs(files: List[str]) -> None:
    for f_path in files:
        print(f_path)
        dataset = []
        with open(f_path) as f:
            for line in f:
                line = line.strip()
                sample = eval(line)
                sample = filter_empty_trees(sample)
                if len(sample["scene_discourse_trees"]) < 1:
                    continue
                for tree_idx, tree in sample["scene_discourse_trees"].items():
                    G = ToSDataset.create_graph_from_const_format(tree["parsed"])
                    G_dict = nx.json_graph.node_link_data(G)
                    # G = nx.json_graph.node_link_graph(G_dict)
                    tree["graph_dict"] = G_dict
                    # only store the dict, networkx graph will be loaded when needed
                    tree["graph_networkx"] = None
                document = Document(**sample)
                dataset.append(document)
        with open(f"{f_path[:-6]}.graph_added.jsonl", "w") as f:
            for document in dataset:
                f.write(f"{document.model_dump(mode='json')}\n")


hc3_files = glob("data/hc3/*.jsonl")
add_graphs(hc3_files)

mage_files = glob("data/mage/*.jsonl")
add_graphs(mage_files)
