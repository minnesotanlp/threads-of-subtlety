import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import regex as re
from datasets import load_dataset
from pydantic import BaseModel
from rich.progress import track
from sentsplit.segment import SentSplit
from transformers import AutoTokenizer

from .tos_utils import split_list_into_n_chunks

random.seed(42)


class DiscourseMotifDists(BaseModel):
    raw: List[float]
    mf: List[float]
    wad: List[float]


class SceneDiscourseTree(BaseModel):
    text: str
    tokenized: List[str]
    segments: List[int]
    edus: Dict[str, str]
    parsed: str
    graph_dict: Optional[Dict[str, Any]]
    graph_networkx: Optional[Any]
    motif_dists: Optional[Dict[str, DiscourseMotifDists]]


class Document(BaseModel):
    text: str
    scenes: List[str]
    scene_discourse_trees: Optional[Dict[int, SceneDiscourseTree]]
    source: Optional[str]
    label: Optional[int]


class ToSDataset:
    def __init__(self, dmrst_parser_dir: str, batch_size: int = 64, gpu_id: int = None):
        assert os.path.isdir(dmrst_parser_dir)
        sys.path.append(dmrst_parser_dir)
        from MUL_main_Infer import DiscourseParser

        self.batch_size = batch_size
        self.gpu_id = gpu_id

        parser_model_path = os.path.join(
            dmrst_parser_dir, "depth_mode/Savings/multi_all_checkpoint.torchsave"
        )
        self.discourse_parser = DiscourseParser(
            model_path=parser_model_path,
            batch_size=self.batch_size,
            device=f"cuda:{gpu_id}" if gpu_id is not None else None,
        )

        self.discourse_tokenizer = AutoTokenizer.from_pretrained(
            "xlm-roberta-base", use_fast=True
        )
        self.max_tokens = self.discourse_tokenizer.model_max_length - 20
        self.sent_splitter = SentSplit("en")

    def parse_scenes_discourse(
        self,
        scenes: List[str],
        filter_none: bool = False,
        scene_indices: List[int] = None,
        disable_progressbar: bool = True,
    ) -> Dict[int, SceneDiscourseTree]:
        """Parse the discourse structure of a list of scenes.

        Args:
            scenes (List[str]): A list of scenes to parse.

        Returns:
            Dict[int, SceneDiscourseTree]: A dictionary of parsed discourse trees.
        """
        assert isinstance(scenes, list)
        tokens, segments, parsed = self.discourse_parser.parse(
            scenes, disable_progressbar=disable_progressbar
        )
        assert len(tokens) == len(segments) == len(parsed) == len(scenes)

        if scene_indices is None:
            scene_indices = range(len(scenes))
        else:
            assert len(scene_indices) == len(scenes)
        scene_discourse_trees = {}
        for i, scene in zip(scene_indices, scenes):
            if filter_none and parsed[i][0] == "NONE":
                print(f"{i}-th sample is filtered.")
                scene_discourse_trees[i] = None
                continue
            edus = {}
            last_end = 0
            for edu_i, edu_end in enumerate(segments[i], start=1):
                edu = "".join(tokens[i][last_end : edu_end + 1]).replace("▁", " ")
                edus[f"span_{edu_i}-{edu_i}"] = edu
                last_end = edu_end + 1

            scene_discourse_trees[i] = SceneDiscourseTree(
                text=scene,
                tokenized=tokens[i],
                segments=segments[i],
                edus=edus,
                parsed=parsed[i][0],
                graph_dict=None,
            )
        return scene_discourse_trees

    def parse_document_discourse(
        self, document: str, filter_none: bool = False
    ) -> Document:
        """Parse the discourse structure of a document.

        Args:
            document (str): The document to parse.
            filter_none (bool, optional): Whether to filter out the samples with no discourse structure.
                                          Defaults to False.

        Returns:
            Document: The parsed document (Dataclass).
        """
        scenes = self.split_document(document)
        assert isinstance(scenes, list)

        scene_discourse_trees = self.parse_scenes_discourse(
            scenes, filter_none=filter_none
        )
        document = Document(
            text=document,
            scenes=scenes,
            scene_discourse_trees=scene_discourse_trees,
            source=None,
            label=None,
        )
        return document

    def split_document(self, document: str) -> List[str]:
        """Split a document into scenes based on the token limit.

        Args:
            document (str): The document to split.

        Returns:
            List[str]: A list of scenes.
        """
        paragraphs = document.split("\n\n")
        scenes = []
        current_scene = ""
        current_token_count = 0

        for paragraph in paragraphs:
            token_count = len(self.discourse_tokenizer.tokenize(paragraph))

            if token_count > self.max_tokens:
                # if the paragraph itself exceeds the token limit,
                # split the paragraph into sentences and continue
                sentences = self.sent_splitter.segment(paragraph, strip_spaces=False)
                for sentence in sentences:
                    token_count = len(self.discourse_tokenizer.tokenize(sentence))
                    if current_token_count + token_count > self.max_tokens:
                        scenes.append(current_scene.strip())
                        current_scene = sentence
                        current_token_count = token_count
                    else:
                        current_scene += sentence
                        current_token_count += token_count
                if current_scene:
                    scenes.append(current_scene)
                    current_scene = ""
                    current_token_count = 0
                continue

            if current_token_count + token_count > self.max_tokens:
                scenes.append(current_scene.strip())
                current_scene = paragraph
                current_token_count = token_count
            else:
                if current_scene:
                    current_scene += "\n\n"
                current_scene += paragraph
                current_token_count += token_count

        if current_scene.strip():
            scenes.append(current_scene.strip())

        return scenes

    def process_hc3_dataset(
        self,
        processed_save_dir: str,
        dataset_name: str = "Hello-SimpleAI/HC3",
        min_char_len: int = 10,
    ):
        """Process the HC3 dataset and save the parsed discourse trees.

        Args:
            processed_save_dir (str): The directory to save the processed dataset.
            dataset_name (str, optional): The name of the dataset. Defaults to "Hello-SimpleAI/HC3".
            min_char_len (int, optional): The minimum character length of the samples to consider. Defaults to 10.
        """
        raw_dataset = load_dataset(dataset_name, name="all")["train"]

        dataset = defaultdict(list)
        sources = set()
        for sample in raw_dataset:
            del sample["id"]
            try:
                human_answer = sample["human_answers"][0].strip()
                llm_answer = sample["chatgpt_answers"][0].strip()
            except IndexError:
                continue

            if len(human_answer) < min_char_len or len(llm_answer) < min_char_len:
                continue

            source = sample["source"]
            sources.add(source)

            human_sample = {
                "text": human_answer,
                "source": f"{source}_human",
                "label": 1,
            }
            llm_sample = {"text": llm_answer, "source": f"{source}_llm", "label": 0}
            dataset[source].append(human_sample)
            dataset[source].append(llm_sample)

        for source in sources:
            print(source, len(dataset[source]) / 2)
            for i, sample in track(
                enumerate(dataset[source]),
                description=f"parsing discourse for {source} samples..",
                total=len(dataset[source]),
            ):
                document = self.parse_document_discourse(sample["text"])
                document.source = sample["source"]
                document.label = sample["label"]
                dataset[source][i] = document

            save_path = os.path.join(
                processed_save_dir, f"hc3_{source}.discourse_parsed.jsonl"
            )
            with open(save_path, "w") as file:
                for document in dataset[source]:
                    file.write(f"{document.model_dump(mode='json')}\n")

    def process_mage_dataset(
        self,
        processed_save_dir: str,
        dataset_name: str = "yaful/MAGE",
        total_gpus: int = 1,
        gpu_id: int = 0,
    ):
        """Process the MAGE dataset and save the parsed discourse trees.

        Args:
            processed_save_dir (str): The directory to save the processed dataset.
            dataset_name (str, optional): The name of the dataset. Defaults to "yaful/MAGE".
            total_gpus (int, optional): The total number of GPUs to split the dataset. Defaults to None.
            gpu_id (int, optional): The GPU ID to process the dataset. Defaults to None.
        """
        raw_dataset = load_dataset(dataset_name)

        for split in ["test", "validation", "train"]:
            # As the dataset is large, we split it into chunks so that each GPU can process a chunk
            chunked_indices = list(
                split_list_into_n_chunks(range(len(raw_dataset[split])), total_gpus)
            )
            target_indices = chunked_indices[gpu_id]

            all_scenes = []
            document_index_to_all_scene_indices = {}
            for doc_idx in target_indices:
                sample = raw_dataset[split][doc_idx]
                scenes = self.split_document(sample["text"])
                start_scene_index = len(all_scenes)
                all_scenes.extend(scenes)
                end_scene_index = start_scene_index + len(scenes)
                document_index_to_all_scene_indices[doc_idx] = (
                    start_scene_index,
                    end_scene_index,
                )
            print(
                f"Number of documents in {split}, chunk: {gpu_id}: {len(document_index_to_all_scene_indices)}"
            )
            print(f"Number of scenes in {split}: {len(all_scenes)}")

            all_scene_discourse_trees = self.parse_scenes_discourse(
                all_scenes, filter_none=False, disable_progressbar=False
            )
            assert len(all_scene_discourse_trees) == len(all_scenes)

            human_dataset = []
            machine_dataset = []
            for doc_idx, (start, end) in document_index_to_all_scene_indices.items():
                document = Document(
                    text=raw_dataset[split][doc_idx]["text"],
                    scenes=all_scenes[start:end],
                    scene_discourse_trees={
                        j - start: all_scene_discourse_trees[j]
                        for j in range(start, end)
                    },
                    source=raw_dataset[split][doc_idx]["src"],
                    label=raw_dataset[split][doc_idx]["label"],
                )
                if document.label == 1:
                    human_dataset.append(document)
                else:
                    machine_dataset.append(document)

            human_save_path = os.path.join(
                processed_save_dir,
                f"mage_{split}_{gpu_id}_human.discourse_parsed.jsonl",
            )
            with open(human_save_path, "w") as file:
                for document in human_dataset:
                    file.write(f"{document.model_dump(mode='json')}\n")

            machine_save_path = os.path.join(
                processed_save_dir,
                f"mage_{split}_{gpu_id}_machine.discourse_parsed.jsonl",
            )
            with open(machine_save_path, "w") as file:
                for document in machine_dataset:
                    file.write(f"{document.model_dump(mode='json')}\n")

    @staticmethod
    def create_graph_from_const_format(format_string: str) -> nx.DiGraph:
        """Create a graph from the constituency format string used in the DMRST parser.

        Args:
            format_string (str): The constituency format string.

        Returns:
            nx.DiGraph: The graph representation of the constituency format.
        """
        spans = format_string.strip().split(" ")
        rgx_span = r"\((\d+):(.+)=(.+):(\d+),(\d+):(.+)=(.+):(\d+)\)"
        edges = []
        nodes_types = {}
        edu_indices = set()
        for i, span in enumerate(spans, start=1):
            m_span = re.match(rgx_span, span)
            assert m_span is not None
            left_most_edu_index = int(m_span.group(1))
            right_most_edu_index = int(m_span.group(8))

            left_type = m_span.group(2)
            left_relation = m_span.group(3)
            left_end_edu_index = int(m_span.group(4))

            right_start_edu_index = int(m_span.group(5))
            right_type = m_span.group(6)
            right_relation = m_span.group(7)

            node_label = f"span_{left_most_edu_index}-{right_most_edu_index}"
            left_node_label = f"span_{left_most_edu_index}-{left_end_edu_index}"
            right_node_label = f"span_{right_start_edu_index}-{right_most_edu_index}"

            edu_indices.add(left_most_edu_index)
            edu_indices.add(right_most_edu_index)
            edu_indices.add(left_end_edu_index)
            edu_indices.add(right_start_edu_index)

            # hyperedge
            edges.append((left_node_label, node_label, "/"))
            edges.append((right_node_label, node_label, "/"))

            left_node_type = left_type
            right_node_type = right_type

            if left_relation != "span":
                edges.append((left_node_label, right_node_label, left_relation))
            if right_relation != "span":
                edges.append((right_node_label, left_node_label, right_relation))

            if i == 1:
                root_node_label = node_label
                nodes_types[root_node_label] = "root"

            nodes_types[left_node_label] = left_node_type
            nodes_types[right_node_label] = right_node_type

        G = nx.DiGraph()
        for u, v, label in edges:
            G.add_edge(u, v, label_0=label)

        nx.set_node_attributes(G, nodes_types, "label_0")
        assert root_node_label == f"span_1-{max(edu_indices)}"
        return G

    @staticmethod
    def calculate_motif_distribution(
        G: nx.DiGraph, graph_motifs: List[nx.DiGraph], root_label: str
    ) -> Dict[str, np.ndarray]:
        G_diameter = nx.diameter(G.to_undirected())
        hist = np.zeros(len(graph_motifs), dtype=float)
        wad = np.zeros(len(graph_motifs), dtype=float)

        for index, motif in enumerate(graph_motifs):
            if motif.number_of_nodes() == 1:
                hist[index] = G.number_of_nodes()
                continue
            if motif.number_of_nodes() == 2:
                hist[index] = G.number_of_edges()
                continue

            DiGM = nx.algorithms.isomorphism.DiGraphMatcher(
                G, motif, edge_match=lambda e1, e2: e1["label_0"] == e2["label_0"]
            )

            counts_per_depth = {}

            for subgraph in DiGM.subgraph_isomorphisms_iter():
                # subgraph e.g.: {'span_1-31': 'span_21-24', 'span_29-31': 'span_23-24', 'span_31-31': 'span_24-24'}, <dict>
                motif_nodes = subgraph.keys()
                motif_depth = np.mean(
                    [
                        nx.shortest_path_length(
                            G.to_undirected(), source=root_label, target=node_label
                        )
                        for node_label in motif_nodes
                    ]
                )
                if motif_depth not in counts_per_depth:
                    counts_per_depth[motif_depth] = 1
                else:
                    counts_per_depth[motif_depth] += 1

                hist[index] += 1

            counts_x_depths = np.sum(
                [depth * counts for depth, counts in counts_per_depth.items()]
            )

            # sum(depth x count) / sum(count)
            wad[index] = counts_x_depths / hist[index] if hist[index] > 0 else -1

        num_of_motifs = np.sum(hist)
        motif_freqs = hist / num_of_motifs if num_of_motifs > 0 else hist
        wad = wad / G_diameter
        # -1 means that the motif does not exist in the graph
        wad[wad < 0] = -1
        return {"raw": hist.tolist(), "mf": motif_freqs.tolist(), "wad": wad.tolist()}

    @staticmethod
    def load_document_corpus(file_path: str) -> List[Document]:
        """Load a document corpus from a file, where each line is a JSON representation of a Document.

        Args:
            file_path (str): The file path to load the document corpus, e.g., "data/hc3/hc3_reddit_eli5.discourse_parsed.graph_added.jsonl".

        Returns:
            List[Document]: A list of Document objects.
        """
        dataset = []
        with open(file_path) as f:
            for line in f:
                sample = eval(line.strip())
                document = Document(**sample)
                for tree_idx, tree in document.scene_discourse_trees.items():
                    G = nx.json_graph.node_link_graph(tree.graph_dict)
                    tree.graph_networkx = G
                dataset.append(document)
        return dataset

    @staticmethod
    def load_datasets(file_paths: str, max_per_file: int = None) -> List[Document]:
        dataset = []
        for file_path in file_paths:
            print(file_path)
            samples = ToSDataset.load_document_corpus(file_path)
            print(f"loaded: {len(samples)} samples")
            if max_per_file is not None:
                random.shuffle(samples)
                samples = samples[:max_per_file]
            dataset.extend(samples)
        print(f"all loaded: {len(dataset)} samples")
        return dataset

    @staticmethod
    def load_motifs(
        motif_path: str, selected_hashes: List[str] = None
    ) -> List[nx.DiGraph]:
        motif_graphs = []
        with open(motif_path, "r") as f:
            _motifs = json.load(f)
            if selected_hashes:
                motif_graphs.extend(
                    [
                        nx.json_graph.node_link_graph(_motifs[hash])
                        for hash in selected_hashes
                    ]
                )
            else:
                motif_graphs.extend(
                    [nx.json_graph.node_link_graph(v) for v in _motifs.values()]
                )
        print(f"loaded motif graphs: {len(motif_graphs)}")
        return motif_graphs

    @staticmethod
    def save_dataset_as_jsonl(dataset: List[Document], f_path: str):
        with open(f_path, "w") as f:
            for document in dataset:
                for tree in document.scene_discourse_trees.values():
                    tree.graph_networkx = None
                f.write(f"{document.model_dump(mode='json')}\n")
