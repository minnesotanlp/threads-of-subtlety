import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from pydantic import BaseModel
from rich.progress import track
from sentsplit.segment import SentSplit
from transformers import AutoTokenizer


class SceneDiscourseTree(BaseModel):
    text: str
    tokenized: List[str]
    segments: List[int]
    edus: Dict[str, str]
    parsed: str


class Document(BaseModel):
    text: str
    scenes: List[str]
    scene_discourse_trees: Optional[Dict[int, SceneDiscourseTree]]
    source: Optional[str]
    label: Optional[int]


class ToSDataset:
    def __init__(self, dmrst_parser_dir: str, batch_size: int = 64):
        assert os.path.isdir(dmrst_parser_dir)
        sys.path.append(dmrst_parser_dir)
        from MUL_main_Infer import DiscourseParser

        parser_model_path = os.path.join(
            dmrst_parser_dir, "depth_mode/Savings/multi_all_checkpoint.torchsave"
        )
        self.discourse_parser = DiscourseParser(model_path=parser_model_path)
        self.discourse_tokenizer = AutoTokenizer.from_pretrained(
            "xlm-roberta-base", use_fast=True
        )
        self.batch_size = batch_size
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
            scenes, batch_size=self.batch_size, disable_progressbar=disable_progressbar
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
        min_char_len: int = 10,
    ):
        raw_dataset = load_dataset(dataset_name)

        for split in ["test", "validation", "train"]:
            all_scenes = []
            document_index_to_all_scene_indices = {}
            for i, sample in enumerate(raw_dataset[split]):
                if len(sample["text"]) < min_char_len:
                    continue

                scenes = self.split_document(sample["text"])
                start_scene_index = len(all_scenes)
                all_scenes.extend(scenes)
                end_scene_index = start_scene_index + len(scenes)
                document_index_to_all_scene_indices[i] = (
                    start_scene_index,
                    end_scene_index,
                )

            print(f"Number of scenes in {split}: {len(all_scenes)}")

            all_scene_discourse_trees = self.parse_scenes_discourse(
                all_scenes, filter_none=False, disable_progressbar=False
            )
            assert len(all_scene_discourse_trees) == len(all_scenes)

            human_dataset = []
            machine_dataset = []
            for document_index, (
                start,
                end,
            ) in document_index_to_all_scene_indices.items():
                document = Document(
                    text=raw_dataset[split][document_index]["text"],
                    scenes=all_scenes[start:end],
                    scene_discourse_trees={
                        j: all_scene_discourse_trees[j] for j in range(start, end)
                    },
                    source=raw_dataset[split][document_index]["src"],
                    label=raw_dataset[split][document_index]["label"],
                )
                if document.label == 1:
                    human_dataset.append(document)
                else:
                    machine_dataset.append(document)

            human_save_path = os.path.join(
                processed_save_dir, f"mage_{split}_human.discourse_parsed.jsonl"
            )
            with open(human_save_path, "w") as file:
                for document in human_dataset:
                    file.write(f"{document.model_dump(mode='json')}\n")

            machine_save_path = os.path.join(
                processed_save_dir, f"mage_{split}_machine.discourse_parsed.jsonl"
            )
            with open(machine_save_path, "w") as file:
                for document in machine_dataset:
                    file.write(f"{document.model_dump(mode='json')}\n")


if __name__ == "__main__":
    tos_dataset = ToSDataset(
        dmrst_parser_dir="/Users/zaemyung/Development/DMRST_Parser"
    )

    # with open(
    #     # "/Users/zaemyung/Development/threads-of-subtlety/tos/long_doc.txt",
    #     "/Users/zaemyung/Development/threads-of-subtlety/tos/short_doc.txt",
    #     "r",
    # ) as file:
    #     long_text = file.read()
    # print(tos_dataset.parse_document_discourse(long_text))

    # tos_dataset.process_hc3_dataset(
    #     processed_save_dir="/Users/zaemyung/Development/threads-of-subtlety/data/hc3",
    # )

    tos_dataset.process_mage_dataset(
        processed_save_dir="/Users/zaemyung/Development/threads-of-subtlety/data/mage",
    )

    # results = tos_dataset.parse_discourse(
    #     [
    #         "The Transformer architecture has been a major component in the success of Large Language Models (LLMs). It has been used for nearly all LLMs that are being used today, from open-source models like Mistral to closed-source models like ChatGPT."
    #     ]
    # )
    # print(results)

    # process_hc3_dataset()

    # # 0 for machine-generated; 1 for human-written
    # all_dataset = load_dataset("yaful/MAGE")
    # all_dataset = load_dataset("Hello-SimpleAI/HC3")

    # print(all_dataset)

    # splits = ["train", "validation", "test"]

    # for split in splits:
    #     for sample in all_dataset[split]:
    #         pass

    # discourse_added_save_path = os.path.join(
    #     SAVE_DIR, f"DeepfakeTextDetect.{split}.discourse_added.jsonl"
    # )
    # dataset = add_discourse_parsed_result(
    #     dataset, output_path=discourse_added_save_path
    # )
    # print(len(dataset))

    # networkx_added_save_path = os.path.join(
    #     SAVE_DIR, f"DeepfakeTextDetect.{split}.discourse_added.networkx_added.pkl"
    # )
    # dataset = add_networkx_graphs(
    #     dataset=discourse_added_save_path, output_path=networkx_added_save_path
    # )
    # print(len(dataset))
