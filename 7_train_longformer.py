import random
import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import PaddingStrategy

from tos.tos_dataset import Document, SceneDiscourseTree, ToSDataset
from tos.tos_utils import compute_metrics

random.seed(42)


@dataclass
class LongformerDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    add_motif: bool = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        text_data = []
        labels = []
        motif_dists = []

        for data in features:
            text_data.append(data["text"])
            labels.append(data["label"])
            if self.add_motif:
                motif_dists.append(data["motif_dists"])

        batch = self.tokenizer(
            text_data,
            padding=self.padding,
            return_tensors="pt",
            truncation=True,
        )
        batch["labels"] = torch.tensor(labels)
        if self.add_motif:
            batch["motif_dists"] = torch.tensor(
                np.stack(motif_dists), dtype=torch.float
            )

        return batch


class LongformerDataset(Dataset):
    def __init__(self, split: str, shuffle: bool, saved_dir: str):
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        dataset_path = os.path.join(saved_dir, f"longformer_dataset.{split}.pkl")
        if os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                self.dataset = pickle.load(f)
                print(f"Dataset loaded from {dataset_path}")
        else:
            self.create_dataset(split, dataset_path)

        if shuffle:
            random.shuffle(self.dataset)

    def create_dataset(self, split: str, save_path: str):
        if split == "train":
            dataset_paths = [
                "data/hc3/hc3_train.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_train_human.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_train_machine.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        elif split == "valid":
            dataset_paths = [
                "data/hc3/hc3_validation.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_validation_human.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_validation_machine.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        elif split == "test":
            dataset_paths = [
                "data/hc3/hc3_test.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_test_human.discourse_parsed.graph_added.motif_dists.jsonl",
                "data/mage/mage_test_machine.discourse_parsed.graph_added.motif_dists.jsonl",
            ]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.dataset = self.prepare_at_scene_level(
            ToSDataset.load_datasets(dataset_paths)
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Dataset saved at {save_path}")

    def prepare_at_scene_level(
        self, _dataset: List[Document]
    ) -> List[SceneDiscourseTree]:
        dataset = []
        for document in _dataset:
            label = document.label
            for scene in document.scene_discourse_trees.values():
                m3_mf = np.asarray(scene.motif_dists["m3"].mf)
                m3_wad = np.asarray(scene.motif_dists["m3"].wad)
                m6_mf = np.asarray(scene.motif_dists["m6"].mf)
                m6_wad = np.asarray(scene.motif_dists["m6"].wad)
                m9_mf = np.asarray(scene.motif_dists["m9"].mf)
                m9_wad = np.asarray(scene.motif_dists["m9"].wad)
                assert m3_mf.shape == m3_wad.shape
                assert m6_mf.shape == m6_wad.shape
                assert m9_mf.shape == m9_wad.shape
                m3_feats = np.zeros(m3_mf.shape[0] * 2, dtype=np.float32)
                m3_feats[::2] += m3_mf
                m3_feats[1::2] += m3_wad
                m6_feats = np.zeros(m6_mf.shape[0] * 2, dtype=np.float32)
                m6_feats[::2] += m6_mf
                m6_feats[1::2] += m6_wad
                m9_feats = np.zeros(m9_mf.shape[0] * 2, dtype=np.float32)
                m9_feats[::2] += m9_mf
                m9_feats[1::2] += m9_wad
                motif_dists = np.concatenate([m3_feats, m6_feats, m9_feats], axis=0)
                sample = {
                    "text": scene.text,
                    "label": label,
                    "motif_dists": motif_dists,
                }
                dataset.append(sample)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Document:
        return self.dataset[idx]


def train_longformer_plain(
    model_path: str = "allenai/longformer-base-4096", num_labels=2
):
    model = LongformerForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", use_fast=True
    )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = LongformerDataset(split="train", shuffle=True, saved_dir="data")
    print(f"train_dataset loaded with {len(train_dataset)} samples.")

    valid_dataset = LongformerDataset(split="valid", shuffle=True, saved_dir="data")
    print(f"valid_dataset loaded with {len(valid_dataset)} samples.")

    test_dataset = LongformerDataset(split="test", shuffle=True, saved_dir="data")
    print(f"test_dataset loaded with {len(test_dataset)} samples.")

    # model_name = "longformer_base_plain"
    # training_args = TrainingArguments(
    #     use_mps_device=True,
    #     # eval_steps=4000,
    #     # save_steps=4000,
    #     # logging_steps=200,
    #     # fp16=True,
    #     eval_steps=400,
    #     save_steps=400,
    #     logging_steps=20,
    #     metric_for_best_model="f1",
    #     greater_is_better=True,
    #     evaluation_strategy="steps",
    #     save_strategy="steps",
    #     gradient_accumulation_steps=2,
    #     learning_rate=5e-5,
    #     load_best_model_at_end=True,
    #     logging_dir=f"./models/{model_name}",  # directory for storing logs
    #     logging_strategy="steps",
    #     num_train_epochs=4,  # total number of training epochs
    #     output_dir=f"./results/{model_name}",  # output directory
    #     per_device_train_batch_size=128,
    #     per_device_eval_batch_size=128,
    #     remove_unused_columns=False,
    #     save_total_limit=3,
    #     weight_decay=4e-5,
    # )

    # trainer = Trainer(
    #     args=training_args,
    #     data_collator=LongformerDataCollator(
    #         tokenizer=tokenizer, padding="longest", add_motif=False
    #     ),
    #     eval_dataset=valid_dataset,
    #     model=model,
    #     train_dataset=train_dataset,
    #     compute_metrics=compute_metrics,
    # )

    # trainer.train()

    # return model


if __name__ == "__main__":
    train_longformer_plain()
    # train_longformer_motif()
