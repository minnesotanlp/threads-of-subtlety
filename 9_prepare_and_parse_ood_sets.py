import random

import pandas as pd
from rich.progress import track

from tos.tos_dataset import ToSDataset

random.seed(42)


# Download these directly from https://huggingface.co/datasets/yaful/MAGE/tree/main
dataset_paths = [
    "data/mage/test_ood_set_gpt.csv",
    "data/mage/test_ood_set_gpt_para.csv",
]

df_ood = pd.read_csv(dataset_paths[0], header=0)
df_ood_para = pd.read_csv(dataset_paths[1], header=0)

print(df_ood.head())
print(df_ood_para.head())

tos_dataset = ToSDataset(
    dmrst_parser_dir="/home/ubuntu/Development/DMRST_Parser",
    batch_size=1024,
    gpu_id=0,
    motif_dir="data/motifs",
)


def process_df(df):
    dataset = []
    for i, row in track(df.iterrows(), total=len(df), description="Processing..."):
        text = row["text"]
        label = row["label"]
        source = row["src"]
        document = tos_dataset.parse_document_discourse(
            document=text,
            source=source,
            label=label,
            filter_none=True,
            add_graph=True,
            add_motif_dists=True,
        )
        dataset.append(document)
    return dataset


dataset_ood = process_df(df_ood)
tos_dataset.save_dataset_as_jsonl(
    dataset_ood,
    "data/mage/test_ood_set_gpt.discourse_parsed.graph_added.motif_dists.jsonl",
)

dataset_ood_para = process_df(df_ood_para)
tos_dataset.save_dataset_as_jsonl(
    dataset_ood_para,
    "data/mage/test_ood_set_gpt_para.discourse_parsed.graph_added.motif_dists.jsonl",
)
