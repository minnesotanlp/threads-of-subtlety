from tos_dataset import Document
from glob import glob
from pydantic_core import from_json
import os

import json

hc3_files = glob("/home/ubuntu/Development/threads-of-subtlety/data/hc3/*.jsonl")
mage_files = glob(
    "/home/ubuntu/Development/threads-of-subtlety/data/mage/mage_test*.jsonl"
)

# for hc3_f in hc3_files:
#     with open(hc3_f) as f:
#         for line in f:
#             line = line.strip()
#             sample = eval(line)
#             document = Document(**sample)
#             num_scenes = len(document.scene_discourse_trees)
#             for i in range(num_scenes):
#                 assert i in document.scene_discourse_trees.keys()

human_samples = []
machine_samples = []
for mage_f in mage_files:
    with open(mage_f) as f:
        for line in f:
            line = line.strip()
            sample = eval(line)
            document = Document(**sample)
            num_scenes = len(document.scene_discourse_trees)
            orig_indices = sorted(list(document.scene_discourse_trees.keys()))
            orig_index_pairs = [
                (orig, new) for orig, new in zip(orig_indices, list(range(num_scenes)))
            ]
            for orig_i, new_i in orig_index_pairs:
                document.scene_discourse_trees[new_i] = (
                    document.scene_discourse_trees.pop(orig_i)
                )

            if document.label == 1:
                human_samples.append(document)
            else:
                machine_samples.append(document)

processed_save_dir = "/home/ubuntu/Development/threads-of-subtlety/data/mage/proc"
human_save_path = os.path.join(
    processed_save_dir,
    "mage_test_human.discourse_parsed.jsonl",
)
machine_save_path = os.path.join(
    processed_save_dir,
    "mage_test_machine.discourse_parsed.jsonl",
)

with open(human_save_path, "w") as file:
    for document in human_samples:
        file.write(f"{document.model_dump(mode='json')}\n")

with open(machine_save_path, "w") as file:
    for document in machine_samples:
        file.write(f"{document.model_dump(mode='json')}\n")
