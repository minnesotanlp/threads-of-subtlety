import os
import sys
from glob import glob

from tos.tos_dataset import Document, ToSDataset

assert len(sys.argv) == 3, "python tos_dataset.py total_gpus gpu_id"
total_gpus = int(sys.argv[1])
gpu_id = int(sys.argv[2])
assert gpu_id < total_gpus
tos_dataset = ToSDataset(
    # clone https://github.com/zaemyung/DMRST_Parser and download the parser model weights
    # following instructions: https://github.com/zaemyung/DMRST_Parser/tree/main/depth_mode/Savings#readme
    # and set the path to the cloned directory
    dmrst_parser_dir="/home/ubuntu/Development/DMRST_Parser",
    batch_size=1024,
    gpu_id=gpu_id,
)

os.makedirs("data/hc3", exist_ok=True)
os.makedirs("data/mage/tmp", exist_ok=True)
tos_dataset.process_hc3_dataset(processed_save_dir="data/hc3")
tos_dataset.process_mage_dataset(
    processed_save_dir="data/mage/tmp", total_gpus=total_gpus, gpu_id=gpu_id
)

hc3_files = glob("data/hc3/*.jsonl")
# sanity checking
for hc3_f in hc3_files:
    with open(hc3_f) as f:
        for line in f:
            line = line.strip()
            sample = eval(line)
            document = Document(**sample)
            num_scenes = len(document.scene_discourse_trees)
            for i in range(num_scenes):
                assert i in document.scene_discourse_trees.keys()

# merge the chunked files and save them separately as human and machine
splits = ["train", "validation", "test"]
for split in splits:
    mage_files = glob(f"data/mage/tmp/mage_{split}*.jsonl")
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
                    (orig, new)
                    for orig, new in zip(orig_indices, list(range(num_scenes)))
                ]
                for orig_i, new_i in orig_index_pairs:
                    document.scene_discourse_trees[new_i] = (
                        document.scene_discourse_trees.pop(orig_i)
                    )

                if document.label == 1:
                    human_samples.append(document)
                else:
                    machine_samples.append(document)

    processed_save_dir = "data/mage"
    human_save_path = os.path.join(
        processed_save_dir,
        f"mage_{split}_human.discourse_parsed.jsonl",
    )
    machine_save_path = os.path.join(
        processed_save_dir,
        f"mage_{split}_machine.discourse_parsed.jsonl",
    )

    with open(human_save_path, "w") as file:
        for document in human_samples:
            file.write(f"{document.model_dump(mode='json')}\n")

    with open(machine_save_path, "w") as file:
        for document in machine_samples:
            file.write(f"{document.model_dump(mode='json')}\n")
