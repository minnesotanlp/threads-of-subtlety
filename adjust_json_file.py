from tos.tos_dataset import Document, ToSDataset
from glob import glob
import os


def adjust_datasets(file_paths: str):
    for file_path in file_paths:
        print(file_path)
        dataset = []
        with open(file_path) as f:
            for line in f:
                sample = eval(line.strip())
                for scene_idx, tree in sample["scene_discourse_trees"].items():
                    tree["motif_dists"] = None
                document = Document(**sample)
                dataset.append(document)
        output_path = os.path.join(
            os.path.dirname(file_path), "new", os.path.basename(file_path)
        )
        with open(output_path, "w") as f:
            for document in dataset:
                f.write(f"{document.model_dump(mode='json')}\n")


if __name__ == "__main__":
    tos_dataset = ToSDataset(
        dmrst_parser_dir="/Users/zaemyung/Development/DMRST_Parser",
        batch_size=1024,
        gpu_id=None,
    )

    # hc3_file_paths = glob("data/hc3/*.graph_added.jsonl")
    # print(hc3_file_paths)
    # adjust_datasets(hc3_file_paths)

    mage_file_paths = glob("data/mage/*.graph_added.jsonl")
    print(mage_file_paths)
    adjust_datasets(mage_file_paths)
