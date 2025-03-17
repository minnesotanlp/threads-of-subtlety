import random
from glob import glob

from tos.tos_dataset import ToSDataset

random.seed(42)


if __name__ == "__main__":
    hc3_file_paths = glob("data/hc3/hc3_*.graph_added.motif_dists.jsonl")

    train = []
    valid = []
    test = []
    VALID_SIZE_PER_DOMAIN = 200
    TEST_SIZE_PER_DOMAIN = 400
    for f_path in hc3_file_paths:
        print(f_path)
        dataset = ToSDataset.load_document_corpus(f_path)

        random.shuffle(dataset)

        valid.extend(dataset[:VALID_SIZE_PER_DOMAIN])
        test.extend(
            dataset[
                VALID_SIZE_PER_DOMAIN : VALID_SIZE_PER_DOMAIN + TEST_SIZE_PER_DOMAIN
            ]
        )
        train.extend(dataset[VALID_SIZE_PER_DOMAIN + TEST_SIZE_PER_DOMAIN :])

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    print(f"train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

    train_save_path = (
        "data/hc3/hc3_train.discourse_parsed.graph_added.motif_dists.jsonl"
    )
    ToSDataset.save_dataset_as_jsonl(train, train_save_path)

    valid_save_path = (
        "data/hc3/hc3_validation.discourse_parsed.graph_added.motif_dists.jsonl"
    )
    ToSDataset.save_dataset_as_jsonl(valid, valid_save_path)

    test_save_path = "data/hc3/hc3_test.discourse_parsed.graph_added.motif_dists.jsonl"
    ToSDataset.save_dataset_as_jsonl(test, test_save_path)
