import os

import evaluate
import torch
from accelerate import Accelerator
from rich.progress import track
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LongformerForSequenceClassification

from tos.tos_dataset import LongformerDataCollator, LongformerDataset
from tos.tos_models import LongformerWithMotifsForSequenceClassification


def evaluate_longformer(testset_name: str, model_path: str, add_motif: bool):
    accelerator = Accelerator()
    accelerator.print(f"\n\n------{testset_name}-------")

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", use_fast=True
    )

    metric = evaluate.combine(
        ["accuracy", "f1", "precision", "recall", "BucketHeadP65/confusion_matrix"]
    )

    if add_motif:
        model = LongformerWithMotifsForSequenceClassification()
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        model = LongformerForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )

    test_dataset = LongformerDataset(
        split=testset_name, shuffle=False, saved_dir="data"
    )
    data_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=LongformerDataCollator(
            tokenizer=tokenizer, padding="longest", add_motif=add_motif
        ),
    )

    model, data_loader = accelerator.prepare(model, data_loader)

    for data in track(data_loader, total=len(data_loader), description="Evaluating..."):
        targets = data["labels"]
        predictions = model(**data).logits

        predictions = torch.argmax(predictions, dim=1)
        all_predictions, all_targets = accelerator.gather_for_metrics(
            (predictions, targets)
        )
        metric.add_batch(predictions=all_predictions, references=all_targets)

    accelerator.print(metric.evaluation_modules[0].__len__())
    accelerator.print(metric.compute())
    accelerator.print("-----------------------")


if __name__ == "__main__":
    # testset_names = ["hc3_test", "mage_test", "mage_ood_test", "mage_ood_para_test"]
    testset_names = ["mage_ood_test", "mage_ood_para_test"]
    for testset_name in testset_names:
        evaluate_longformer(
            testset_name,
            model_path="results/longformer_base_plain/checkpoint-800",
            # model_path="results/longformer_base_motif/checkpoint-1300",
            # add_motif=True,
            add_motif=False,
        )
