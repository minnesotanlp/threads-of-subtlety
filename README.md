# threads-of-subtlety
Code repository for the paper: "Threads of Subtlety: Detecting Machine-Generated Texts Through Discourse Motifs" (ACL 2024)

- Code: this repo.
- Models: [zaemyung/ToS-Longformer-Plain](https://huggingface.co/zaemyung/ToS-Longformer-Plain/tree/main), [zaemyung/ToS-Longformer-Motif](https://huggingface.co/zaemyung/ToS-Longformer-Motif/tree/main)
- Dataset: [zaemyung/ToS-Dataset](https://huggingface.co/datasets/zaemyung/ToS-Dataset/tree/main)

Some preprocessing updates were made, which may have contributed to performance improvements on the OOD and OOD-Para test sets.

|                  | HC3    | MAGE-Test | MAGE-OOD | MAGE-OOD-Para |
|------------------|--------|-----------|----------|---------------|
| Longformer-Plain | 96.74% | 88.64%    | 69.48%   | 55.95%        |
| Longformer-Motif | 97.42% | 91.83%    | 82.63%   | 76.15%        |
