# evaluation/download.py
from datasets import load_dataset

DATASETS = [
    "openlifescienceai/medqa",
    "openlifescienceai/medmcqa",
    "openlifescienceai/pubmedqa",
    "openlifescienceai/mmlu_clinical_knowledge",
    "openlifescienceai/mmlu_college_medicine",
    "openlifescienceai/mmlu_college_biology",
    "openlifescienceai/mmlu_professional_medicine",
    "openlifescienceai/mmlu_anatomy",
    "openlifescienceai/mmlu_medical_genetics",
]

for path in DATASETS:
    name = path.split("/")[1]
    print(f"\nDownloading {name}...")

    ds_dict = load_dataset(path)
    print(f"  Splits available: { {k: len(v) for k, v in ds_dict.items()} }")

    # Ưu tiên test → validation → train
    for split in ["test", "validation", "train"]:
        if split in ds_dict:
            chosen = ds_dict[split]
            print(f"  Using split: '{split}' ({len(chosen):,} samples)")
            break

    chosen.save_to_disk(f"./evaluation/test_qa_medrag/{name}")
    print(f"  → Saved to ./evaluation/test_qa_medrag/{name}")

print("\nAll done!")