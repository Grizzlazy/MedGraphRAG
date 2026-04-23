from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    base_dir: Path


RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def iter_run_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    # Run folders are like 20260423_093146
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and RUN_DIR_RE.match(p.name)]
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs (timestamp folders) under: {base_dir}")
    return run_dirs


def newest_run_dir(base_dir: Path) -> Path:
    run_dirs = iter_run_dirs(base_dir)

    # Timestamp folders are like 20260423_093146; lexicographic works if consistent.
    run_dirs.sort(key=lambda p: p.name, reverse=True)
    return run_dirs[0]


def read_summary_csv(summary_csv: Path) -> Dict[str, float]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary: {summary_csv}")

    out: Dict[str, float] = {}
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_cols = {"dataset", "accuracy(%)"}
        if not expected_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Unexpected columns in {summary_csv}. "
                f"Got {reader.fieldnames}, expected at least {sorted(expected_cols)}"
            )

        for row in reader:
            ds = (row.get("dataset") or "").strip()
            acc = (row.get("accuracy(%)") or "").strip()
            if not ds:
                continue
            try:
                out[ds] = float(acc)
            except ValueError:
                # Keep as missing if parse fails
                pass
    return out


def format_table(rows: List[List[str]]) -> str:
    # simple fixed-width table
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, r in enumerate(rows):
        line = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))
        lines.append(line)
        if idx == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def collect_run_artifacts(spec: ModelSpec, run_dir: Path, dest_root: Path) -> Path:
    """
    Copy chosen run artifacts into dest_root for provenance.
    Copies all rag_*.csv under the chosen run directory.
    """
    model_dir = dest_root / safe_slug(spec.name)
    out_dir = model_dir / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy all rag_*.csv files (summary + per-dataset outputs)
    for p in sorted(run_dir.glob("rag_*.csv")):
        shutil.copy2(p, out_dir / p.name)

    return out_dir


def pick_run_dir(base_dir: Path, pick: str) -> Path:
    """
    pick:
      - latest: newest timestamp folder
      - worst: (sweep) worst per-dataset across all runs (non-zero)
      - best: (sweep) best per-dataset across all runs
    """
    if pick == "latest":
        return newest_run_dir(base_dir)

    raise ValueError("pick_run_dir no longer supports best/worst as single-run selection.")


def sweep_best_worst(
    base_dir: Path,
    mode: str,
    ignore_zero: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Path]]:
    """
    Sweep all runs and select best/worst per dataset.
    Returns:
      - dataset -> selected accuracy(%)
      - dataset -> source run_dir (where that value came from)
    """
    if mode not in {"best", "worst"}:
        raise ValueError(f"Unknown sweep mode: {mode}")

    chosen_acc: Dict[str, float] = {}
    chosen_run: Dict[str, Path] = {}

    for run_dir in iter_run_dirs(base_dir):
        summary = run_dir / "rag_summary.csv"
        try:
            data = read_summary_csv(summary)
        except Exception:
            continue

        for ds, acc in data.items():
            if ignore_zero and acc <= 0.0:
                continue
            if ds not in chosen_acc:
                chosen_acc[ds] = acc
                chosen_run[ds] = run_dir
                continue

            if mode == "best" and acc > chosen_acc[ds]:
                chosen_acc[ds] = acc
                chosen_run[ds] = run_dir
            elif mode == "worst" and acc < chosen_acc[ds]:
                chosen_acc[ds] = acc
                chosen_run[ds] = run_dir

    if not chosen_acc:
        raise FileNotFoundError(f"No usable rag_summary.csv found under: {base_dir}")
    return chosen_acc, chosen_run


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    default_specs = [
        ModelSpec("Qwen2.5-7B-Instruct", repo_root / "rag_eval_results"),
        ModelSpec("Llama3-3B-Instruct", repo_root / "rag_eval_results_llama32_3B"),
        ModelSpec("Llama3-8B-Instruct", repo_root / "rag_eval_results_llama3_8B"),
    ]

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=str, default="", help="Optional output CSV path")
    ap.add_argument(
        "--collect-dir",
        type=str,
        default="",
        help="Optional directory to copy chosen run artifacts into (e.g. results/rag)",
    )
    ap.add_argument(
        "--use-run",
        type=str,
        default="latest",
        help="Run folder name (e.g. 20260423_093146) or one of: latest, worst, best",
    )
    args = ap.parse_args()

    model_results: List[Tuple[ModelSpec, str, Dict[str, float]]] = []
    all_datasets: set[str] = set()

    collected: List[Tuple[str, Path]] = []
    collect_root = (repo_root / args.collect_dir) if args.collect_dir else None

    for spec in default_specs:
        if args.use_run == "latest":
            run_dir = newest_run_dir(spec.base_dir)
            data = read_summary_csv(run_dir / "rag_summary.csv")
            source_label = run_dir.name
            if collect_root is not None:
                out_dir = collect_run_artifacts(spec, run_dir, collect_root)
                collected.append((spec.name, out_dir))
        elif args.use_run in {"best", "worst"}:
            data, ds_to_run = sweep_best_worst(spec.base_dir, args.use_run, ignore_zero=(args.use_run == "worst"))
            source_label = f"sweep_{args.use_run}"
            if collect_root is not None:
                # Collect unique run dirs that contributed to the sweep results.
                for run_dir in sorted(set(ds_to_run.values()), key=lambda p: p.name):
                    out_dir = collect_run_artifacts(spec, run_dir, collect_root)
                    collected.append((spec.name, out_dir))
        else:
            run_dir = spec.base_dir / args.use_run
            data = read_summary_csv(run_dir / "rag_summary.csv")
            source_label = run_dir.name
            if collect_root is not None:
                out_dir = collect_run_artifacts(spec, run_dir, collect_root)
                collected.append((spec.name, out_dir))

        all_datasets.update(data.keys())
        model_results.append((spec, source_label, data))

    datasets = sorted(all_datasets)

    header = ["dataset"] + [spec.name for (spec, _, _) in model_results]
    rows: List[List[str]] = [header]

    for ds in datasets:
        row = [ds]
        for (_, _, data) in model_results:
            row.append("" if ds not in data else f"{data[ds]:.2f}")
        rows.append(row)

    # Print chosen runs
    print("Chosen runs:")
    for (spec, source_label, _) in model_results:
        print(f"- {spec.name}: {source_label}")

    print("\nSummary (accuracy %):")
    print(format_table(rows))

    if collected:
        print("\nCollected artifacts:")
        for name, out_dir in collected:
            print(f"- {name}: {out_dir}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()