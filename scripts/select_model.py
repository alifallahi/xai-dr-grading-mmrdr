"""Step 3 - choose the final model using validation results only.

Reads train_summary.json of every fine-tuned run in <output_dir>/runs, ranks them by the
validation selection metric, writes the winner to <output_dir>/selected_run.txt and the
comparison table to <output_dir>/model_selection.csv. The test split is not read.

Once the selected run has been evaluated on the test set (evaluate.py), the selection is frozen:
re-running this script after training further runs only reports the validation ranking, so the
final model is never chosen after test results have been seen. --reselect overrides this.

Usage:  python scripts/select_model.py --config configs/default.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai.config import base_parser, load_config  # noqa: E402
from drxai.utils import load_json, save_json  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--reselect", action="store_true",
                        help="change the selection even though the selected run was already evaluated on the test "
                             "set (breaks the test-once protocol; not recommended)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg["paths"]["output_dir"])
    marker = out / "selected_run.txt"
    current = marker.read_text().strip() if marker.exists() else None
    frozen = current is not None and (out / "evaluation" / current / "metrics.json").exists()
    rows = []
    for summary in sorted((out / "runs").glob("*/train_summary.json")):
        s = load_json(summary)
        m = s["best_val_metrics"]
        rows.append({
            "run": s["run"],
            "fine_tuned": (summary.parent / "best.pt").exists(),
            "replicate_of": s.get("replicate_of"),
            "lr": s.get("lr"),
            "class_weighting": s.get("class_weighting"),
            "seed": s.get("seed"),
            "val_qwk": m.get("val_qwk", m.get("qwk")),
            "val_macro_f1": m.get("val_macro_f1", m.get("macro_f1")),
            "val_accuracy": m.get("val_accuracy", m.get("accuracy")),
            "best_epoch": s.get("best_epoch"),
            "minutes": s.get("minutes"),
        })
    if not rows:
        raise SystemExit("No runs found; train a model first.")
    metric = "val_" + cfg["train"].get("selection_metric", "qwk")
    table = pd.DataFrame(rows).sort_values(metric, ascending=False)
    # Seed replicates (train_seeds.py) estimate variability; they are never selection candidates.
    candidates = table[table["fine_tuned"] & table["replicate_of"].isna()]
    if candidates.empty:
        raise SystemExit("No fine-tuned run with best.pt found.")
    best_on_val = candidates.iloc[0]["run"]

    if frozen and not args.reselect:
        # The selected model has already been scored on the test set. Letting a run trained afterwards
        # replace it would mean choosing the final model after seeing test results, so the selection
        # stays fixed and later runs are reported as secondary comparisons.
        winner = current
        note = (f"selection frozen: '{current}' was evaluated on the test set before the other runs were "
                f"compared; best on validation now = '{best_on_val}'")
    else:
        winner = best_on_val
        note = f"selected '{winner}' by {metric} on the validation split"
        marker.write_text(winner + "\n")
    table["selected"] = table["run"] == winner
    table.to_csv(out / "model_selection.csv", index=False)
    save_json({"selected": winner, "best_on_validation": best_on_val, "frozen": bool(frozen and not args.reselect),
               "metric": metric, "note": note}, out / "selection_log.json")
    print(table.to_string(index=False))
    print(f"[select] {note} -> {marker}")


if __name__ == "__main__":
    main()
