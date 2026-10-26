"""Step 7 - gather every number and figure the report needs into one place.

Writes <output_dir>/report/tables.md (markdown tables, ready to paste) and copies the key
figures to <output_dir>/report/figures/. Missing pieces (e.g. a step not yet run) are skipped.

Usage:  python scripts/collect_results.py --config configs/default.yaml
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai.config import base_parser, load_config  # noqa: E402
from drxai.data.mmrdr import GRADE_SHORT  # noqa: E402
from drxai.utils import df_to_markdown, load_json, selected_run_name  # noqa: E402


def ci(v) -> str:
    if isinstance(v, str):
        v = [float(x) for x in v.strip("[]").split(",")]
    return f"{v[0]:.3f}-{v[1]:.3f}" if isinstance(v, (list, tuple)) and len(v) == 2 else ""


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    root = Path(cfg["paths"]["output_dir"])
    prepared = Path(cfg["paths"]["prepared_dir"])
    run = selected_run_name(cfg)
    rep = root / "report"
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    md: list[str] = [f"# Results tables (selected run: `{run}`)", ""]

    # ------------------------------------------------------------------ data audit
    audit_path = prepared / "audit" / "audit.json"
    if audit_path.exists():
        a = load_json(audit_path)
        md += ["## Data", "", "Images per grade by split (model splits):", "",
               df_to_markdown(pd.DataFrame(a["class_table_model_splits"]).rename(index=lambda i: GRADE_SHORT[int(i)] if str(i).isdigit() else i)), "",
               f"- Grade counts match paper Table 2: {a['paper_check']['grade_counts_match_paper']}; "
               f"lesion counts match: {a['paper_check']['lesion_counts_match_paper']}",
               f"- Imbalance (largest/smallest class, train): {a['imbalance_ratio_train']:.1f}",
               f"- Distinct original resolutions: {a['resolutions']['n_distinct_resolutions']}",
               f"- Near-duplicate pairs: {a['duplicates']['pairs_by_split']}",
               f"- Test images with near-duplicate in training: {a['duplicates']['test_images_with_near_duplicate_in_train']} / {a['duplicates']['test_images_total']}",
               f"- Grade/lesion rule violations: {a['label_consistency']['counts']}", "",
               "Lesion prevalence by grade (%):", "",
               df_to_markdown(pd.DataFrame(a["lesion_prevalence_by_grade_pct"]), floatfmt=".1f"), ""]
        for f in ("class_distribution.png", "nearest_neighbour_ncc.png", "duplicate_examples.png"):
            if (prepared / "audit" / f).exists():
                shutil.copy(prepared / "audit" / f, rep / "figures" / f)

    # ------------------------------------------------------------- model selection
    if (root / "model_selection.csv").exists():
        md += ["## Model selection (validation split only)", "",
               df_to_markdown(pd.read_csv(root / "model_selection.csv"), index=False), ""]

    # ---------------------------------------------------------------- test summary
    if (root / "evaluation" / "summary_test.csv").exists():
        s = pd.read_csv(root / "evaluation" / "summary_test.csv")
        for col in ("acc_ci", "f1_ci", "qwk_ci"):
            s[col] = s[col].map(lambda v: ci(v) if isinstance(v, str) else "")
        md += ["## Test performance (95% bootstrap CIs over images)", "", df_to_markdown(s, index=False), ""]
    if (root / "evaluation" / "summary_seeds.csv").exists():
        md += ["## Seed variability (mean and SD over replicates of one configuration)", "",
               df_to_markdown(pd.read_csv(root / "evaluation" / "summary_seeds.csv"), index=False), ""]
    if (root / "selection_log.json").exists():
        md += [f"Selection log: {load_json(root / 'selection_log.json')}", ""]

    ev = root / "evaluation" / str(run)
    if (ev / "metrics.json").exists():
        m = load_json(ev / "metrics.json")
        t = m["test"]
        pc = pd.DataFrame(t["per_class"]).T.rename(index=lambda i: GRADE_SHORT[int(i)])
        pc["support"] = pc["support"].astype(int).astype(str)
        if "per_class_auc" in t:
            pc["auc_ovr"] = [t["per_class_auc"][str(k)] for k in range(5)]
        md += ["## Selected model: per-class (test)", "", df_to_markdown(pc), ""]
        cm = pd.DataFrame(t["confusion_matrix"], index=[f"ref {g}" for g in GRADE_SHORT], columns=[f"pred {g}" for g in GRADE_SHORT])
        md += ["Confusion matrix (rows = reference):", "", df_to_markdown(cm), ""]
        rows = []
        for name, thr in (("Any DR (>=1)", "any_dr"), ("Referable (>=2)", "referable"), ("Severe+ (>=3)", "severe")):
            rows.append({"task": name, "sensitivity": t[f"{thr}_sensitivity"], "specificity": t[f"{thr}_specificity"],
                         "PPV": t[f"{thr}_ppv"], "NPV": t[f"{thr}_npv"], "AUC": t.get(f"{thr}_auc"),
                         "missed": t[f"{thr}_missed"], "positives": t[f"{thr}_n_positive"]})
        md += ["Clinically framed binary views (test):", "", df_to_markdown(pd.DataFrame(rows), index=False), ""]
        cs = m["cost_sensitive"]["test"]
        comp = pd.DataFrame([
            {"rule": "argmax", **{k: t[k] for k in ("accuracy", "macro_f1", "qwk", "undergrade_rate_diseased", "error_2plus_steps", "referable_sensitivity", "referable_specificity")}},
            {"rule": f"cost-sensitive (under-grading x{m['cost_sensitive']['undergrade_cost']})", **{k: cs[k] for k in ("accuracy", "macro_f1", "qwk", "undergrade_rate_diseased", "error_2plus_steps", "referable_sensitivity", "referable_specificity")}},
        ])
        md += ["Ordinal errors and decision rule (test):", "", df_to_markdown(comp, index=False), "",
               f"Under-grading rate by reference grade: {t['undergrade_by_grade']}", "",
               f"Test without images that have a near-duplicate in training (n={m['test_without_train_duplicates']['n']}): "
               f"acc {m['test_without_train_duplicates']['accuracy']:.3f}, QWK {m['test_without_train_duplicates']['qwk']:.3f}, "
               f"macro-F1 {m['test_without_train_duplicates']['macro_f1']:.3f}", ""]
        cal = pd.DataFrame(m["calibration"]).T
        md += [f"## Reliability (temperature T = {m['temperature']:.3f}, fitted on validation)", "", df_to_markdown(cal), "",
               f"Conformal 90% prediction sets: coverage {m['conformal']['test_coverage']:.3f}, mean size {m['conformal']['mean_set_size']:.2f}, "
               f"singletons {m['conformal']['singleton_rate']:.2f}; coverage by grade {m['conformal']['coverage_by_grade']}", ""]
        ref = pd.DataFrame(m["referral"]).T
        md += [f"Selective referral at target coverage {cfg['evaluation']['referral_coverage']:.0%} (threshold from validation):", "",
               df_to_markdown(ref), "", "## Error analysis by subgroup (test)", "",
               df_to_markdown(pd.read_csv(ev / "stratified_test.csv"), index=False), ""]
        for f in ("confusion_matrix.png", "confusion_matrix_cost_sensitive.png", "reliability.png", "risk_coverage_error.png",
                  "risk_coverage_missed_referable.png", "training_curves.png"):
            if (ev / f).exists():
                shutil.copy(ev / f, rep / "figures" / f)

    # ------------------------------------------------------------------ explanations
    ex = root / "explanations" / str(run)
    if (ex / "explain_summary.json").exists():
        e = load_json(ex / "explain_summary.json")
        md += ["## Explainability checks", ""]
        if (ex / "faithfulness.csv").exists():
            md += ["Deletion / insertion AUC (mean, SE; minus_random = paired difference vs random order):", "",
                   df_to_markdown(pd.read_csv(ex / "faithfulness.csv"), index=False), "",
                   f"IG completeness gap (median |relative|): {e.get('ig_completeness_gap_median_abs', float('nan')):.3f}", ""]
        if "sanity" in e:
            sn = pd.DataFrame(e["sanity"]["spearman"], index=[f"+{s}" for s in e["sanity"]["stages_cumulative"]])
            md += ["Model-randomisation sanity check (Spearman rho with trained-model map; near 0 = map depends on weights):", "",
                   df_to_markdown(sn), ""]
        if "gradcam_fov" in e:
            md += [f"Grad-CAM mass outside FOV / on rim: {e['gradcam_fov']}; images with >10% outside: "
                   f"{e['gradcam_fov_images_over_10pct_outside']}", ""]
        if (ex / "concept_probes.csv").exists():
            md += ["Lesion concept probes (test AUC):", "", df_to_markdown(pd.read_csv(ex / "concept_probes.csv"), index=False), ""]
        md += [f"Representative cases: {e['examples']}", ""]
        for f in sorted(ex.glob("*.png")):
            shutil.copy(f, rep / "figures" / f.name)

    # ------------------------------------------------------------------- robustness
    rb = root / "robustness" / str(run) / "robustness.csv"
    if rb.exists():
        md += ["## Image-quality sensitivity (synthetic degradations, test)", "", df_to_markdown(pd.read_csv(rb), index=False), ""]
        for f in rb.parent.glob("*.png"):
            shutil.copy(f, rep / "figures" / f.name)

    for loc in sorted((root / "localisation" / str(run)).glob("*/localisation_summary.csv")):
        md += [f"## Localisation vs pixel masks ({loc.parent.name})", "", df_to_markdown(pd.read_csv(loc), index=False), ""]

    (rep / "tables.md").write_text("\n".join(md))
    print(f"[collect] wrote {rep / 'tables.md'} and {len(list((rep / 'figures').glob('*.png')))} figures")


if __name__ == "__main__":
    main()
