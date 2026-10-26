# Explainable diabetic retinopathy grading on MMRDR colour fundus photographs

A reproducible research prototype that grades diabetic retinopathy (DR) from colour fundus
photographs (CFP) into the five ICDR grades, and then asks whether its **explanations** and
**confidence** can be trusted by a clinician reviewing the result.

> **Research prototype.** Not a medical device, not clinically validated, not for patient care.

| Grade | Label |
|---|---|
| 0 | No DR |
| 1 | Mild non-proliferative DR (NPDR) |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | Proliferative DR (PDR) |

**What the pipeline does**

1. **Data audit and preparation** - checks the annotation file against the dataset paper, crops each
   image to its field of view (FOV) and resizes it, measures image quality, searches for near-duplicate
   images (including train-test overlap) and checks grade/lesion label consistency.
2. **Models** - a majority-class baseline, a linear probe on frozen ImageNet ResNet-50 features, and an
   ImageNet-pretrained ResNet-50 fine-tuned at 512 x 512, with a 2 x 2 grid over learning rate and loss
   weighting and two extra seeds of the chosen configuration.
3. **Validation-only model selection** - the test split is used once, after all training is finished.
4. **Evaluation** - accuracy, macro-F1, quadratic weighted kappa (QWK), per-class metrics, confusion
   matrix, referable/severe DR sensitivity and specificity, under-grading rates, bootstrap 95% CIs, and
   an a-priori cost-sensitive decision rule that penalises under-grading.
5. **Reliability** - temperature scaling, ECE, conformal prediction sets, selective referral
   (risk-coverage) and sensitivity to synthetic image degradation.
6. **Explainability** - Grad-CAM and Integrated Gradients on representative cases, plus checks on the
   explanations themselves: deletion/insertion faithfulness, model-randomisation sanity test,
   attribution outside the FOV, lesion concept probes against MMRDR's image-level lesion labels, and
   (optional) localisation against pixel-level lesion masks from an external dataset.

## Results at a glance

Official MMRDR-CFP test split (2,225 images), selected model chosen on validation only.
Full tables and figures: [`docs/example_outputs/`](docs/example_outputs/).

| Model | Accuracy | Macro-F1 | QWK (95% CI) | Referable DR sens. / spec. |
|---|---|---|---|---|
| Majority class | 0.595 | 0.149 | 0.000 | 0.000 / 1.000 |
| Linear probe (frozen ImageNet ResNet-50) | 0.678 | 0.566 | 0.782 (0.758-0.803) | 0.745 / 0.893 |
| **ResNet-50 fine-tuned (selected)** | **0.813** | **0.709** | **0.897 (0.882-0.911)** | **0.810 / 0.951** |
| Same configuration, 3 seeds (mean +/- SD) | 0.818 +/- 0.005 | 0.713 +/- 0.008 | 0.897 +/- 0.005 | 0.833 / 0.946 |
| MMRDR paper, ResNet-50 (reported) | 0.823 | 0.702 | - | - |

* Referring the 21% most uncertain cases (expected misgrading cost, threshold from validation)
  captured 71% of under-gradings and cut missed referable DR among auto-accepted cases from 122 to 25.
* Blur and under-exposure raised uncertainty; contrast loss and JPEG compression degraded the model
  silently (JPEG quality 20: QWK 0.09 at mean confidence 0.84).
* On IDRiD pixel masks, Grad-CAM peaks hit haemorrhages in 50% of images (image-independent prior: 11%)
  but rarely microaneurysms; Integrated Gradients only partly passed the model-randomisation check.
* The audit found 41% of Mild NPDR labels recording hard exudates, contrary to the dataset's own
  definition of Mild (microaneurysms only).

## Repository layout

```
configs/                  YAML configs (default + three grid points; CPU smoke test)
src/drxai/
  data/mmrdr.py           reading FP.csv (grades, 7 lesion labels, official split)
  data/preprocess.py      FOV mask, crop, square pad, resize, quality metrics, duplicate thumbnails
  data/audit.py           near-duplicate search, grade/lesion consistency rules, summaries
  data/splits.py          grade-stratified, duplicate-grouped train/validation split
  data/dataset.py         PyTorch dataset and transforms
  models.py               timm model construction, Grad-CAM layer, randomisation stages
  train.py                training loop with validation-based checkpointing
  metrics.py              classification, ordinal and clinical metrics, bootstrap CIs
  calibration.py          temperature scaling, ECE, NLL, Brier
  reliability.py          cost-sensitive rule, uncertainty scores, risk-coverage, conformal sets
  corruptions.py          synthetic image-quality degradations
  xai/attribution.py      Grad-CAM, Integrated Gradients (implemented here, no external XAI library)
  xai/evaluation.py       deletion/insertion, randomisation similarity, FOV mass, localisation scores
  plotting.py             all figures
scripts/                  one script per pipeline step (see below)
tests/                    unit tests (metrics, calibration, attribution correctness, data handling)
docs/example_outputs/     figures and tables from the real MMRDR-CFP run
```

## Data

MMRDR (Tang et al., *Scientific Data* 2026) is on figshare:
[doi:10.6084/m9.figshare.29423747](https://doi.org/10.6084/m9.figshare.29423747), licensed CC BY 4.0.
Only the CFP subset is used: 11,118 images (official split approx. 8:2, train `tr*.jpg` / test `ts*.jpg`),
with `FP.csv` giving the grade (0-4), a 7-element image-level lesion vector
`[MA, HE (hard exudate), IH, VB/IRMA, NV, VH, RD]` and laterality. The CFP images originate from
OIA-DDR (Li et al., 2019), released under **CC BY-NC-SA 4.0**; check both licences before any
non-academic use. **No data are included in this repository.**

**Option A - script** (downloads all 9 archive parts, ~18.6 GB, verifies MD5, extracts only CFP):

```bash
python scripts/download_mmrdr.py --out data/MMRDR --work data/mmrdr_download
```

**Option B - manual**: download `MMRDR.zip.001` ... `MMRDR.zip.009` from figshare into one folder, then

```bash
python scripts/download_mmrdr.py --parts-dir /path/to/parts --out data/MMRDR
# or: cat MMRDR.zip.0* > MMRDR.zip && unzip MMRDR.zip "*MMRDR-CFP/*"   (or 7z x MMRDR.zip.001)
```

Expected result (set `paths.mmrdr_cfp_dir` if you put it elsewhere):

```
data/MMRDR/MMRDR-CFP/
  FP.csv
  img/tr000001.jpg ... ts00XXXX.jpg
```

`prepare_data.py` compares the grade and lesion totals with Table 2 of the paper and warns if they differ
(this verifies the lesion column order).

## Installation

Python 3.10+ and a CUDA GPU are recommended (a 16 GB T4 is enough).

```bash
git clone https://github.com/alifallahi/xai-dr-grading-mmrdr.git && cd xai-dr-grading-mmrdr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q          # 21 unit tests, about a minute on CPU
bash scripts/run_smoke_test.sh   # whole pipeline on synthetic data, CPU, 5-10 minutes
```

`requirements.txt` gives minimum versions; `requirements-lock.txt` the exact versions the code was tested with.

**Google Colab / Kaggle**: keep the preinstalled PyTorch, then in a notebook cell:

```bash
!git clone https://github.com/alifallahi/xai-dr-grading-mmrdr.git && cd xai-dr-grading-mmrdr && pip install -q -r requirements.txt && pip install -q -e .
!cd xai-dr-grading-mmrdr && python scripts/download_mmrdr.py --out data/MMRDR
!cd xai-dr-grading-mmrdr && bash scripts/run_all.sh
```

The ImageNet weights (`resnet50.tv2_in1k`) are downloaded by timm from the Hugging Face Hub on first use.

## Running the experiments

`bash scripts/run_all.sh` runs every step below, in the order used for the reported results. Each step
reads the previous step's outputs from disk, so steps can be re-run on their own after a disconnect.
Times are for a Colab T4 (about 3 minutes per training epoch at 512 px).

| Step | Command | Output | Time |
|---|---|---|---|
| 1. Audit, preprocess, split | `python scripts/prepare_data.py` | `data/prepared/` (`metadata.csv`, `splits.csv`, `audit/`) | ~1 h (CPU-bound) |
| 2. Linear-probe baseline | `python scripts/linear_probe.py` | `outputs/runs/linear_probe/` | ~5 min |
| 3. Hyperparameter grid (4 runs) | `python scripts/train.py --config configs/<grid config>` | `outputs/runs/<run>/` | ~1 h each |
| 4. Model selection (validation) | `python scripts/select_model.py` | `outputs/selected_run.txt`, `model_selection.csv` | seconds |
| 5. Seed replicates (2 runs) | `python scripts/train_seeds.py --seeds 1 2` | `outputs/runs/<selected>_seed1`, `_seed2` | ~1 h each |
| 6. Test evaluation (all runs) | `python scripts/evaluate.py` | `outputs/evaluation/`, `summary_test.csv`, `summary_seeds.csv` | ~10 min |
| 7. Explanations and checks | `python scripts/explain.py` | `outputs/explanations/<run>/` | ~20 min |
| 8. Image-quality sensitivity | `python scripts/robustness.py` | `outputs/robustness/<run>/` | ~10 min |
| 9. Localisation (optional, IDRiD) | see below | `outputs/localisation/<run>/idrid/` | ~5 min |
| 10. Tables and figures for the report | `python scripts/collect_results.py` | `outputs/report/tables.md`, `figures/` | seconds |

**Grid and selection.** The four grid configurations are `configs/default.yaml` (lr 3e-4, class-weighted),
`configs/ablation_unweighted.yaml` (lr 3e-4, unweighted), `configs/lr1e-4_weighted.yaml` and
`configs/lr1e-4_unweighted.yaml`, all with seed 42. `select_model.py` picks the run with the highest
validation QWK. `train_seeds.py` then retrains that configuration with seeds 1 and 2 to measure run-to-run
variability; the replicates are never selection candidates, and the final model remains the seed-42 run.
All training finishes before `evaluate.py` first reads the test split. After that, `select_model.py`
freezes the selection (it only reports the validation ranking) unless `--reselect` is passed.

All scripts take `--config` (default `configs/default.yaml`) and `--set key=value` overrides.

### Localisation against pixel-level lesion masks (optional)

MMRDR's lesion labels say *whether* a lesion is present, not *where*. IDRiD (Porwal et al., 2018) has
81 fundus images with pixel masks for microaneurysms (MA), haemorrhages (HE), hard exudates (EX) and
soft exudates (SE). Download `A. Segmentation.zip` (~560 MB) from
[IEEE DataPort](https://ieee-dataport.org/node/956)
(free account; check the licence there), unzip it, then:

```bash
python scripts/prepare_idrid.py --src "path/to/A. Segmentation" --out data/idrid
python scripts/localisation_external.py --images data/idrid/images --masks data/idrid/masks \
       --lesions MA HE EX SE --name idrid
```

For each lesion type this reports the pointing-game hit rate and the share of attribution inside the
lesion masks for Grad-CAM, Integrated Gradients and an image-independent prior (mean Grad-CAM map).
IDRiD is an external dataset (Indian population, one camera), so this tests transfer as well as
localisation. Note the naming: in IDRiD HE = haemorrhage and EX = hard exudate; in MMRDR HE = hard exudate.

## Experimental protocol

* **Splits.** The official MMRDR-CFP test split is kept intact. One eighth of the official training split
  (7:1, as in the MMRDR benchmark) becomes validation, stratified by grade. MMRDR-CFP has no patient
  identifiers (the official split is image-level), so a patient-level split is impossible; as a partial
  safeguard, confirmed near-duplicate images are kept in the same fold (`StratifiedGroupKFold`).
* **Leakage audit.** Near-duplicates are found by exhaustive normalised cross-correlation of high-pass
  green-channel thumbnails (threshold `data.dup_ncc_min`, check `audit/nearest_neighbour_ncc.png` and
  `audit/duplicate_examples.png` by eye). Test metrics are also reported without test images that have
  a near-duplicate in training. Fellow eyes and repeat visits of the same patient cannot be detected.
* **Model selection and hyperparameters.** Learning rate (3e-4, 1e-4) and loss weighting (1/sqrt(count),
  none) are chosen on a 2 x 2 grid; checkpoints (early stopping on validation QWK) and the choice between
  runs use validation data only. Other settings are fixed a priori. Two extra seeds of the selected
  configuration estimate run-to-run variability. Temperature, referral thresholds and conformal
  thresholds are fitted on validation. The cost multiplier for under-grading (2.0) is fixed a priori.
* **Test set.** Read only by `evaluate.py` (and the analyses that follow it), after selection is final.
  All training, including seed replicates, finishes before this; the selection is then frozen.
* **Seeds.** `seed: 42` seeds Python, NumPy, PyTorch, data-loader workers and the split. cuDNN autotuning
  is on for speed, so GPU runs are repeatable only to within small numerical differences; set
  `train.deterministic: true` for bitwise repeatability (slower).

Key hyperparameters (`configs/default.yaml`; grid values chosen on validation):

| Setting | Value |
|---|---|
| Backbone | ResNet-50, ImageNet-1k weights (`timm: resnet50.tv2_in1k`) |
| Input | FOV crop, square pad, 512 x 512, ImageNet normalisation |
| Augmentation | horizontal flip, rotation +/-25 deg, scale 0.9-1.1, translate 5%, brightness/contrast 0.2, saturation 0.1 |
| Loss | cross-entropy; grid: class weights proportional to 1/sqrt(class count), or none |
| Optimiser | AdamW, grid: lr 3e-4 or 1e-4; weight decay 1e-4 (none on norms/biases), 1 warm-up epoch, cosine decay |
| Batch / epochs | 16 / up to 25, early stopping patience 7 on validation QWK |
| Precision | mixed (fp16) on GPU, gradient clipping 1.0 |

## Outputs

`outputs/report/tables.md` collects every number used in the report, and `outputs/report/figures/` the
figures: confusion matrices, reliability diagram, risk-coverage curves, explanation panels, decision-card
prototypes, deletion/insertion curves, the randomisation test and the robustness curves.
Per-image predictions (with probabilities, uncertainty, referral flag and conformal set) are in
`outputs/evaluation/<run>/test_predictions.csv`. See `docs/example_outputs/` for representative outputs.

## External resources and acknowledgements

* **Data**: MMRDR (Tang et al., 2026, CC BY 4.0); CFP images from OIA-DDR (Li et al., 2019, CC BY-NC-SA 4.0).
* **Pretrained weights**: ImageNet-1k ResNet-50 (`resnet50.tv2_in1k`, torchvision recipe) via timm.
* **Libraries**: PyTorch, torchvision, timm, scikit-learn, SciPy, OpenCV, NumPy, pandas, Matplotlib.
* **Consulted, not copied**: the MMRDR authors' evaluation code
  (github.com/Vladimirovich2019/MMRDR_Evaluation) to confirm the `FP.csv` format and lesion order.
* **Methods**: Grad-CAM (Selvaraju et al., 2017), Integrated Gradients (Sundararajan et al., 2017),
  blurred baselines (Sturmfels et al., 2020), deletion/insertion (Petsiuk et al., 2018), randomisation
  test (Adebayo et al., 2018), temperature scaling (Guo et al., 2017), split conformal prediction.
  Grad-CAM, IG and all checks are implemented in this repository.
* **AI assistance**: Claude (Anthropic) was used as an AI coding assistant to support code development and for proofreading, grammar checking, and improving the clarity of the report. All experiments were conducted independently by me, and I was responsible for all decisions, interpreting the results, and reviewing and validating the final text.

## Licence

Code: MIT (see `LICENSE`). Data are not covered by this licence; see the dataset licences above.
