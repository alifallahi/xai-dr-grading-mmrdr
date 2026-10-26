# Example outputs (real MMRDR-CFP run)

Outputs of `bash scripts/run_all.sh` on a Colab T4, copied from `outputs/report/`. Every number
is in [`tables.md`](tables.md). The selected model is `resnet50_wce_512` (ImageNet ResNet-50,
lr 3e-4, class weights proportional to 1/sqrt(count), seed 42), chosen on validation QWK.

| File | What it shows |
|---|---|
| `class_distribution.png` | Images per grade in the train, validation and test splits |
| `duplicate_examples.png` | Most similar image pairs found by the duplicate audit (check by eye) |
| `training_curves.png` | Loss and validation metrics per epoch for the selected run |
| `confusion_matrix.png`, `confusion_matrix_cost_sensitive.png` | Test confusion matrices, argmax and cost-sensitive rule |
| `reliability.png` | Reliability diagram before and after temperature scaling |
| `risk_coverage_missed_referable.png` | Missed referable DR among auto-accepted cases vs coverage |
| `robustness_qwk.png`, `robustness_referral.png` | QWK and referral rate under synthetic degradation |
| `panel_errors.png`, `panel_correct.png` | Grad-CAM and Integrated Gradients for rule-selected cases |
| `deletion_curves.png`, `insertion_curves.png` | Faithfulness of the maps against random ordering |
| `sanity_randomisation.png` | Cascading model-randomisation sanity check |
| `decision_card_*.png` | Decision-card prototypes (a referred case and the worst error) |

**Licence of the images in these figures.** Figures that show fundus photographs contain images
from MMRDR (Tang et al., 2026; CC BY 4.0), which originate from OIA-DDR (Li et al., 2019;
CC BY-NC-SA 4.0). Those figures are shared under CC BY-NC-SA 4.0 for non-commercial research use,
not under the repository's MIT code licence.
