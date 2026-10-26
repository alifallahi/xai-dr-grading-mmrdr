# Results tables (selected run: `resnet50_wce_512`)

## Data

Images per grade by split (model splits):

|  | test | train | val |
|---|---|---|---|
| 0 No DR | 1323 (59.5%) | 4579 (58.9%) | 677 (60.7%) |
| 1 Mild | 259 (11.6%) | 904 (11.6%) | 139 (12.5%) |
| 2 Moderate | 409 (18.4%) | 1386 (17.8%) | 164 (14.7%) |
| 3 Severe | 100 (4.5%) | 441 (5.7%) | 62 (5.6%) |
| 4 PDR | 134 (6.0%) | 468 (6.0%) | 73 (6.5%) |
| total | 2225 | 7778 | 1115 |

- Grade counts match paper Table 2: True; lesion counts match: True
- Imbalance (largest/smallest class, train): 10.4
- Distinct original resolutions: 1857
- Near-duplicate pairs: {'test-test': {'pairs': 6, 'grade_discordant_pairs': 0, 'mean_abs_grade_difference': 0.0}, 'test-train': {'pairs': 58, 'grade_discordant_pairs': 1, 'mean_abs_grade_difference': 0.017241379310344827}, 'train-train': {'pairs': 105, 'grade_discordant_pairs': 3, 'mean_abs_grade_difference': 0.02857142857142857}}
- Test images with near-duplicate in training: 50 / 2225
- Grade/lesion rule violations: {'R1_noDR_with_lesion': 0, 'R2_DR_without_lesion': 0, 'R3_mild_with_non_MA': 537, 'R4_PDR_without_PDR_sign': 0, 'R5_PDR_sign_below_PDR': 0, 'R6_VB_IRMA_below_moderate': 0, 'label_inconsistent': 537}

Lesion prevalence by grade (%):

|  | MA | HE | IH | VB_IRMA | NV | VH | RD |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 100.0 | 41.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 100.0 | 62.7 | 86.9 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | 100.0 | 82.6 | 88.4 | 32.7 | 0.0 | 0.0 | 0.0 |
| 4 | 79.1 | 56.9 | 51.1 | 5.8 | 82.4 | 53.0 | 13.8 |

## Model selection (validation split only)

| run | fine_tuned | replicate_of | lr | class_weighting | seed | val_qwk | val_macro_f1 | val_accuracy | best_epoch | minutes | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50_wce_512 | True |  | 0.000 | inverse_sqrt | 42.000 | 0.920 | 0.758 | 0.842 | 11.000 | 54.600 | True |
| resnet50_ce_512_lr1e-4 | True |  | 0.000 | none | 42.000 | 0.917 | 0.756 | 0.839 | 8.000 | 45.900 | False |
| resnet50_wce_512_lr1e-4 | True |  | 0.000 | inverse_sqrt | 42.000 | 0.912 | 0.747 | 0.842 | 12.000 | 57.700 | False |
| resnet50_ce_512 | True |  | 0.000 | none | 42.000 | 0.906 | 0.732 | 0.836 | 9.000 | 50.500 | False |
| linear_probe | False |  |  |  |  | 0.814 | 0.597 | 0.701 |  |  | False |

## Test performance (95% bootstrap CIs over images)

| run | selected | accuracy | acc_ci | macro_f1 | f1_ci | qwk | qwk_ci | referable_sens | referable_spec | referable_auc | undergrade_diseased | ece_after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority_class | False | 0.595 | 0.573-0.615 | 0.149 | 0.146-0.152 | 0.000 | 0.000-0.000 | 0.000 | 1.000 | 0.500 | 1.000 | 0.002 |
| linear_probe | False | 0.678 | 0.657-0.698 | 0.566 | 0.537-0.592 | 0.782 | 0.758-0.803 | 0.745 | 0.893 | 0.923 | 0.315 | 0.024 |
| resnet50_wce_512 | True | 0.813 | 0.796-0.828 | 0.709 | 0.683-0.734 | 0.897 | 0.882-0.911 | 0.810 | 0.951 | 0.971 | 0.211 | 0.015 |
| resnet50_ce_512 | False | 0.825 | 0.809-0.840 | 0.697 | 0.668-0.724 | 0.891 | 0.873-0.906 | 0.900 | 0.919 | 0.972 | 0.210 | 0.020 |
| resnet50_ce_512_lr1e-4 | False | 0.818 | 0.801-0.834 | 0.705 | 0.680-0.729 | 0.897 | 0.881-0.911 | 0.865 | 0.939 | 0.974 | 0.183 | 0.019 |
| resnet50_wce_512_lr1e-4 | False | 0.825 | 0.809-0.840 | 0.718 | 0.691-0.741 | 0.893 | 0.877-0.908 | 0.810 | 0.937 | 0.970 | 0.244 | 0.023 |
| resnet50_wce_512_seed1 | False | 0.822 | 0.805-0.836 | 0.722 | 0.696-0.746 | 0.903 | 0.886-0.916 | 0.834 | 0.952 | 0.971 | 0.193 | 0.023 |
| resnet50_wce_512_seed2 | False | 0.820 | 0.804-0.836 | 0.707 | 0.678-0.733 | 0.893 | 0.876-0.907 | 0.855 | 0.935 | 0.970 | 0.218 | 0.022 |
| MMRDR paper: ResNet-50 FT (reported) |  | 0.823 |  | 0.702 |  |  |  |  |  |  |  |  |

## Seed variability (mean and SD over replicates of one configuration)

| configuration | n_seeds | val_qwk_mean | val_qwk_sd | val_macro_f1_mean | val_macro_f1_sd | accuracy_mean | accuracy_sd | macro_f1_mean | macro_f1_sd | qwk_mean | qwk_sd | referable_sens_mean | referable_sens_sd | referable_spec_mean | referable_spec_sd | ece_after_mean | ece_after_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50_wce_512 | 3 | 0.912 | 0.009 | 0.746 | 0.011 | 0.818 | 0.005 | 0.713 | 0.008 | 0.897 | 0.005 | 0.833 | 0.023 | 0.946 | 0.010 | 0.020 | 0.004 |

Selection log: {'selected': 'resnet50_wce_512', 'best_on_validation': 'resnet50_wce_512', 'frozen': False, 'metric': 'val_qwk', 'note': "selected 'resnet50_wce_512' by val_qwk on the validation split"}

## Selected model: per-class (test)

|  | precision | recall | f1 | support | auc_ovr |
|---|---|---|---|---|---|
| 0 No DR | 0.953 | 0.937 | 0.945 | 1323 | 0.984 |
| 1 Mild | 0.498 | 0.629 | 0.556 | 259 | 0.910 |
| 2 Moderate | 0.732 | 0.533 | 0.617 | 409 | 0.925 |
| 3 Severe | 0.466 | 0.890 | 0.612 | 100 | 0.976 |
| 4 PDR | 0.908 | 0.739 | 0.815 | 134 | 0.989 |

Confusion matrix (rows = reference):

|  | pred 0 No DR | pred 1 Mild | pred 2 Moderate | pred 3 Severe | pred 4 PDR |
|---|---|---|---|---|---|
| ref 0 No DR | 1239 | 69 | 13 | 1 | 1 |
| ref 1 Mild | 34 | 163 | 56 | 4 | 2 |
| ref 2 Moderate | 25 | 88 | 218 | 74 | 4 |
| ref 3 Severe | 0 | 4 | 4 | 89 | 3 |
| ref 4 PDR | 2 | 3 | 7 | 23 | 99 |

Clinically framed binary views (test):

| task | sensitivity | specificity | PPV | NPV | AUC | missed | positives |
|---|---|---|---|---|---|---|---|
| Any DR (>=1) | 0.932 | 0.937 | 0.909 | 0.953 | 0.984 | 61 | 902 |
| Referable (>=2) | 0.810 | 0.951 | 0.871 | 0.925 | 0.971 | 122 | 643 |
| Severe+ (>=3) | 0.915 | 0.957 | 0.713 | 0.990 | 0.985 | 20 | 234 |

Ordinal errors and decision rule (test):

| rule | accuracy | macro_f1 | qwk | undergrade_rate_diseased | error_2plus_steps | referable_sensitivity | referable_specificity |
|---|---|---|---|---|---|---|---|
| argmax | 0.813 | 0.709 | 0.897 | 0.211 | 0.030 | 0.810 | 0.951 |
| cost-sensitive (under-grading x2.0) | 0.783 | 0.682 | 0.896 | 0.129 | 0.028 | 0.894 | 0.917 |

Under-grading rate by reference grade: {'1': 0.13127413127413126, '2': 0.2762836185819071, '3': 0.08, '4': 0.26119402985074625}

Test without images that have a near-duplicate in training (n=2175): acc 0.810, QWK 0.896, macro-F1 0.710

## Reliability (temperature T = 1.219, fitted on validation)

|  | ece | nll | brier | mean_confidence | accuracy |
|---|---|---|---|---|---|
| test_before | 0.031 | 0.505 | 0.266 | 0.841 | 0.813 |
| test_after | 0.015 | 0.494 | 0.265 | 0.809 | 0.813 |
| val_after | 0.035 | 0.460 | 0.238 | 0.817 | 0.841 |

Conformal 90% prediction sets: coverage 0.885, mean size 1.21, singletons 0.79; coverage by grade {'0': 0.9523809523809523, '1': 0.7915057915057915, '2': 0.7506112469437652, '3': 0.9, '4': 0.7985074626865671}

Selective referral at target coverage 80% (threshold from validation):

|  | threshold | coverage | n_referred | errors_captured | undergrades_captured | missed_referable_all | missed_referable_accepted | accepted_accuracy | accepted_qwk | accepted_macro_f1 | accepted_undergrade_rate_diseased | referred_accuracy | aurc_error | aurc_missed_referable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | 0.368 | 0.776 | 499.000 | 0.559 | 0.563 | 122.000 | 46.000 | 0.893 | 0.945 | 0.762 | 0.162 | 0.533 | 0.056 | 0.013 |
| entropy | 0.834 | 0.772 | 507.000 | 0.559 | 0.532 | 122.000 | 51.000 | 0.893 | 0.947 | 0.758 | 0.171 | 0.540 | 0.057 | 0.014 |
| exp_cost | 0.651 | 0.786 | 477.000 | 0.508 | 0.705 | 122.000 | 25.000 | 0.883 | 0.946 | 0.757 | 0.098 | 0.556 | 0.066 | 0.010 |

## Error analysis by subgroup (test)

| subgroup | n | accuracy | undergrade_rate_diseased | mean_grade | qwk |
|---|---|---|---|---|---|
| all test images | 2225 | 0.813 | 0.211 | 0.860 | 0.897 |
| lesion present: MA | 876 | 0.622 | 0.215 | 2.065 | 0.712 |
| lesion present: HE | 506 | 0.619 | 0.178 | 2.200 | 0.726 |
| lesion present: IH | 511 | 0.624 | 0.254 | 2.438 | 0.620 |
| lesion present: VB_IRMA | 43 | 0.744 | 0.233 | 3.116 | 0.011 |
| lesion present: NV | 121 | 0.719 | 0.281 | 4.000 |  |
| lesion present: VH | 53 | 0.943 | 0.057 | 4.000 |  |
| lesion present: RD | 21 | 0.952 | 0.048 | 4.000 |  |
| severe NPDR with VB/IRMA | 38 | 0.816 | 0.158 | 3.000 |  |
| severe NPDR without VB/IRMA | 62 | 0.935 | 0.032 | 3.000 |  |
| PDR with NV annotated | 121 | 0.719 | 0.281 | 4.000 |  |
| PDR without NV (VH/RD only) | 13 | 0.923 | 0.077 | 4.000 |  |
| sharpness Q1 (lowest) | 580 | 0.769 | 0.224 | 1.207 | 0.890 |
| sharpness Q2 | 522 | 0.774 | 0.186 | 1.067 | 0.895 |
| sharpness Q3 | 562 | 0.806 | 0.236 | 0.662 | 0.857 |
| sharpness Q4 (highest) | 561 | 0.900 | 0.196 | 0.506 | 0.924 |
| brightness Q1 (lowest) | 546 | 0.819 | 0.220 | 0.866 | 0.895 |
| brightness Q2 | 508 | 0.797 | 0.230 | 0.797 | 0.892 |
| brightness Q3 | 605 | 0.833 | 0.214 | 0.838 | 0.914 |
| brightness Q4 (highest) | 566 | 0.799 | 0.185 | 0.933 | 0.884 |
| near-duplicate in training set | 50 | 0.920 | 0.500 | 0.240 | 0.915 |
| no near-duplicate in training set | 2175 | 0.810 | 0.209 | 0.874 | 0.896 |
| resolution 1657x1657 (camera proxy) | 91 | 0.857 | 0.227 | 0.462 | 0.897 |
| resolution 1571x1570 (camera proxy) | 86 | 0.860 | 0.250 | 0.628 | 0.936 |
| resolution 1659x1659 (camera proxy) | 86 | 0.767 | 0.300 | 0.709 | 0.838 |
| resolution 1569x1568 (camera proxy) | 80 | 0.825 | 0.171 | 0.838 | 0.919 |
| resolution 1713x1713 (camera proxy) | 65 | 0.969 | 0.667 | 0.092 | 0.696 |
| resolution 1602x1602 (camera proxy) | 61 | 0.672 | 0.136 | 2.197 | 0.751 |
| resolution 1655x1655 (camera proxy) | 47 | 1.000 | 0.000 | 0.064 | 1.000 |
| resolution 1380x1382 (camera proxy) | 46 | 0.696 | 0.206 | 1.674 | 0.911 |
| resolution 1571x1569 (camera proxy) | 45 | 0.778 | 0.227 | 1.111 | 0.925 |
| resolution 1658x1658 (camera proxy) | 41 | 0.902 | 0.167 | 0.244 | 0.843 |
| resolution 2354x2354 (camera proxy) | 36 | 0.806 | 0.200 | 0.389 | 0.793 |
| resolution 1655x1654 (camera proxy) | 33 | 1.000 |  | 0.000 |  |
| resolution 1662x1662 (camera proxy) | 31 | 0.806 | 0.143 | 1.258 | 0.874 |
| grade/lesion labels inconsistent | 116 | 0.543 | 0.078 | 1.000 |  |
| grade/lesion labels consistent | 2109 | 0.827 | 0.230 | 0.852 | 0.905 |

## Explainability checks

Deletion / insertion AUC (mean, SE; minus_random = paired difference vs random order):

| method | mode | auc_mean | auc_se | n_images | minus_random_mean | minus_random_se |
|---|---|---|---|---|---|---|
| Grad-CAM | deletion | 0.289 | 0.019 | 200 | -0.116 | 0.013 |
| Grad-CAM | insertion | 0.513 | 0.023 | 200 | 0.116 | 0.010 |
| Integrated Gradients | deletion | 0.294 | 0.019 | 200 | -0.111 | 0.014 |
| Integrated Gradients | insertion | 0.538 | 0.023 | 200 | 0.140 | 0.008 |
| Random | deletion | 0.405 | 0.023 | 200 |  |  |
| Random | insertion | 0.397 | 0.023 | 200 |  |  |

IG completeness gap (median |relative|): 0.012

Model-randomisation sanity check (Spearman rho with trained-model map; near 0 = map depends on weights):

|  | Grad-CAM | Integrated Gradients |
|---|---|---|
| +fc | 0.183 | 0.749 |
| +layer4 | 0.135 | 0.579 |
| +layer3 | 0.072 | 0.470 |
| +layer2 | 0.019 | 0.389 |
| +layer1 | 0.001 | 0.419 |

Grad-CAM mass outside FOV / on rim: {'mass_outside': {'mean': 0.042829089447453075, '50%': 0.03871277719736099, 'max': 0.2276173084974289}, 'area_outside': {'mean': 0.21802968914589185, '50%': 0.21290969848632812, 'max': 0.3209266662597656}, 'mass_rim': {'mean': 0.09705052962170893, '50%': 0.08954619616270065, 'max': 0.3954221308231354}, 'area_rim': {'mean': 0.1475051845593399, '50%': 0.14886474609375, 'max': 0.1500244140625}}; images with >10% outside: 94

Lesion concept probes (test AUC):

| lesion | n_pos_test | auc_fine-tuned | within_grade_auc_fine-tuned | auc_ImageNet (frozen) | within_grade_auc_ImageNet (frozen) |
|---|---|---|---|---|---|
| MA | 876 | 0.980 | 0.850 | 0.926 | 0.756 |
| HE | 506 | 0.947 | 0.803 | 0.919 | 0.768 |
| IH | 511 | 0.942 | 0.631 | 0.886 | 0.547 |
| VB_IRMA | 43 | 0.946 | 0.545 | 0.916 | 0.593 |
| NV | 121 | 0.990 | 0.764 | 0.959 | 0.644 |
| VH | 53 | 0.997 | 0.932 | 0.983 | 0.833 |
| RD | 21 | 0.993 | 0.908 | 0.989 | 0.863 |

Representative cases: {'correct': ['ts000247', 'ts001526', 'ts000629', 'ts000441', 'ts000025'], 'errors': ['ts002065', 'ts002176', 'ts002077', 'ts000632'], 'referred': ['ts002100']}

## Image-quality sensitivity (synthetic degradations, test)

| corruption | level | parameter | accuracy | qwk | macro_f1 | undergrade_rate_diseased | referable_sensitivity | mean_pred_grade | mean_confidence | referral_rate | errors_referred |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 0.000 | 0.813 | 0.897 | 0.709 | 0.211 | 0.810 | 0.868 | 0.809 | 0.214 | 0.508 |
| gaussian_blur | 1 | 1.000 | 0.796 | 0.888 | 0.687 | 0.214 | 0.818 | 0.884 | 0.764 | 0.284 | 0.500 |
| gaussian_blur | 2 | 2.000 | 0.772 | 0.869 | 0.642 | 0.265 | 0.823 | 0.864 | 0.756 | 0.313 | 0.546 |
| gaussian_blur | 3 | 4.000 | 0.612 | 0.533 | 0.412 | 0.212 | 0.893 | 1.467 | 0.723 | 0.511 | 0.562 |
| darken | 1 | 1.500 | 0.807 | 0.880 | 0.687 | 0.269 | 0.796 | 0.810 | 0.814 | 0.205 | 0.506 |
| darken | 2 | 2.000 | 0.791 | 0.849 | 0.653 | 0.347 | 0.778 | 0.748 | 0.778 | 0.293 | 0.621 |
| darken | 3 | 3.000 | 0.691 | 0.699 | 0.519 | 0.465 | 0.742 | 0.776 | 0.672 | 0.526 | 0.825 |
| low_contrast | 1 | 0.750 | 0.813 | 0.890 | 0.713 | 0.251 | 0.773 | 0.825 | 0.786 | 0.243 | 0.562 |
| low_contrast | 2 | 0.500 | 0.802 | 0.855 | 0.677 | 0.345 | 0.760 | 0.737 | 0.786 | 0.256 | 0.646 |
| low_contrast | 3 | 0.300 | 0.747 | 0.766 | 0.538 | 0.513 | 0.687 | 0.601 | 0.795 | 0.271 | 0.645 |
| jpeg | 1 | 50.000 | 0.689 | 0.798 | 0.576 | 0.149 | 0.897 | 1.130 | 0.639 | 0.599 | 0.739 |
| jpeg | 2 | 20.000 | 0.190 | 0.085 | 0.132 | 0.020 | 0.997 | 3.373 | 0.840 | 0.219 | 0.161 |
| jpeg | 3 | 8.000 | 0.179 | -0.017 | 0.106 | 0.299 | 0.731 | 2.667 | 0.776 | 0.385 | 0.333 |

## Localisation vs pixel masks (idrid)

| lesion | method | n | pointing_hit_rate | mass_in_lesion | lesion_area_chance | mass_enrichment |
|---|---|---|---|---|---|---|
| EX | Grad-CAM | 81 | 0.136 | 0.250 | 0.179 | 1.397 |
| EX | Integrated Gradients | 81 | 0.185 | 0.243 | 0.179 | 1.356 |
| EX | Prior (mean Grad-CAM) | 81 | 0.062 | 0.178 | 0.179 | 0.990 |
| HE | Grad-CAM | 80 | 0.500 | 0.294 | 0.128 | 2.287 |
| HE | Integrated Gradients | 80 | 0.550 | 0.329 | 0.128 | 2.563 |
| HE | Prior (mean Grad-CAM) | 80 | 0.113 | 0.133 | 0.128 | 1.038 |
| MA | Grad-CAM | 81 | 0.185 | 0.191 | 0.133 | 1.442 |
| MA | Integrated Gradients | 81 | 0.259 | 0.178 | 0.133 | 1.343 |
| MA | Prior (mean Grad-CAM) | 81 | 0.025 | 0.135 | 0.133 | 1.015 |
| SE | Grad-CAM | 40 | 0.075 | 0.076 | 0.034 | 2.234 |
| SE | Integrated Gradients | 40 | 0.100 | 0.100 | 0.034 | 2.965 |
| SE | Prior (mean Grad-CAM) | 40 | 0.100 | 0.038 | 0.034 | 1.114 |
