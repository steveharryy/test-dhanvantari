# Results — External 100-image test of Dhanvantri Drishti (INT8 student)

## Test set
100 dermoscopic images from the public ISIC Archive (CC-0 licence), one image per lesion, sampled with a fixed seed (42). Image IDs in the HAM10000 range (ISIC_0024306–0034320) were excluded to limit overlap with the likely training data. ISIC diagnoses were mapped to the 7 DD classes (SCC in situ → akiec; seborrheic keratosis, solar lentigo, lichen-planus-like keratosis → bkl; hemangioma → vasc; invasive melanoma and melanoma in situ → mel).

| Class | akiec | bcc | bkl | df | nv | vasc | mel | Total |
|---|---|---|---|---|---|---|---|---|
| Images | 12 | 15 | 15 | 10 | 18 | 10 | 20 | 100 |

Of these, 47 are high-risk (akiec/bcc/mel) and 53 are not. The list of cases is in `test_split.csv`. Per-image outputs are in `results/per_image_predictions.csv`.

Pipeline: `evaluate_dd.py` unchanged (tf.image.resize bilinear to 96×96, scaling to [-1, 1], INT8 inference, T = 3.1843, τ = 0.2282, high-risk = akiec + bcc + mel).

## Headline results

| Metric | Value | 95% CI (bootstrap) |
|---|---|---|
| 7-class top-1 accuracy | 43 / 100 = **0.43** | 0.33–0.53 |
| Macro-F1 | **0.378** | – |
| REVIEW ADVISED sensitivity | 38 / 47 = **0.809** | 0.69–0.92 |
| REVIEW ADVISED specificity | 18 / 53 = **0.340** | 0.22–0.48 |
| PPV / NPV | 0.521 / 0.667 | – |
| AUROC of R (high-risk vs rest) | **0.746** | – |

## Per-class (top-1)

| Class | Recall | Precision | F1 | n |
|---|---|---|---|---|
| akiec | 0.000 | 0.000 | 0.000 | 12 |
| bcc | 0.133 | 0.500 | 0.211 | 15 |
| bkl | 0.333 | 0.263 | 0.294 | 15 |
| df | 0.100 | 0.167 | 0.125 | 10 |
| nv | 0.722 | 0.342 | 0.464 | 18 |
| vasc | 0.800 | 1.000 | 0.889 | 10 |
| mel | 0.700 | 0.636 | 0.667 | 20 |

## Confusion matrix (rows = actual, columns = predicted)

| | akiec | bcc | bkl | df | nv | vasc | mel |
|---|---|---|---|---|---|---|---|
| akiec | 0 | 1 | 4 | 2 | 2 | 0 | 3 |
| bcc | 1 | 2 | 3 | 0 | 9 | 0 | 0 |
| bkl | 0 | 0 | 5 | 3 | 6 | 0 | 1 |
| df | 0 | 0 | 5 | 1 | 4 | 0 | 0 |
| nv | 0 | 0 | 1 | 0 | 13 | 0 | 4 |
| vasc | 0 | 1 | 0 | 0 | 1 | 8 | 0 |
| mel | 2 | 0 | 1 | 0 | 3 | 0 | 14 |

Figures: `results/fig19_confusion_matrix.png`, `results/fig20_risk_threshold.png`, `results/fig21_occlusion_heatmaps.png`.

## Findings
1. **Risk flag works as a screen, not as a classifier.** It caught 38 of 47 high-risk lesions (81%) including 18 of 20 melanomas (14 of 20 by top-1 class). Specificity is low (34%): 35 of 53 benign lesions were flagged, so about half of all flags are false alarms.
2. **9 high-risk lesions were missed** (NOT FLAGGED): 4 akiec, 3 bcc, 2 mel. Two of the four akiec misses are SCC in situ. Most had R between 0.17 and 0.23, just under τ = 0.2282. Cases: TC002, TC006, TC009, TC012, TC015, TC024, TC026, TC086, TC094.
3. **akiec and bcc are almost never named correctly** (0/12 and 2/15). bcc is mostly called nv (9/15). The model over-predicts nv (38 predictions for 18 true nv).
4. **Vascular lesions and melanoma are the strongest classes** (F1 0.89 and 0.67).
5. **Mean R by true class**: mel 0.57, bcc 0.37, akiec 0.36, bkl 0.31, nv 0.31, df 0.25, vasc 0.24. Melanoma separates clearly; bcc and akiec sit close to benign classes.

## Limitations (state these in the report)
- n = 100 and some classes have only 10 images, so intervals are wide (accuracy ±10 points).
- The set is deliberately not class-balanced like the real world, so PPV/NPV do not transfer to clinical prevalence.
- ISIC images come from different clinics and devices than HAM10000, so this partly measures domain shift. It is an external test, not a replacement for the held-out HAM10000 test split.
- Overlap with training data cannot be fully ruled out: only HAM10000 IDs were excluded, and lesion-level overlap with other training sources is not verifiable.
- Labels are ISIC metadata (mostly histopathology; nevi are often clinically diagnosed). The class mapping is a judgement call, especially SCC in situ → akiec.
- Run with TensorFlow's `tf.lite.Interpreter` because `ai_edge_litert` would not load on Python 3.9 here. This is the same TFLite runtime but should be confirmed on the deployment interpreter.
- Research prototype result. Not a diagnostic claim.
