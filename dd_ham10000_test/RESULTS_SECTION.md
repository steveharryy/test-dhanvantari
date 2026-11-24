# Results: HAM10000 test split (Dhanvantri Drishti INT8)

**Important:** this is a lesion-level split I generated (seed 42, ~15% of lesions per class), not the notebook's original split, which was not available. Some test images may have been in training, so treat the numbers as possibly optimistic.

## Test set
HAM10000 from the ISIC Archive: 10,015 images, 7,470 lesions. Test split: 1,512 images from 1,120 lesions.

| Class | akiec | bcc | bkl | df | nv | vasc | mel | Total |
|---|---|---|---|---|---|---|---|---|
| Images | 41 | 81 | 171 | 15 | 1022 | 21 | 161 | 1512 |

## Overall (95% CI from lesion-clustered bootstrap)
| Metric | Value | 95% CI |
|---|---|---|
| Top-1 accuracy | 0.761 (1150/1512) | 0.733-0.787 |
| Macro-F1 | 0.524 | 0.470-0.575 |
| Review-advised sensitivity | 0.905 (256/283) | 0.866-0.940 |
| Review-advised specificity | 0.656 (806/1229) | 0.624-0.690 |
| PPV / NPV | 0.377 / 0.968 | - |
| AUROC of R | 0.867 | 0.840-0.893 |

## Per class
| Class | Recall | Precision | F1 | n |
|---|---|---|---|---|
| akiec | 0.195 | 0.471 | 0.276 | 41 |
| bcc | 0.383 | 0.756 | 0.508 | 81 |
| bkl | 0.439 | 0.647 | 0.523 | 171 |
| df | 0.533 | 0.258 | 0.348 | 15 |
| nv | 0.914 | 0.858 | 0.885 | 1022 |
| vasc | 0.810 | 0.607 | 0.694 | 21 |
| mel | 0.478 | 0.403 | 0.438 | 161 |

## Findings
- The flag catches 90.5% of high-risk lesions: 142/161 mel, 74/81 bcc, 40/41 akiec. NPV is 0.968.
- 423 benign lesions were flagged (292 of 1,022 nv), so PPV is only 0.377.
- Top-1 melanoma recall is 0.478 (69 called nv), yet 52 of those were still flagged by R.
- Accuracy is lifted by nv (68% of images); macro-F1 is the fairer headline.
- Compared with the 100-image external ISIC test (accuracy 0.43, sensitivity 0.81, specificity 0.34, AUROC 0.75), performance here is much higher. That gap fits domain shift and possible training overlap.

## Limitations
Split not the original; df (15) and vasc (21) are tiny; HAM10000 is nv-dominated; run with `tf.lite.Interpreter`; research prototype only.
