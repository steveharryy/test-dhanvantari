# Dhanvantri Drishti: INT8 evaluation

- `dd_student_int8.tflite`, `evaluate_dd.py`: model and evaluation script.
- `dd_test100/`: 100-image external ISIC test (accuracy 0.43, flag sensitivity 0.81, AUROC 0.75). See `dd_test100/RESULTS_SECTION.md`.
- `dd_ham10000_test/`: HAM10000 lesion-level test split, 1,512 images (accuracy 0.76, macro-F1 0.52, flag sensitivity 0.905, specificity 0.656, AUROC 0.867). See `dd_ham10000_test/RESULTS_SECTION.md` and `DD_HAM10000_Test_Results.docx`. The split is my own, not the notebook's, so it may overlap training data.
