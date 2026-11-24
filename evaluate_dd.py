"""
evaluate_dd.py -- test-set evaluation for Dhanvantri Drishti (dd_student_int8.tflite)

Uses the SAME preprocessing, temperature scaling and risk rule as your notebook cell
(tf.image.resize bilinear, T = 3.1843, tau = 0.2282, HIGH_RISK = akiec/bcc/mel).

Input : a CSV with two columns  image_path,label   (label = one of CLASS_NAMES)
        built from your held-out TEST split only (split by lesion_id, never by image).
Output (in ./results):
        per_class_metrics.csv   recall / precision / F1 / support per class (+ macro-F1, accuracy)
        risk_flag_metrics.csv   sensitivity, specificity, PPV, NPV, AUROC for the REVIEW ADVISED rule
        fig19_confusion_matrix.png   confusion-matrix heatmap (counts and row-normalised)
        fig20_risk_threshold.png     ROC curve and sensitivity/specificity vs tau
        fig21_occlusion_heatmaps.png occlusion-sensitivity heatmaps for example images

Usage : python evaluate_dd.py test_split.csv [--model dd_student_int8.tflite] [--n_examples 6]
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']   # must match training order
HIGH_RISK   = [0, 1, 6]
T_OPT_INT8, TAU_INT8 = 3.1843, 0.2282

def build_predictor(model_path, fake=False):
    """Returns f(img_float[96,96,3] in [-1,1]) -> (probs[7], logits[7])."""
    if fake:                                    # plumbing test only: NOT the real model
        rng = np.random.default_rng(0)
        Wf = rng.normal(size=(96 * 96 * 3, 7)).astype(np.float32) * 0.02
        def f(img):
            z = img.reshape(-1) @ Wf
            zz = z / T_OPT_INT8; zz -= zz.max(); p = np.exp(zz) / np.exp(zz).sum()
            return p, z
        return f
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=model_path); it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    i_s, i_z = i_d['quantization']; o_s, o_z = o_d['quantization']
    def f(img):
        q = np.round(img / i_s + i_z).clip(-128, 127).astype(np.int8)
        it.set_tensor(i_d['index'], q[None, ...]); it.invoke()
        logits = (it.get_tensor(o_d['index'])[0].astype(np.float32) - o_z) * o_s
        z = logits / T_OPT_INT8; z = z - z.max(); p = np.exp(z) / np.exp(z).sum()
        return p, logits
    return f

def load_image(path, fake=False):
    if fake:
        rng = np.random.default_rng(abs(hash(path)) % (2**32))
        return rng.uniform(-1, 1, (96, 96, 3)).astype(np.float32)
    import tensorflow as tf                      # same resize as training (bilinear, no antialias)
    img = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (96, 96), method='bilinear')
    return tf.cast(img, tf.float32).numpy() / 127.5 - 1.0

def occlusion_map(predict, img, patch=16, stride=8):
    """Drop in risk score R when a patch is replaced by neutral grey (0 after scaling)."""
    base = predict(img)[0][HIGH_RISK].sum()
    n = (96 - patch) // stride + 1
    heat = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            x = img.copy(); x[i*stride:i*stride+patch, j*stride:j*stride+patch, :] = 0.0
            heat[i, j] = base - predict(x)[0][HIGH_RISK].sum()
    return heat, base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('--model', default='dd_student_int8.tflite')
    ap.add_argument('--n_examples', type=int, default=6); ap.add_argument('--out', default='results')
    ap.add_argument('--fake', action='store_true', help='plumbing test with a fake model')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv); predict = build_predictor(a.model, a.fake)
    y = np.array([CLASS_NAMES.index(l) for l in df['label']])
    P = np.zeros((len(df), 7), np.float32); imgs = []
    for k, pth in enumerate(df['image_path']):
        im = load_image(pth, a.fake); P[k] = predict(im)[0]; imgs.append(im)
        if k % 200 == 0: print(f'{k}/{len(df)}', file=sys.stderr)
    yhat = P.argmax(1)
    # ---- per-class metrics
    pr, rc, f1, sup = precision_recall_fscore_support(y, yhat, labels=range(7), zero_division=0)
    pc = pd.DataFrame({'class': CLASS_NAMES, 'recall': rc, 'precision': pr, 'F1': f1, 'support': sup})
    pc.loc[len(pc)] = ['macro / accuracy', rc.mean(), pr.mean(), f1.mean(), int((yhat == y).sum())]
    pc.round(4).to_csv(f'{a.out}/per_class_metrics.csv', index=False); print(pc.round(3).to_string(index=False))
    # ---- confusion matrix heatmap
    cm = confusion_matrix(y, yhat, labels=range(7)); cmn = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a_, M, t, fmt in [(ax[0], cm, 'Counts', 'd'), (ax[1], cmn, 'Row-normalised (recall)', '.2f')]:
        im_ = a_.imshow(M, cmap='Blues'); a_.set_title(t)
        a_.set_xticks(range(7)); a_.set_yticks(range(7))
        a_.set_xticklabels(CLASS_NAMES, rotation=45, ha='right'); a_.set_yticklabels(CLASS_NAMES)
        a_.set_xlabel('Predicted'); a_.set_ylabel('Actual')
        for i in range(7):
            for j in range(7):
                a_.text(j, i, format(M[i, j], fmt), ha='center', va='center', fontsize=7,
                        color='white' if M[i, j] > M.max() * 0.6 else 'black')
    plt.tight_layout(); plt.savefig(f'{a.out}/fig19_confusion_matrix.png', dpi=200); plt.close()
    # ---- risk-flag (REVIEW ADVISED) metrics
    R = P[:, HIGH_RISK].sum(1); yb = np.isin(y, HIGH_RISK).astype(int); flag = R >= TAU_INT8
    tp = int((flag & (yb == 1)).sum()); fn = int((~flag & (yb == 1)).sum())
    fp = int((flag & (yb == 0)).sum()); tn = int((~flag & (yb == 0)).sum())
    rm = pd.DataFrame([{'tau': TAU_INT8, 'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
                        'sensitivity': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1),
                        'PPV': tp / max(tp + fp, 1), 'NPV': tn / max(tn + fn, 1),
                        'AUROC': roc_auc_score(yb, R) if 0 < yb.sum() < len(yb) else float('nan')}])
    rm.round(4).to_csv(f'{a.out}/risk_flag_metrics.csv', index=False); print(rm.round(3).to_string(index=False))
    fpr, tpr, _ = roc_curve(yb, R); taus = np.linspace(0, 1, 201)
    se = [(R[yb == 1] >= t).mean() for t in taus]; sp = [(R[yb == 0] < t).mean() for t in taus]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(fpr, tpr, color='#0B2A5B'); ax[0].plot([0, 1], [0, 1], 'k:', lw=0.8)
    ax[0].plot(1 - rm.specificity[0], rm.sensitivity[0], 'o', color='#C0392B', label=f'tau = {TAU_INT8}')
    ax[0].set_xlabel('1 - specificity'); ax[0].set_ylabel('Sensitivity'); ax[0].legend(frameon=False)
    ax[1].plot(taus, se, color='#C0392B', label='Sensitivity'); ax[1].plot(taus, sp, color='#1B7F7A', label='Specificity')
    ax[1].axvline(TAU_INT8, color='k', ls='--', lw=0.8); ax[1].set_xlabel('Threshold tau on R'); ax[1].legend(frameon=False)
    plt.tight_layout(); plt.savefig(f'{a.out}/fig20_risk_threshold.png', dpi=200); plt.close()
    # ---- occlusion-sensitivity heatmaps (highest-R correctly flagged + missed high-risk + benign examples)
    order = np.argsort(-R); picks = list(order[:a.n_examples // 2]) + list(order[-(a.n_examples - a.n_examples // 2):])
    fig, ax = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.4))
    for c, k in enumerate(picks):
        h, base = occlusion_map(predict, imgs[k]); up = np.kron(h, np.ones((12, 12)))
        ax[0, c].imshow((imgs[k] + 1) / 2); ax[0, c].set_title(f'{CLASS_NAMES[y[k]]}  R={base:.2f}', fontsize=7)
        ax[1, c].imshow((imgs[k] + 1) / 2); v = np.abs(up).max() or 1
        ax[1, c].imshow(up, cmap='jet', alpha=0.45, vmin=-v, vmax=v)
        for r_ in (0, 1): ax[r_, c].axis('off')
    plt.tight_layout(); plt.savefig(f'{a.out}/fig21_occlusion_heatmaps.png', dpi=200); plt.close()
    print('saved to', a.out)

if __name__ == '__main__':
    main()
