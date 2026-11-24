import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
d=Document(); d.styles['Normal'].font.name='Calibri'; d.styles['Normal'].font.size=Pt(11)
def table(df):
    t=d.add_table(rows=1,cols=len(df.columns)); t.style='Light Grid Accent 1'
    for i,c in enumerate(df.columns): t.rows[0].cells[i].text=str(c)
    for _,r in df.iterrows():
        cs=t.add_row().cells
        for i,v in enumerate(r): cs[i].text=str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs: run.font.size=Pt(9)
def fig(path,cap,w=6.0):
    d.add_picture(path,width=Inches(w)); d.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=d.add_paragraph(cap); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.runs[0].italic=True; p.runs[0].font.size=Pt(9)
R='results/'
d.add_heading('Results: HAM10000 Test Split',1)
d.add_heading('1. Test set and method',2)
d.add_paragraph('HAM10000 was obtained from the ISIC Archive (collection "HAM10000", 10,015 images, 7,470 lesions). A lesion-level split was made with a fixed seed (42): about 15% of lesions in each class were assigned entirely to the test set, so no lesion appears in both sets. This gives 1,512 test images from 1,120 lesions. The deployed INT8 model was evaluated with the unchanged evaluate_dd.py (T = 3.1843, tau = 0.2282, high-risk = akiec, bcc, mel). Confidence intervals use a lesion-clustered bootstrap (1,000 resamples).')
table(pd.DataFrame([['Images',41,81,171,15,1022,21,161,1512]],columns=['','akiec','bcc','bkl','df','nv','vasc','mel','Total']))
d.add_heading('2. Overall performance',2)
table(pd.DataFrame([['7-class top-1 accuracy','0.761 (1150/1512)','0.733-0.787'],['Macro-F1','0.524','0.470-0.575'],
 ['Review-advised sensitivity','0.905 (256/283)','0.866-0.940'],['Review-advised specificity','0.656 (806/1229)','0.624-0.690'],
 ['PPV / NPV','0.377 / 0.968','-'],['AUROC of risk score R','0.867','0.840-0.893']],columns=['Metric','Value','95% CI']))
d.add_heading('3. Per-class results',2)
pc=pd.read_csv(R+'per_class_metrics.csv').iloc[:7]; pc.columns=['Class','Recall','Precision','F1','Support']; pc['Support']=pc.Support.astype(int); table(pc)
d.add_heading('4. Confusion matrix',2)
fig(R+'report_fig1_confusion_heatmap.png','Figure 1. Confusion matrix (count and row percentage).',5.4)
d.add_heading('5. Risk score and flag behaviour',2)
fig(R+'report_fig4_risk_by_class.png','Figure 2. Risk score R by true class; dashed line is tau = 0.2282.',5.2)
fig(R+'fig20_risk_threshold.png','Figure 3. ROC curve (operating point at tau) and sensitivity/specificity versus threshold.',6.2)
d.add_heading('6. Occlusion-sensitivity maps',2)
fig(R+'report_fig3_occlusion_heatmaps.png','Figure 4. Occlusion sensitivity for true positives, missed cases, false alarms and true negatives. Red = masking lowers R; blue = masking raises it. The map covers the central region sampled by the 16-pixel patches.',6.8)
d.add_heading('7. Findings',2)
for s in ['The risk flag detects 90.5% of high-risk lesions (256/283), with NPV 0.968. By class it flagged 142/161 melanomas, 74/81 bcc and 40/41 akiec. Specificity is 65.6% and PPV 0.377: 423 benign lesions were flagged, including 292 of 1,022 nv, so most flags are false alarms.',
 'nv is recognised well (recall 0.914, F1 0.885) and vasc is also good (recall 0.81, n = 21).',
 'Top-1 melanoma recall is only 0.478: 69 of 161 were called nv, although 52 of those were still flagged by the risk score. The risk score is therefore more reliable than the top-1 class for high-risk lesions.',
 'akiec (recall 0.195) and bcc (0.383) are the weakest classes by top-1, mostly confused with mel, bkl and nv, yet the flag catches almost all of them.',
 'Accuracy of 0.761 is lifted by the nv majority (68% of images); macro-F1 (0.524) is the fairer headline.']: d.add_paragraph(s,style='List Bullet')
d.add_heading('8. Limitations',2)
for s in ['This split is not the notebook split. The original test split (seed and lesion grouping) was not available, so images in this test set may have been seen in training. Results may be optimistic and must not be reported as a strict held-out test.',
 'Small classes: df has 15 images and vasc 21, so their per-class numbers are unstable.',
 'HAM10000 is dominated by nv, so accuracy, PPV and NPV reflect that prevalence.',
 'Evaluation used the TensorFlow Lite interpreter, not ai_edge_litert; confirm on the deployment runtime.',
 'Research prototype; not a diagnostic claim.']: d.add_paragraph(s,style='List Bullet')
d.save('DD_HAM10000_Test_Results.docx')
