import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
d=Document(); d.styles['Normal'].font.name='Calibri'; d.styles['Normal'].font.size=Pt(11)
def table(df,widths=None):
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
d.add_heading('Results: External Test on 100 Dermoscopic Images',1)
d.add_heading('1. Test set and method',2)
d.add_paragraph('The deployed INT8 student model (dd_student_int8.tflite) was evaluated on 100 dermoscopic images from the public ISIC Archive (CC-0 licence), one image per lesion, drawn with a fixed random seed. Image IDs in the HAM10000 range were excluded to limit overlap with the training data. ISIC diagnoses were mapped to the seven DD classes. Preprocessing, temperature (T = 3.1843), risk threshold (tau = 0.2282) and the high-risk set (akiec, bcc, mel) are identical to the deployed pipeline.')
table(pd.DataFrame([['Images',12,15,15,10,18,10,20,100]],columns=['','akiec','bcc','bkl','df','nv','vasc','mel','Total']))
d.add_heading('2. Overall performance',2)
table(pd.DataFrame([
 ['7-class top-1 accuracy','0.43 (43/100)','0.33-0.53'],
 ['Macro-F1','0.378','-'],
 ['Review-advised sensitivity','0.809 (38/47)','0.69-0.92'],
 ['Review-advised specificity','0.340 (18/53)','0.22-0.48'],
 ['PPV / NPV','0.521 / 0.667','-'],
 ['AUROC of risk score R','0.746','-']],columns=['Metric','Value','95% CI (bootstrap)']))
d.add_heading('3. Per-class results',2)
pc=pd.read_csv(R+'per_class_metrics.csv').iloc[:7]; pc.columns=['Class','Recall','Precision','F1','Support']; pc['Support']=pc.Support.astype(int); table(pc)
d.add_heading('4. Confusion matrix',2)
fig(R+'report_fig1_confusion_heatmap.png','Figure 1. Confusion matrix (count and row percentage). Most errors are lesions predicted as nv or bkl; akiec and bcc are rarely identified.',5.2)
d.add_heading('5. Per-image probability heatmap',2)
fig(R+'report_fig2_probability_heatmap.png','Figure 2. Predicted class probabilities for each of the 100 images, grouped by true class. Confident predictions appear only for vasc and nv; akiec and bcc rows are diffuse.',4.2)
d.add_heading('6. Risk score and flag behaviour',2)
fig(R+'report_fig4_risk_by_class.png','Figure 3. Risk score R by true class with the threshold tau = 0.2282. Melanoma separates well; bcc and akiec overlap with benign classes.',5.0)
fig(R+'fig20_risk_threshold.png','Figure 4. ROC curve (operating point at tau) and sensitivity/specificity versus threshold.',6.0)
d.add_heading('7. Occlusion-sensitivity maps',2)
fig(R+'report_fig3_occlusion_heatmaps.png','Figure 5. Occlusion sensitivity for true positives, missed cases, false alarms and true negatives. Red marks regions whose masking lowers R; blue marks regions whose masking raises it. The map covers the central region sampled by the 16-pixel patches. In the true positives the model attends to the lesion; the missed akiec cases show a faint, diffuse lesion with no focal evidence.',6.8)
d.add_heading('8. Findings',2)
for s in ['The flag detects 81% of high-risk lesions, including 18 of 20 melanomas, but 66% of benign lesions are also flagged, so it functions as a screening aid.',
 '9 high-risk lesions were missed (4 akiec, 3 bcc, 2 mel); most scored R between 0.17 and 0.23, just under the threshold.',
 'Fine-grained classification is weak: akiec 0/12 and bcc 2/15 correct. The model over-predicts nv (38 predictions for 18 true nv).',
 'Vascular lesions (F1 0.89) and melanoma (F1 0.67) are the strongest classes.']: d.add_paragraph(s,style='List Bullet')
d.add_heading('9. Limitations',2)
for s in ['Small sample (n = 100; 10-20 per class), so confidence intervals are wide (accuracy about +/-10 points).',
 'Class mix is not representative of clinical prevalence, so PPV and NPV do not transfer.',
 'ISIC images come from different sources than HAM10000, so results partly reflect domain shift. This complements, but does not replace, the held-out HAM10000 test split.',
 'Overlap with training data cannot be fully excluded; only the HAM10000 ID range was removed.',
 'Labels follow ISIC metadata and the class mapping (for example SCC in situ to akiec) involves judgement.',
 'Evaluation used the TensorFlow Lite interpreter; confirm on the deployment runtime.',
 'This is a research prototype and the results are not a diagnostic claim.']: d.add_paragraph(s,style='List Bullet')
d.save('DD_External_Test_Results.docx')
