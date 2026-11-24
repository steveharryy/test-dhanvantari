import sys, types, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import tensorflow as tf
sys.argv=['x']; sys.path.insert(0,'/Users/abhaypandit/Downloads/untitled folder')
m=types.ModuleType('ai_edge_litert'); i=types.ModuleType('ai_edge_litert.interpreter'); i.Interpreter=tf.lite.Interpreter; m.interpreter=i
sys.modules['ai_edge_litert']=m; sys.modules['ai_edge_litert.interpreter']=i
import evaluate_dd as E
C=E.CLASS_NAMES; pred=E.build_predictor('../dd_student_int8.tflite')
o=pd.read_csv('results/per_image_predictions.csv'); df=pd.read_csv('test_split.csv')
# 1 confusion heatmap (counts + recall %)
cm=pd.crosstab(o.true_label,o.pred_label).reindex(index=C,columns=C,fill_value=0).values; n=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(6.2,5.2)); im=ax.imshow(n,cmap='Blues',vmin=0,vmax=1)
for r in range(7):
    for c in range(7): ax.text(c,r,f'{cm[r,c]}\n({n[r,c]*100:.0f}%)',ha='center',va='center',fontsize=8,color='white' if n[r,c]>0.5 else 'black')
ax.set_xticks(range(7));ax.set_yticks(range(7));ax.set_xticklabels(C);ax.set_yticklabels([f'{c} (n={cm[k].sum()})' for k,c in enumerate(C)])
ax.set_xlabel('Predicted class');ax.set_ylabel('Actual class');ax.set_title('Confusion matrix, 100 external ISIC images\ncount (row %)')
plt.colorbar(im,fraction=0.046,label='Row-normalised recall');plt.tight_layout();plt.savefig('results/report_fig1_confusion_heatmap.png',dpi=200);plt.close()
# 2 probability heatmap per image, sorted by true class then R
o['k']=range(len(o)); s=o.assign(o_=o.true_label.map(C.index)).sort_values(['o_','R'],ascending=[True,False])
P=s[[f'p_{c}' for c in C]].values
fig,ax=plt.subplots(figsize=(6,10)); im=ax.imshow(P,aspect='auto',cmap='viridis',vmin=0,vmax=1)
ax.set_xticks(range(7));ax.set_xticklabels(C);ax.set_yticks([]);ax.xaxis.tick_top()
start=0
for c in C:
    k=(s.true_label==c).sum()
    if start: ax.axhline(start-0.5,color='white',lw=1.2)
    ax.text(-0.7,start+k/2-0.5,c,ha='right',va='center',fontsize=9); start+=k
ax.set_title('Predicted probability per image (rows grouped by true class)',y=1.05,fontsize=10)
plt.colorbar(im,fraction=0.04,pad=0.02,label='Probability');plt.tight_layout();plt.savefig('results/report_fig2_probability_heatmap.png',dpi=200);plt.close()
# 3 aligned occlusion maps: 2 flagged correctly high R, 2 missed high-risk, 2 false alarms, 2 correct benign
hr=o.truly_high_risk; fl=o.flag=='REVIEW ADVISED'
sel=[('True positive',o[hr&fl].sort_values('R',ascending=False).head(2)),
     ('Missed (FN)',o[hr&~fl].sort_values('R').head(2)),
     ('False alarm (FP)',o[~hr&fl].sort_values('R',ascending=False).head(2)),
     ('True negative',o[~hr&~fl].sort_values('R').head(2))]
rows=[(t,r) for t,d in sel for _,r in d.iterrows()]
fig,ax=plt.subplots(2,8,figsize=(17,4.8))
for c,(t,r) in enumerate(rows):
    img=E.load_image(df.image_path[r.k]); h,base=E.occlusion_map(pred,img); v=np.abs(h).max() or 1
    ax[0,c].imshow((img+1)/2); ax[0,c].set_title(f'{t}\ntrue {r.true_label}, R={base:.2f}',fontsize=8)
    ax[1,c].imshow((img+1)/2); hm=ax[1,c].imshow(h,cmap='jet',alpha=0.5,vmin=-v,vmax=v,extent=(8,88,88,8),interpolation='bilinear')
    ax[1,c].set_xlim(0,96);ax[1,c].set_ylim(96,0)
    for a in ax[:,c]: a.axis('off')
fig.suptitle('Occlusion sensitivity (red = covering this region lowers risk score R; blue = raises it)',fontsize=10)
plt.tight_layout();plt.savefig('results/report_fig3_occlusion_heatmaps.png',dpi=200);plt.close()
# 4 R distribution + threshold
fig,ax=plt.subplots(figsize=(6,3.4)); 
ax.boxplot([o[o.true_label==c].R for c in C],labels=C); ax.axhline(E.TAU_INT8,color='r',ls='--',label=f'tau = {E.TAU_INT8}')
ax.set_ylabel('Risk score R');ax.set_title('Risk score by true class');ax.legend(frameon=False);plt.tight_layout();plt.savefig('results/report_fig4_risk_by_class.png',dpi=200)
