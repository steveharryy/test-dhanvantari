import sys, types, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import tensorflow as tf
sys.argv=['x']; sys.path.insert(0,'/Users/abhaypandit/Downloads/untitled folder')
m=types.ModuleType('ai_edge_litert'); i=types.ModuleType('ai_edge_litert.interpreter'); i.Interpreter=tf.lite.Interpreter; m.interpreter=i
sys.modules['ai_edge_litert']=m; sys.modules['ai_edge_litert.interpreter']=i
import evaluate_dd as E
C=E.CLASS_NAMES; pred=E.build_predictor('../dd_student_int8.tflite'); df=pd.read_csv('test_split.csv'); rows=[]
for k,r in df.iterrows():
    p,_=pred(E.load_image(r.image_path)); R=p[E.HIGH_RISK].sum(); top=int(p.argmax()); hr=C.index(r.label) in E.HIGH_RISK
    rows.append({'k':k,'isic_id':r.isic_id,'lesion_id':r.lesion_id,'true_label':r.label,'pred_label':C[top],'top1_prob':round(float(p[top]),4),
      'R':round(float(R),4),'flag':'REVIEW ADVISED' if R>=E.TAU_INT8 else 'NOT FLAGGED','top1_correct':C[top]==r.label,'truly_high_risk':hr,
      **{f'p_{c}':round(float(x),4) for c,x in zip(C,p)}})
o=pd.DataFrame(rows); o.to_csv('results/per_image_predictions.csv',index=False)
# lesion-clustered bootstrap
les=o.lesion_id.values; ul=np.unique(les); idx={l:np.where(les==l)[0] for l in ul}
y=o.truly_high_risk.values; f=(o.flag=='REVIEW ADVISED').values; c=o.top1_correct.values; Rv=o.R.values
from sklearn.metrics import roc_auc_score, f1_score
yt=o.true_label.map(C.index).values; yp=o.pred_label.map(C.index).values
rng=np.random.default_rng(0); B=[]
for _ in range(1000):
    s=np.concatenate([idx[l] for l in rng.choice(ul,len(ul))])
    B.append([c[s].mean(),f1_score(yt[s],yp[s],average='macro',labels=range(7),zero_division=0),
      (f[s]&y[s]).sum()/max(y[s].sum(),1),(~f[s]&~y[s]).sum()/max((~y[s]).sum(),1),
      roc_auc_score(y[s],Rv[s]) if 0<y[s].sum()<len(s) else np.nan])
B=np.nanpercentile(B,[2.5,97.5],axis=0).round(3)
print('CI acc,macroF1,sens,spec,auc',B.T.tolist()); print('acc',c.mean(),'n',len(o))
cm=pd.crosstab(o.true_label,o.pred_label).reindex(index=C,columns=C,fill_value=0); print(cm)
print(o.groupby('true_label').R.mean().round(3).to_dict())
# figs
n=cm.values/cm.values.sum(1,keepdims=True); fig,ax=plt.subplots(figsize=(6.4,5.3)); im=ax.imshow(n,cmap='Blues',vmin=0,vmax=1)
for r in range(7):
    for q in range(7): ax.text(q,r,f'{cm.values[r,q]}\n({n[r,q]*100:.0f}%)',ha='center',va='center',fontsize=8,color='white' if n[r,q]>0.5 else 'black')
ax.set_xticks(range(7));ax.set_yticks(range(7));ax.set_xticklabels(C);ax.set_yticklabels([f'{k} (n={cm.values[j].sum()})' for j,k in enumerate(C)])
ax.set_xlabel('Predicted class');ax.set_ylabel('Actual class');ax.set_title('HAM10000 test split (1,512 images)\ncount (row %)')
plt.colorbar(im,fraction=0.046,label='Row-normalised recall');plt.tight_layout();plt.savefig('results/report_fig1_confusion_heatmap.png',dpi=200);plt.close()
fig,ax=plt.subplots(figsize=(6.4,3.6)); ax.boxplot([o[o.true_label==k].R for k in C],labels=C,flierprops=dict(markersize=2))
ax.axhline(E.TAU_INT8,color='r',ls='--',label=f'tau = {E.TAU_INT8}'); ax.set_ylabel('Risk score R');ax.set_title('Risk score by true class');ax.legend(frameon=False)
plt.tight_layout();plt.savefig('results/report_fig4_risk_by_class.png',dpi=200);plt.close()
hr=o.truly_high_risk; fl=o.flag=='REVIEW ADVISED'
sel=[('True positive',o[hr&fl].sort_values('R',ascending=False).head(2)),('Missed (FN)',o[hr&~fl].sort_values('R').head(2)),
     ('False alarm (FP)',o[~hr&fl].sort_values('R',ascending=False).head(2)),('True negative',o[~hr&~fl].sort_values('R').head(2))]
rws=[(t,r) for t,d in sel for _,r in d.iterrows()]; fig,ax=plt.subplots(2,8,figsize=(17,4.8))
for cc,(t,r) in enumerate(rws):
    img=E.load_image(df.image_path[r.k]); h,base=E.occlusion_map(pred,img); v=np.abs(h).max() or 1
    ax[0,cc].imshow((img+1)/2); ax[0,cc].set_title(f'{t}\ntrue {r.true_label}, R={base:.2f}',fontsize=8)
    ax[1,cc].imshow((img+1)/2); ax[1,cc].imshow(h,cmap='jet',alpha=0.5,vmin=-v,vmax=v,extent=(8,88,88,8),interpolation='bilinear')
    ax[1,cc].set_xlim(0,96);ax[1,cc].set_ylim(96,0)
    for a in ax[:,cc]: a.axis('off')
fig.suptitle('Occlusion sensitivity (red = covering this region lowers R; blue = raises it)',fontsize=10)
plt.tight_layout();plt.savefig('results/report_fig3_occlusion_heatmaps.png',dpi=200);plt.close()
