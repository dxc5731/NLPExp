"""Only reads archived CSV/JSON; never imports project ML code or runs inference."""
from pathlib import Path
import sys,json,csv
import argparse
BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True,help='New, empty output directory; archived figures are not overwritten')
parser.add_argument('--font',default='Microsoft YaHei')
args=parser.parse_args()
OUTPUT=args.output_dir.resolve()
OUTPUT.mkdir(parents=True,exist_ok=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
plt.rcParams.update({'font.family':args.font,'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'figure.facecolor':'white'})
colors=['#526b83','#ac7540','#28755b','#865e9b']
rows=list(csv.DictReader((BASE/'figures/epoch_data.csv').open(encoding='utf-8-sig')))
groups=[[{k:(float(v) if k not in ['run_id'] and v else v) for k,v in x.items()} for x in rows if int(x['round'])==r] for r in range(1,5)]
evidence=json.loads((BASE/'figures/plot_evidence.json').read_text(encoding='utf-8'));best=evidence['best']
names=['R1 初始基准','R2 Dropout 0.2','R3 LR 1e-5','R4 边界辅助']
def save(name):
 plt.savefig(OUTPUT/f'{name}.png',bbox_inches='tight');plt.savefig(OUTPUT/f'{name}.svg',bbox_inches='tight');plt.close()
fig,axs=plt.subplots(1,2,figsize=(8.2,3.0))
for r,g in enumerate(groups):
 for ax,key in zip(axs,['train_loss','val_loss']):ax.plot([x['epoch'] for x in g],[x[key] for x in g],marker='o',markersize=4,color=colors[r],label=names[r]);ax.set_xticks(range(1,6));ax.set_xlabel('Epoch');ax.grid(alpha=.18)
axs[0].set_ylabel('样本平均 CRF 序列 NLL');axs[0].set_title('内部 train');axs[1].set_title('内部 val');axs[1].legend(fontsize=8)
fig.tight_layout();save('crf_loss')
fig,ax=plt.subplots(figsize=(8.2,3.1))
for r,g in enumerate(groups):
 ax.plot([x['epoch'] for x in g],[100*x['val_entity_f1'] for x in g],marker='o',color=colors[r],label=names[r]);b=best[r]['metrics'];ep=b['best_epoch'];ax.scatter([ep],[b['best_metric']*100],marker='*',s=130,color=colors[r],zorder=4)
ax.set(xticks=range(1,6),xlabel='Epoch',ylabel='严格实体 Micro-F1（%）');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.18);fig.tight_layout();save('crf_val_f1')
fig,axs=plt.subplots(1,2,figsize=(8.2,3.0),gridspec_kw={'width_ratios':[1,1.15]})
for r,b in enumerate(best):
 m=b['metrics']['validation']['entity']['micro'];axs[0].barh(names[r],100*m['f1'],color=colors[r]);axs[0].text(100*m['f1']+.01,r,f"{100*m['f1']:.4f}",va='center',fontsize=9)
axs[0].invert_yaxis();axs[0].set(xlim=(89.4,90.12),xlabel='内部 val Micro-F1（%）');axs[0].grid(axis='x',alpha=.15)
for j,r in enumerate([2,3]):
 m=best[r]['metrics']['validation']['entity']['micro'];v=[100*m[k] for k in ['precision','recall','f1']];xs=[i+(j-.5)*.32 for i in range(3)];bars=axs[1].bar(xs,v,width=.32,color=colors[r],label='B1 / R3' if r==2 else 'B2 / R4')
 for x,y in zip(xs,v):axs[1].text(x,y+.045,f'{y:.2f}',ha='center',fontsize=8,rotation=0)
axs[1].set(xticks=range(3),xticklabels=['Precision','Recall','Micro-F1'],ylim=(88.9,91.0),ylabel='%');axs[1].legend(fontsize=8);axs[1].grid(axis='y',alpha=.15);fig.tight_layout();save('best_and_ablation')
fig,axs=plt.subplots(1,2,figsize=(8.2,2.8));g=groups[3]
for ax,keys,title in [(axs[0],['train_boundary_loss','val_boundary_loss'],'R4 边界辅助损失'),(axs[1],['train_total_loss','val_total_loss'],'R4 联合目标')]:
 for key,lab,c in zip(keys,['train','val'],[colors[2],colors[3]]):ax.plot(range(1,6),[x[key] for x in g],marker='o',label=lab,color=c)
 ax.set(title=title,xlabel='Epoch',xticks=range(1,6));ax.legend(fontsize=8);ax.grid(alpha=.18)
fig.tight_layout();save('auxiliary_loss')
manifest=evidence['dev']
fig,ax=plt.subplots(figsize=(8.2,2.8))
for j,(key,label,col) in enumerate([('rule','规则 mc10',colors[0]),('softmax','BERT-Softmax',colors[1]),('crf','BERT-CRF E5',colors[2])]):
 m=manifest['methods'][key]['metrics'];ys=[100*m[k] for k in ['precision','recall','f1']];xs=[i+(j-1)*.23 for i in range(3)];ax.bar(xs,ys,width=.23,label=label,color=col)
 for x,y in zip(xs,ys):ax.text(x,y+.3,f'{y:.2f}',ha='center',fontsize=8)
ax.set(xticks=range(3),xticklabels=['Precision','Recall','Micro-F1'],ylim=(0,100),ylabel='官方 dev（%）');ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=8);ax.grid(axis='y',alpha=.15);fig.tight_layout();save('official_dev')
fig,ax=plt.subplots(figsize=(8.2,2.7));ax.set_xlim(0,4);ax.set_ylim(0,1);ax.axis('off')
dates=['10-20 18:54\n→21:51 恢复','10-21 00:48','10-21 19:57','10-21 23:17']
labels=['初始基准\nLR 2e-5 / Dropout .1\nE5：89.6837%','增大输出头 Dropout\nLR 2e-5 / Dropout .2\nE3：89.8623%','减小学习率\nLR 1e-5 / Dropout .1\nE5：89.9902%','边界辅助 λ=.2\nLR 1e-5 / Dropout .1\nE3：89.9380%']
for i in range(4):
 ax.scatter(i+.5,.52,s=80,color=colors[i]);ax.text(i+.5,.86,f'R{i+1}\n{dates[i]}',ha='center',va='top',fontsize=9);ax.text(i+.5,.36,labels[i],ha='center',va='top',fontsize=8.8)
 if i<3:ax.annotate('',xy=(i+1.38,.52),xytext=(i+.62,.52),arrowprops={'arrowstyle':'->','color':'#999999'})
save('timeline')
fig,ax=plt.subplots(figsize=(8.2,2.6));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3)
boxes=[(.05,1.55,1.45,'原始字符串\n字符 BIO 金标'),(2,1.55,1.65,'BERT 词片编码\n[B,T,768]'),(4.18,1.55,1.65,'offset 唯一映射\n[B,C,768]'),(6.35,1.55,1.45,'Dropout + Linear\n[B,C,11]'),(8.25,1.55,1.65,'严格 BIO CRF\nViterbi 路径')]
for x,y,w,label in boxes:ax.add_patch(Rectangle((x,y),w,.9,facecolor='#eef2f4',edgecolor='#7b8a96',linewidth=.8));ax.text(x+w/2,y+.45,label,ha='center',va='center',fontsize=9)
for i in range(4):ax.annotate('',xy=(boxes[i+1][0]-.05,2),xytext=(boxes[i][0]+boxes[i][2]+.04,2),arrowprops={'arrowstyle':'->','color':'#666'})
ax.text(5,.65,'字符对齐：词片 [s,e) 的隐藏向量复制到原字符 s…e−1；每个字符必须恰好被覆盖一次。\nCLS / SEP / PAD 不进入 CRF；有效字符 Mask 与原句长度一致。',ha='center',va='center',fontsize=9)
save('architecture')
print('Generated seven figures from archived data; no prediction or training executed.')
