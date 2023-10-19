import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

plt.rcParams.update({'xtick.direction': 'in'})
plt.rcParams.update({'ytick.direction': 'in'})
plt.rcParams.update({'legend.frameon': True})
#plt.rcParams.update({'grid.color': '#dddddd'})
plt.rcParams.update({'grid.linestyle': '--'})
plt.rcParams.update({'axes.axisbelow': True})
plt.rcParams["font.family"] = "Times New Roman"


FS = 13
MS=15
LW=4
LINESTYLE = ['', '', '-.', ':']
MARKERSTYLE=['o', 'X', '*', 'p', '.']
COLORS = ["xkcd:dark periwinkle", "xkcd:coral",'xkcd:goldenrod', "xkcd:jade green", "xkcd:coffee"]
#COLORS = ["xkcd:bright blue", "xkcd:coral",'xkcd:goldenrod']
legends = []

recall=0

##################
#plt.figure()
fig, axs = plt.subplots(1, sharex=True,figsize=(10, 2.5))


datasets = ["Ruby", "JavaScript", "Go", "Python", "Java", "PHP"]

so_dec = [74.7, 69.8, 91.7, 75.4, 74.9, 69.0]
csn = [72.7, 66.7, 90.8, 72.5, 71.1, 66.9]
so_dec_csn = [77.8, 72.5, 92.4, 76.1, 75.7, 70.1]
no = [73.0, 68.5, 91.2, 73.4, 73.2, 68.3]

idx = [i[0] for i in sorted(enumerate(so_dec_csn), key=lambda x:x[1])]
tmp = [so_dec[i] for i in idx]
so_dec = tmp
tmp = [csn[i] for i in idx]
csn = tmp
tmp = [so_dec_csn[i] for i in idx]
so_dec_csn = tmp
tmp = [datasets[i] for i in idx]
datasets= tmp
tmp = [no[i] for i in idx]
no = tmp


mean = lambda l: round(sum(l)/len(l), 1)

datasets.append("Average")
so_dec.append(mean(so_dec))
csn.append(mean(csn))
so_dec_csn.append(mean(so_dec_csn))
no.append(mean(no))


x = [i for i in range(len(so_dec_csn))]
x[len(x)-1] = x[len(x)-1]+1
w=.25

# NO = {
#   "linestyle": LINESTYLE[0],
#   "marker": MARKERSTYLE[0],
#   "color": COLORS[4],
#   'x': [i for i in x],
#   'recall': no,
#   'label' : 'No',
#   'legend' : 'No',
# }
# legends.append(NO['legend'])
# axs.bar(NO['x'], NO['recall'],
#   width=w,
#   color=NO['color'],
#   label=NO['label'])



CSN = {
  "linestyle": LINESTYLE[0],
  "marker": MARKERSTYLE[0],
  "color": COLORS[3],
  'x': [i + w*1 for i in x],
  'recall': csn,
  'label' : 'CSN',
  'legend' : 'CSN',
}
legends.append(CSN['legend'])
axs.bar(CSN['x'], CSN['recall'],
        width=w,
        color=CSN['color'],
        label=CSN['label'])


ProCQA = {
  "linestyle": LINESTYLE[0],
  "marker": MARKERSTYLE[0],
  "color": COLORS[2],
  'x': [i + w*2 for i in x],
  'recall': so_dec,
  'label' : 'ProCQA',
  'legend' : 'ProCQA',
}
legends.append(ProCQA['legend'])
axs.bar(ProCQA['x'], ProCQA['recall'],
	width=w,
	color=ProCQA['color'],
	label=ProCQA['label'])

# BM25 = {
#   "linestyle": LINESTYLE[1],
#   "marker": MARKERSTYLE[1],
#   "color": COLORS[1],
#   'x': [i+w*2  for i in x],
#   'recall': bm25_recall,
#   'label' : 'bm25',
#   'legend' : 'BM25',
# }
# legends.append(BM25['legend'])
# axs.bar(BM25['x'], BM25['recall'],
# 	width=w,
# 	color=BM25['color'],
# 	label=BM25['label'],
# 	 align='center')

MACL = {
  "linestyle": LINESTYLE[0],
  "marker": MARKERSTYLE[0],
  "color": COLORS[0],
  'x': [i + w*3 for i in x],
  'recall': so_dec_csn,
  'label' : 'Both',
  'legend' : 'Both',
}
legends.append(MACL['legend'])
axs.bar(MACL['x'], MACL['recall'],
	width=w,
	color=MACL['color'],
	label=MACL['label'])


axs.set_ylabel('MRR@1k', fontsize=FS)
axs.legend(legends, fontsize=FS, loc='upper left',bbox_to_anchor=[0,1], handlelength=1, ncol=5)

axs.grid()

xticks = [x[i] + 2*w for i in range(len(x))]
xticks[len(x)-1] -= w
xticklabels = datasets
axs.set_xticks(xticks)
plt.draw()
i=0
for tick in axs.get_xticklabels():
  i = i + 1
  tick.set_rotation(45)
  tick.set_ha('right')
  if i ==len(x)-1:
    break
axs.set_xticklabels(xticklabels, fontsize=FS)

yticks=[0, 50, 100]
axs.set_yticks(yticks)
axs.set_yticklabels([0,50,100],fontsize=FS)

a = axs.get_ygridlines()
#b = a[1]
#b.set_color('#332c13')

axs.set_ylim([0,100])

plt.savefig("data_ablation.pdf", bbox_inches='tight')
