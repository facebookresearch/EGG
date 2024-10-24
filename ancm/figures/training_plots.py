import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import seaborn as sns
import style


confidence = 0.95


all_dfs = defaultdict(list)
for d in os.listdir('runs/'):
    directory = os.path.join('runs', d)
    if not d.startswith('erasure') or not os.path.isdir(directory):
        continue
    erasure_pr = float(d.strip('erasure_pr_'))

    for file in os.listdir(directory):
        if file.endswith('csv'):
            max_len, seed = (int(item) for item in file[:file.index('-')].split('_'))
        else:
            continue

        fpath = os.path.join(directory, file)
        df = pd.read_csv(fpath)
        if erasure_pr != 0.:
            df = df[df.phase == 'val (no noise)']
        else:
            df = df[df.phase == 'val']
        df = df.reset_index(drop=True)
        df['accuracy'] = df['accuracy'] / 100
        all_dfs[(max_len, erasure_pr)].append(df)


metrics = 'accuracy length alignment topographic_rho pos_dis bos_dis'.split()

comp_long = defaultdict(list)
acc_long = defaultdict(list)

for max_len, erasure_pr in all_dfs:
    for i in range(len(all_dfs[(max_len, erasure_pr)][0])):
        for metric in metrics:
            vals = [df[metric][i] for df in all_dfs[(max_len, erasure_pr)]]
            vals = [v for v in vals if v == v]
            for v in vals:
                if metric in 'topographic_rho pos_dis bos_dis'.split():
                    comp_long['epoch'].append(i+1)
                    comp_long['max_len'].append(max_len)
                    comp_long['erasure_pr'].append(erasure_pr)
                    comp_long['metric'].append(metric)
                    comp_long['value'].append(v)
                if metric == 'accuracy':
                    acc_long['epoch'].append(i+1)
                    acc_long['erasure_pr'].append(erasure_pr)
                    acc_long['max_len'].append(max_len)
                    acc_long['metric'].append('accuracy')
                    acc_long['accuracy'].append(v)

            # mean = np.mean(vals) 
            # stde = stats.sem(vals)
            # ci = stde * stats.t.ppf((1+confidence) / 2., len(vals)-1)

hue_order = ['topographic_rho', 'pos_dis', 'bos_dis', 'accuracy']
sns.set_palette(sns.color_palette("husl", 4))
plot = sns.relplot(
    data=comp_long, row='max_len', col='erasure_pr',
    x='epoch', y='value', hue='metric', kind='line', errorbar=None, facet_kws=dict(margin_titles=True, legend_out=True))
plot.set_titles("{col_name}")
plot.savefig("figures/training_compositionality.png") 

sns.set_palette(sns.husl_palette(4, 0.76))
acc_plot = sns.relplot(legend=True,
    data=acc_long, row='max_len', col='erasure_pr', x='epoch', y='accuracy', kind='line', errorbar=('se',2),facet_kws=dict(margin_titles=True))
acc_plot.set_titles("{col_name}")
acc_plot.savefig("figures/training_accuracy.png") 

# sns.set_palette(sns.hls_palette(3, h=0.2))

