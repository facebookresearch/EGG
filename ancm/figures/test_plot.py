import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import seaborn as sns

with open('figures/data/means-noise.json') as fp:
    data = json.load(fp)

rows = defaultdict(list)

metrics = 'accuracy length alignment topographic_rho pos_dis bos_dis'.split()
for metric in data:
    if metric not in metrics:
        continue
    for max_len in sorted(data[metric], key=lambda x: int(x)):
        for erasure_pr in data[metric][max_len]:

            rows['max_len'].append(max_len)
            rows['erasure_pr'].append(erasure_pr)
            rows['metric'].append(metric)
            rows['value'].append(data[metric][max_len][erasure_pr][0])

for row in rows:
    print(row, len(rows[row]), rows[row])

df = pd.DataFrame(rows)
print(df)

plot = sns.FacetGrid(df, col="metric", sharey=False)
plot.map_dataframe(sns.lineplot,
    x="erasure_pr", y="value", 
    errorbar=("se", 2),
    hue="max_len")
plot.add_legend()
plot.savefig("figures/test_plot.png") 


