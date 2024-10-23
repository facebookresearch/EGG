import json


with open('figures/data/means-noise.json') as fp:
    data = json.load(fp)


key2desc = {
    'accuracy': 'Accuracy',
    'f1': 'F1 (micro)',
    'embedding_alignment': 'Embedding alignment',
    'topographic_rho': 'Topographic $\\rho$',
    'pos_dis': 'PosDis',
    'bos_dis': 'BosDic',
    #'unique_msg_no_noise': 'Unique messages sent',
    #'unique_msg': 'Unique messages received',
}


for metric in data:
    if metric not in key2desc:
        continue
    line = [key2desc[metric]]
    for max_len in data[metric]:
        for noise_level in data[metric][max_len]:
            mean, ci = data[metric][max_len][noise_level]
            line.extend((f"{round(mean, 2):.2f}", f"{round(ci,2):.2f}"))
    print(' & '.join(line) + ' \\\\')


print('n cells:', len(line))
