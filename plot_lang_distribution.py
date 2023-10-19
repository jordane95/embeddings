import json

dataset_to_size = {}
with open('config/pt_data_config_so_v3_csn.json') as f:
    data_configs = json.load(f)
    for ds_conf in data_configs:
        dataset_to_size[ds_conf['name']] = ds_conf['lines']


langs = ['ruby', 'go', 'java', 'javascript', 'php', 'python']

data = {}
for lang in langs:
    data[lang] = dataset_to_size[f'stackoverflow_{lang}_question_answer.jsonl']
total = sum(data.values())
for lang in langs:
    data[lang] /= total
print(data)


data = {}
for lang in langs:
    data[lang] = dataset_to_size[f'codesearchnet_{lang}_docstring_code.jsonl']
total = sum(data.values())
for lang in langs:
    data[lang] /= total
print(data)

import numpy as np
import matplotlib.pyplot as plt

def draw_polar_fig(keys, values):
    N = len(keys)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    bars = ax.bar(theta, values, width=0, bottom=0)

    for i in range(N):
        bar1 = bars[i]
        bar2 = bars[(i + 1) % N]
        ax.plot([bar1.get_x() + bar1.get_width() / 2, bar2.get_x() + bar2.get_width() / 2],
                [bar1.get_height(), bar2.get_height()], color='black')
        ax.fill_between([bar1.get_x() + bar1.get_width() / 2, bar2.get_x() + bar2.get_width() / 2],
                        [0, 0], [bar1.get_height(), bar2.get_height()], color='lightblue', alpha=0.3)

    plt.thetagrids(np.degrees(theta), labels=keys)
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')
    # add legend relative to top-left plot, named 'so'
    ax.legend(bars, ['codesearchnet'], bbox_to_anchor=(0, 1.1), loc='upper left')
    plt.show()
    plt.savefig('lang_distribution_csn.png')

# data = {"A":10, "B":30, "C":50, "D":100, "E":80, "F":40}
draw_polar_fig(data.keys(), data.values())