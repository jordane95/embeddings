import json
import os

model_path = 'ckpt/ft_csn_{}_clean_2a64bs512msl3e2e-5_pt_graphcodebert_so_v3_2a3096bs128msl10k2e-4'
# model_path = 'ckpt/ft_csn_{}_2a64bs512msl3e2e-5_pt_graphcodebert'

langs = ['ruby', 'javascript', 'go', 'python', 'java', 'php']

all_score = []

print("model: ", model_path)

for lang in langs:
    model_lang_path = model_path.format(lang)
    result_path = f'results/csn_hard_clean/{model_lang_path}'
    f = open(os.path.join(result_path, 'avg.json'))
    result = json.load(f)
    score = list(result.values())[0]
    all_score.append(round(score * 100, 1))
    print(f"{lang}: {all_score[-1]}")

avg_score = round(sum(all_score) / len(all_score), 1)
all_score.append(avg_score)
print(f"Avg: {all_score[-1]}")


latex_str = " & ".join(map(str, all_score))
print(latex_str)
