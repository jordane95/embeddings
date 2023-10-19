import json
import os

model_path = 'ckpt/ft_{}_2a64bs512msl3e2e-5_pt_graphcodebert_so_dec_csn_vault_jupyter_2a3096bs128msl10k2e-4'
# model_path = 'ckpt/ft_{}_2a64bs512msl3e2e-5_pt_graphcodebert'
model_path = 'ckpt/ft_{}_2a64bs512msl3e2e-5_pt_graphcodebert_csn_2a3096bs128msl10k2e-4'

langs = ['adv_token', 'cosqa', 'conala', 'so-ds', 'staqc']

all_score = []

print("model: ", model_path)



for lang in langs:
    model_lang_path = model_path.format(lang)
    result_path = f'results/{lang}/{model_lang_path}'
    if lang in ['conala', 'staqc', 'so-ds']:
        result_path = f'results/{lang}_concat/{model_lang_path}'
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
