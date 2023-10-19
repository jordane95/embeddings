import json
import os

# model_path = 'ckpt/ft_{}_2a64bs512msl3e2e-5_pt_graphcodebert_so_v3_2a3096bs128msl10k2e-4'
# model_path = 'ckpt/pt_graphcodebert_so_dec_csn_2a3096bs128msl10k2e-4/checkpoint-10000'
model_path = 'ckpt/pt_graphcodebert_so_dec_2a3096bs128msl10k2e-4/checkpoint-10000'

langs = ['ruby', 'python', 'java']

all_score = []

print("model: ", model_path)


model_lang_path = model_path
result_path = f'results/codenet_clean/{model_lang_path}'

for src_lang in langs:
    for tgt_lang in langs:
        f = open(os.path.join(result_path, f'{src_lang}_{tgt_lang}.json'))
        result = json.load(f)
        score = list(result.values())[0]
        all_score.append(round(score * 100, 2))
        print(f"{src_lang}-{tgt_lang}: {all_score[-1]}")

avg_score = round(sum(all_score) / len(all_score), 2)
all_score.append(avg_score)
print(f"Avg: {all_score[-1]}")


latex_str = " & ".join(map(str, all_score))
print(latex_str)
