import json
import os

model_path = 'ckpt/ft_{}_2a64bs512msl3e2e-5_pt_graphcodebert_so_dec_csn_2a3096bs128msl10k2e-4'
model_path = 'ckpt/ft_{}_2a64bs512msl3e2e-5_pt_graphcodebert'
model_path = 'ckpt/ft_{}_2a64bs512msl3e2e-5_pt_graphcodebert_so_dec_2a3096bs128msl10k2e-4'

langs = ['Solidity', 'SQL']

all_score = []

print("model: ", model_path)



for lang in langs:
    model_lang_path = model_path.format(lang.lower())
    result_path = f'results/{lang.lower()}/{model_lang_path}'
    f = open(os.path.join(result_path, f'{lang}.json'))
    result = json.load(f)
    score = [result['hit@1'], result['hit@5'], result['hit@10'], result['mrr@1000']]
    all_score.extend([round(s * 100, 1) for s in score])


avg_score = round(sum(all_score) / len(all_score), 1)
# all_score.append(avg_score)
# print(f"Avg: {all_score[-1]}")


latex_str = " & ".join(map(str, all_score))
print(latex_str)
