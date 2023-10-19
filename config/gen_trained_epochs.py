import json


p = 0.5
all_trained_lines = 3096 * 2 * 10000

data_cfg = json.load(open('pt_data_config_so_dec_csn_vault_jupyter.json'))

total_lines = sum([data['lines'] for data in data_cfg])

for data in data_cfg:
    data['weight'] = (data['lines'] / total_lines)**p

# renormalize weights
all_weights = sum([data['weight'] for data in data_cfg])
for data in data_cfg:
    data['weight'] /= all_weights
    data['epoch'] = round(all_trained_lines * data['weight'] / data['lines'], 1)

json.dump(data_cfg, open('pt_data_config_so_dec_csn_vault_jupyter_trained_epoches.json', 'w'), indent=2)

