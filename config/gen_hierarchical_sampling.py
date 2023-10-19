import json


source_prefixes = ['stackoverflow', 'codesearchnet', 'thevault', 'jupyter']

source_lines = {
    source_prefix: 0
    for source_prefix in source_prefixes
}

source_to_dataset = {
    source_prefix: []
    for source_prefix in source_prefixes
}


data_cfg = json.load(open('pt_data_config_so_dec_csn_vault_jupyter.json'))


for data in data_cfg:
    prefix = data['name'].split('_')[0]
    source_lines[prefix] += data['lines']
    source_to_dataset[prefix].append(data)

total = 1.0
source_weight = total / len(source_prefixes)

p = 0.5


all_trained_lines = 3096 * 2 * 10000


for source_prefix in source_prefixes:
    source_datasets = source_to_dataset[source_prefix] # list of dict
    all_lines = sum([data['lines'] for data in source_datasets])
    for data in source_datasets:
        data['weight'] = (data['lines'] / all_lines)**p
    # renormalize weights
    all_weights = sum([data['weight'] for data in source_datasets])
    for data in source_datasets:
        data['weight'] /= all_weights
        data['size'] = data['lines']
        data['lines'] = data['weight'] * source_weight
        data['epoch'] = round(all_trained_lines * data['lines'] / data['size'], 1)
    print(f"Source {source_prefix}: weight sum: {sum([data['lines'] for data in source_datasets])}")

new_data_cfg = []
for source_prefix in source_prefixes:
    new_data_cfg += source_to_dataset[source_prefix]


json.dump(new_data_cfg, open('pt_data_config_so_dec_csn_vault_jupyter_hws.json', 'w'), indent=2)
