import yaml


langs = ['ruby', 'javascript', 'go', 'python', 'java', 'php']

for lang in langs:
    data_config = {
        "name": "CodeSearchNet",
        "data_path": "/data01/lizehan/proqa/CodeBERT/GraphCodeBERT/codesearch/dataset/{}/train.jsonl",
        "langs": [lang],
        "train_group_size": 8,
    }
    # with open(f'ft_data_config_csn_{lang}.yaml', 'w') as f:
    #     yaml.dump([data_config], f, default_flow_style=False)
    
    data_config['name'] = 'CodeSearchNetRemove'
    with open(f'ft_data_config_csn_{lang}_remove.yaml', 'w') as f:
        yaml.dump([data_config], f, default_flow_style=False)
    
    data_config['name'] = 'CodeSearchNetToken'
    with open(f'ft_data_config_csn_{lang}_token.yaml', 'w') as f:
        yaml.dump([data_config], f, default_flow_style=False)
    
    data_config['name'] = 'CodeSearchNetClean'
    with open(f'ft_data_config_csn_{lang}_clean.yaml', 'w') as f:
        yaml.dump([data_config], f, default_flow_style=False)
    
    data_config['name'] = 'CodeSearchNetRemoveDoc'
    with open(f'ft_data_config_csn_{lang}_remove_doc.yaml', 'w') as f:
        yaml.dump([data_config], f, default_flow_style=False)
    



