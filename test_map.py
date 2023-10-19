
from datasets import load_dataset


ds = load_dataset('json', data_files='/data01/lizehan/proqa/raw_data/so-ds/so-ds-feb20.jsonl', split='train', cache_dir='cache')


print(ds[0])
def normalize(item):
    item['code'] = item['code'].replace('\n', ' ').replace('\t', ' ')
    return item


ds = ds.map(normalize)

print(ds[0])


