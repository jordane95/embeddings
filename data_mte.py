import os
import random
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Any
import itertools
import json

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BERRIIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_config: Dict):
        data_path = data_config['data_file']
        train_group_size = data_config['train_group_size']
        f = open(data_path, 'r')
        # self.data_stream = itertools.cycle(f)
        self.data_stream = f
        self.train_group_size = train_group_size
        self.epoch = 0
    
    def verbalize_doc(self, doc: Dict[str, Any]):
        return "{} {}".format(doc.get('title', ''), doc['text']).strip()
    
    def __iter__(self):
        while True:
            try:
                line = next(self.data_stream)
            except StopIteration:
                self.epoch += 1
                self.data_stream = open(data_path, 'r')
            item = json.loads(line)
            instruction, query = item['question'].strip().split(' [SEP] ')
            gold = random.choice(item["positive_ctxs"])
            pos = self.verbalize_doc(gold)
            negs = []
            if len(item["hard_negative_ctxs"]) < self.train_group_size - 1:
                negs.extend([self.verbalize_doc(neg) for neg in item["hard_negative_ctxs"]])
                # pad to train_group_size with random negs from 'negative_ctxs' (whose amount may not be enough)
                random_negs = random.choices(item["negative_ctxs"], k=self.train_group_size - 1 - len(negs))
                negs.extend([self.verbalize_doc(neg) for neg in random_negs])
            else:
                negatives = random.sample(item["hard_negative_ctxs"], k=self.train_group_size - 1)
                negs.extend([self.verbalize_doc(neg) for neg in negatives])

            yield {"query": query, "pos": pos, "negs": negs}


class MSMARCOIterableDataset:
    pass

class NQIterableDataset:
    pass


class NLIIterableDataset:
    pass

class MEDIIterableDataset:
    pass


class StreamDatasetMNKD(torch.utils.data.IterableDataset):
    # in case your finetuning dataset is too large to fit in machine memory
    def __init__(self, data_configs: List[Dict], batch_size: int, coeff: float = 1.0, seed: int = 42):
        self.batch_size = batch_size
        self.all_data_streams = []
        self.data_sizes = []

        ITERABLE_DATASET_CLS = {
            # "MSMARCO": RetrievalDataset,
            "MSMARCO": MSMARCOIterableDataset,
            "NQ": NQIterableDataset,
            "NLI": NLIIterableDataset,
            "MEDI": MEDIIterableDataset,
            "BERRI": BERRIIterableDataset,
        }

        for i, data_config in enumerate(data_configs):
            data_stream = BERRIIterableDataset(data_config)
            self.all_data_streams.append(iter(data_stream))
            data_size = 1
            self.data_sizes.append(data_size)
        
        prob = np.array(list(map(float, self.data_sizes)))
        prob /= prob.sum()
        prob = np.array([p**coeff for p in prob])
        prob /= prob.sum()
        self.prob = prob
    
    def __iter__(self):
        while True:
            # dataset_idx = random.randint(0, len(self.all_data_streams)-1)
            dataset_idx = np.random.choice(range(len(self.prob)), 1, p=self.prob)[0]
            current_data_stream = self.all_data_streams[dataset_idx]
            n = self.batch_size
            while n > 0:
                n -= 1
                yield next(current_data_stream)

