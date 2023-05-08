import os
import random
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import numpy as np
import torch

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.utils import is_datasets_available
import torch.distributed as dist

from utils import normalize_instruction

logger = logging.getLogger(__name__)


QUERY_KEY = "query"
DOC_KEY = "doc"


class InfiniteIterableDataset:
    def __init__(
        self,
        data_path: str,
        column_mapping: Dict[str, str] = {},
        seed: int = 42,
        buffer_size: int = 10000,
    ):
        self.data_path = data_path
        self.epoch = 0
        self.buffer_size = buffer_size

        self.dataset = load_dataset(
            'json',
            data_files=self.data_path,
            split='train',
            streaming=True,
        ).rename_columns(column_mapping)
        self.shuffled_dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)

    def __iter__(self):
        while True:
            self.epoch += 1
            logger.info(f"Current epoch for {self.data_path}: {self.epoch}")
            self.shuffled_dataset.set_epoch(self.epoch)
            for data in self.shuffled_dataset:
                yield data


class InfiniteMultipleIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        train_dir,
        data_config,
        batch_size: int,
        query_field: str,
        doc_field: str,
        coeff: float = 0.0, # equally sample from each datasource
    ):
        self.batch_size = batch_size
        self.all_data_streams = []
        self.data_sizes = []

        self.column_mapping = {}
        if query_field != QUERY_KEY:
            self.column_mapping[query_field] = QUERY_KEY
        if doc_field != DOC_KEY:
            self.column_mapping[doc_field] = DOC_KEY

        for i, data_info in enumerate(data_config):
            data_path = os.path.join(train_dir, data_info["name"])
            data_size = int(data_info["lines"])

            iterable_dataset = InfiniteIterableDataset(data_path=data_path, column_mapping=self.column_mapping)
            self.all_data_streams.append(iter(iterable_dataset))
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


@dataclass
class QDCollator(DataCollatorWithPadding):
    max_q_len: int = 32
    max_d_len: int = 128
    with_prompt: bool = False
    with_instruction: bool = False

    input_keys = [QUERY_KEY, DOC_KEY]

    def __post_init__(self):
        assert not (self.with_prompt and self.with_instruction), "Cannot add prompt and instruction in the same time."

    def __call__(self, features):
        collated_batch = {}

        for key in self.input_keys:
            texts: Union[List[str], List[List[str]]] = [f[key] for f in features]
            # print(text)
            if self.with_instruction: # add instruction
                assert isinstance(texts[0], list), "No instruction in input text."
                instructions = [normalize_instruction(text[0]) for text in texts]
                texts = ['{}: {}'.format(instruction, text[1]) for instruction, text in zip(instructions, texts)]
                instruction_mask = self.tokenizer(
                    instructions,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_d_len if key == DOC_KEY else self.max_q_len,
                    return_tensors='pt',
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    return_attention_mask=True,
                )['attention_mask'] # Tensor shape (batch_size, max_seq_len)
                # instruction_mask[:, 0] = 0 # unmask cls tokens # commented out since this only works for bert-family models
            else: # do not add instruction
                if isinstance(texts[0], list): # if input format is [instruction, text] with instruction
                    texts = [text[1] for text in texts] # List[str]
                if self.with_prompt: # if add simple prompt
                    texts = ['{}: {}'.format(key, text) for text in texts]

            text_batch = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_d_len if key == DOC_KEY else self.max_q_len,
                return_tensors="pt",
            )
            if self.with_instruction:
                text_batch["pooling_mask"] = (~(instruction_mask.bool()) & text_batch["attention_mask"].bool())
            else:
                text_batch["pooling_mask"] = text_batch["attention_mask"]
            collated_batch[key] = text_batch

        return collated_batch
