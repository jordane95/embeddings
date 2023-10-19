import json
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass
import random

import numpy as np

import torch
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from dataset import RetrievalDataset, NLIDataset, NQDataset, MEDIDataset, BERRIDataset, CodeSearchNetDataset
from dataset import CoSQADataset, CoNaLaDataset, AdvDataset, StaQCDataset, SODSDataset, POJ104Dataset
from dataset import load_berri_data, load_medi_data, load_code_search_net_data

from dataset import CodeSearchNetRemoveDataset, AdvRemoveDataset
from dataset import CodeSearchNetTokenDataset, AdvTokenDataset
from dataset import CodeSearchNetCleanDataset, AdvCleanDataset

from dataset import AdvRemoveDocDataset, CodeSearchNetRemoveDocDataset

from dataset import SolidityDataset, SQLDataset

from utils import normalize_instruction


QUERY_KEY = "query"
DOC_KEY = "doc"



DATASET_CLS = {
    "MSMARCO": RetrievalDataset,
    "NQ": NQDataset,
    "NLI": NLIDataset,
    "MEDI": MEDIDataset,
    "BERRI": BERRIDataset,
    "CodeSearchNet": CodeSearchNetDataset,
    "CoNaLa": CoNaLaDataset,
    "Adv": AdvDataset,
    "SO-DS": SODSDataset,
    "CoSQA": CoSQADataset,
    "StaQC": StaQCDataset,
    "POJ104": POJ104Dataset,
    "AdvRemove": AdvRemoveDataset,
    "CodeSearchNetRemove": CodeSearchNetRemoveDataset,
    "AdvToken": AdvTokenDataset,
    "CodeSearchNetToken": CodeSearchNetTokenDataset,
    "AdvClean": AdvCleanDataset,
    "CodeSearchNetClean": CodeSearchNetCleanDataset,
    "AdvRemoveDoc": AdvRemoveDocDataset,
    "CodeSearchNetRemoveDoc": CodeSearchNetRemoveDocDataset,
    "Solidity": SolidityDataset,
    "SQL": SQLDataset,
}

LOADER_FUNC = {
    "MEDI": load_medi_data,
    "BERRI": load_berri_data,
    "CodeSearchNet": load_code_search_net_data,
    "CodeSearchNetToken": load_code_search_net_data,
    "CodeSearchNetRemove": load_code_search_net_data,
    "CodeSearchNetClean": load_code_search_net_data,
    "CodeSearchNetRemoveDoc": load_code_search_net_data,
}

class MultiDatasetMNKD(torch.utils.data.Dataset):
    def __init__(
        self,
        data_configs: List[Dict],
        batch_size: int,
    ):
        self.task_to_dataset: Dict[str, Any] = {}
        for data_config in data_configs:
            task_name = data_config["name"]
            if task_name not in LOADER_FUNC:
                self.task_to_dataset[task_name] = DATASET_CLS[task_name](data_config)
            else:
                multi_data = LOADER_FUNC[task_name](data_config) # Dict[str, List[Dict]]
                self.task_to_dataset.update({
                    "{}-{}".format(task_name, task): DATASET_CLS[task_name](data, train_group_size=data_config['train_group_size'], task=task)
                    for task, data in multi_data.items()
                })

        self.batch_size = batch_size

        self.task_to_datasize: Dict[str, int] = {task: len(task_data) for task, task_data in self.task_to_dataset.items()} 

        self.task_data_idxs = self.batched_shuffle(self.task_to_datasize, self.batch_size) # List[Dict[str, Any]]
        # only shuffle data indice, i.e., task name and local data idx in this task
    
    def __len__(self):
        return len(self.task_data_idxs)
    
    def shuffle_batch(self):
        """Shuld be called at the begin of each epoch"""
        self.task_data_idxs = self.batched_shuffle(self.task_to_datasize, self.batch_size)
    
    @staticmethod
    def batched_shuffle(task_to_datasize: Dict[str, int], batch_size: int) -> List[Dict[str, Any]]:
        task_idxs_batches = [] # List[Dict[str, Any]], list of batches, each batch is a dict with task name and in task batched idxs
        for task, data_size in task_to_datasize.items():
            shuffled_idxs = np.random.permutation(data_size)
            local_batched_shuffled_idxs = [shuffled_idxs[i:i+batch_size] for i in range(0, data_size, batch_size)]
            if len(local_batched_shuffled_idxs[-1]) < batch_size:
                local_batched_shuffled_idxs.pop()
            task_idxs_batches.extend([{"task_name": task, "batch_idxs": idxs} for idxs in local_batched_shuffled_idxs])
        
        random.shuffle(task_idxs_batches)

        batched_task_idx = []
        for task_batch in task_idxs_batches:
            batched_task_idx.extend([{"task": task_batch['task_name'], "idx": int(idx)} for idx in task_batch['batch_idxs']])
        return batched_task_idx
                
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task_data_idx = self.task_data_idxs[idx]
        task_name = task_data_idx['task']
        local_idx = task_data_idx['idx']
        example = self.task_to_dataset[task_name][local_idx]
        return example


@dataclass
class TripleCollatorMNKD(DataCollatorWithPadding):
    max_q_len: int = 32
    max_d_len: int = 128
    with_prompt: bool = False
    with_instruction: bool = False
    mask_instruction_pooling: bool = True

    input_keys = ['query', 'pos', 'negs']

    key2prompt = {"query": QUERY_KEY, "pos": DOC_KEY, "negs": DOC_KEY}

    def __post_init__(self):
        assert not (self.with_prompt and self.with_instruction), "Cannot add prompt and instruction in the same time."

    def __call__(self, features):
        collated_batch = {}

        for key in self.input_keys:
            texts: Union[List[str], List[List[str]]] = [f[key] for f in features]
            if key == 'negs': # for negs
                texts = sum(texts, [])
            # print(text)
            if self.with_instruction: # add instruction
                assert isinstance(texts[0], list), "No instruction in input text."
                instructions = [normalize_instruction(text[0]) for text in texts]
                # it seems that some instructions are dropped out in medi data
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
                    texts = ['{}: {}'.format(key2prompt(key), text) for text in texts]

            text_batch = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_d_len if key == DOC_KEY else self.max_q_len,
                return_tensors="pt",
            )
            if self.with_instruction and self.mask_instruction_pooling:
                text_batch["pooling_mask"] = (~(instruction_mask.bool()) & text_batch["attention_mask"].bool())
            collated_batch[key] = text_batch

        if "teacher_score" in features[0] and features[0]["teacher_score"] is not None:
            teacher_scores = [f["teacher_score"] for f in features]
            teacher_scores = torch.FloatTensor(teacher_scores)
            collated_batch["teacher_score"] = teacher_scores
        
        return collated_batch
