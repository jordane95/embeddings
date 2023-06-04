import collections
import logging
import os
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict

import datasets
from datasets import load_dataset
import torch

def read_mapping_id(id_file):
    id_dict = {}
    for line in open(id_file, encoding='utf-8'):
        id, offset = line.strip().split('\t')
        id_dict[id] = int(offset)
    return id_dict


def read_train_file(train_file):
    train_data = []
    for line in open(train_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        pos = line[1].split(',')
        train_data.append((qid, pos))
    return train_data


def read_neg_file(neg_file):
    neg_data = collections.defaultdict(list)
    for line in open(neg_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        neg = line[1].split(',')
        neg_data[qid].extend(neg)
    return neg_data


def read_teacher_score(score_files):
    teacher_score = collections.defaultdict(dict)
    for file in score_files.split(','):
        if not os.path.exists(file):
            logging.info(f"There is no score file:{file}, skip reading the score")
            return None
        for line in open(file):
            qid, did, score = line.strip().split()
            score = float(score.strip('[]'))
            teacher_score[qid][did] = score
    return teacher_score


def generate_random_neg(qids, pids, k=30):
    qid_negatives = {}
    for q in qids:
        negs = random.sample(pids, k)
        qid_negatives[q] = negs
    return qid_negatives


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, data_config: Dict[str, Any]):
        self.corpus_dataset = load_dataset('json', data_files=data_config['corpus_file'], split='train', cache_dir='cache')
        self.query_dataset = load_dataset('json', data_files=data_config['train_query_file'], split='train', cache_dir='cache')
        self.train_qrels = read_train_file(data_config['train_qrels'])
        
        self.corpus_id = read_mapping_id(data_config['corpus_id_file'])
        self.query_id = read_mapping_id(data_config['train_query_id_file'])

        if data_config['neg_file']:
            self.train_negative = read_neg_file(data_config['neg_file'])
        else:
            self.train_negative = generate_random_neg(list(self.query_id.keys()), list(self.corpus_id.keys()))

        self.teacher_score = None
        if data_config['teacher_score_files'] is not None:
            self.teacher_score = read_teacher_score(data_config['teacher_score_files'])

        self.sample_neg_from_topk = data_config['sample_neg_from_topk'] # int
        self.train_group_size = data_config['train_group_size']

    def __len__(self):
        return len(self.train_qrels)

    def create_query_example(self, id: Any) -> str:
        return self.query_dataset[self.query_id[id]]['text']

    def create_doc_example(self, id: Any) -> str:
        doc = self.corpus_dataset[self.corpus_id[id]]
        if "title" in  doc and doc["title"] and len(doc["title"]) > 2:
            doc_text = doc["title"] + " " + doc["text"]
        else:
            doc_text = doc["text"]
        return doc_text

    def __getitem__(self, item) -> Dict[str, Any]:
        group = self.train_qrels[item]

        qid = group[0]
        query = self.create_query_example(qid) # str

        teacher_scores = None
        pos_id = random.choice(group[1])
        pos_doc = self.create_doc_example(pos_id) # str
        if self.teacher_score:
            teacher_scores = []
            teacher_scores.append(self.teacher_score[qid][pos_id])

        query_negs = self.train_negative[qid][:self.sample_neg_from_topk]
        if len(query_negs) < self.train_group_size - 1:
            negs = random.sample(self.corpus_id.keys(), k=self.train_group_size - 1 - len(query_negs))
            negs.extend(query_negs)
        else:
            negs = random.sample(query_negs, k=self.train_group_size - 1)

        neg_docs = [] # List[str]
        for neg_id in negs:
            neg_docs.append(self.create_doc_example(neg_id))
            if self.teacher_score:
                teacher_scores.append(self.teacher_score[qid][neg_id])

        return {"query": query, "pos": pos_doc, "negs": neg_docs, "teacher_score": teacher_scores}


class NQDataset(torch.utils.data.Dataset):
    def __init__(self, data_config: Dict[str, Any]):
        self.dataset = load_dataset('json', data_files=data_config["train_files"], split='train', cache_dir='cache')
        self.train_group_size = data_config['train_group_size']
    
    def __len__(self):
        return len(self.dataset)
    
    def get_doc_text(self, doc: Dict[str, Any]):
        return "{} {}".format(doc.get('title', ''), doc['text']).strip()
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        query = item['question'] # str
        positives = item['positive_ctxs']
        pos = random.choice(positives)
        pos = self.get_doc_text(pos)
        negatives = item['hard_negative_ctxs']
        negative_size = self.train_group_size - 1
        if len(negatives) < negative_size: # TODO: sampling duplicate negatives is not compatible with full contrastive loss
            negs = random.choices(negatives, k=negative_size)
        else:
            negs = random.sample(negatives, k=negative_size)
        negs = [self.get_doc_text(neg) for neg in negs] # List[str]
        return {"query": query, "pos": pos, "negs": negs}


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data_config: Dict[str, Any]):
        self.dataset = load_dataset('csv', data_files=data_config["data_file"], split='train', cache_dir='cache')
        self.corpus = sum((self.dataset[key] for key in self.dataset.column_names), []) # List[str]
        self.train_group_size = data_config['train_group_size']
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        triple = self.dataset[idx]
        query = triple['sent0'] # str
        pos = triple['sent1']
        negs = [triple['hard_neg']]
        # random sampled negatives
        random_negs = random.sample(self.corpus, k=self.train_group_size-2)
        negs += random_negs
        return {"query": query, "pos": pos, "negs": negs}

class MEDIDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Any], train_group_size: int = 16):
        self.data: List[Dict[str, Any]] = data
        self.train_group_size = train_group_size
        self.corpus = [item[key][1] for item in self.data for key in ['query', 'pos', 'neg']] # List[str]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query'][1] # str
        pos = item['pos'][1] # str
        negs = [item['neg'][1]] # str
        # random sample negatives from the corpus
        random_negs = random.sample(self.corpus, k=self.train_group_size-2)
        negs += random_negs
        return {"query": query, "pos": pos, "negs": negs}

