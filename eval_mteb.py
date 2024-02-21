import os
import torch
import torch.nn.functional as F
import tqdm
import json
import numpy as np
import argparse

from transformers import AutoTokenizer
from typing import List, Dict
from mteb import MTEB

from models import AutoModelForSentenceEmbedding

from utils import logger, move_to_cuda, TASK_LIST, get_task_def_by_task_name_and_type


def get_args():
    parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
    parser.add_argument('--task-types', nargs='+', default=[], help='task types to evaluate')
    parser.add_argument('--output-dir', default='', type=str, metavar='N', help='output directory')
    parser.add_argument('--multilingual', action='store_true', help='whether to use multilingual model')

    parser.add_argument('--model-name-or-path', default='tmp-outputs/', type=str, metavar='N', help='which model to use')
    parser.add_argument('--pooling', default='mean', help='pool type')
    parser.add_argument('--normalize', action='store_true', help='normalize embeddings?')
    parser.add_argument('--add-pooler', default=None, type=str, help='projection head type')
    parser.add_argument('--n-experts', default=8, type=int, help='number of experts')
    parser.add_argument('--topk', default=2, type=int, help='topk activation experts')
    parser.add_argument('--residual-pooler', action='store_true', help='add residual conntection to pooler')

    parser.add_argument('--instruct', action='store_true', help='add instruction')

    parser.add_argument('--instruction_path', type=str, default=None, help="path to instructions for mteb")

    args = parser.parse_args()

    logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    assert args.pooling in ['mean', 'last', 'weightedmean'], 'pool_type should be cls / avg / last'
    os.makedirs(args.output_dir, exist_ok=True)
    return args


class AutoModelForSentenceEmbeddingDP(AutoModelForSentenceEmbedding):
    def forward(self, input_ids, attention_mask):
        outputs = self.lm(input_ids, attention_mask)
        embeddings = self.compress(outputs, attention_mask)
        return embeddings


class DenseEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = AutoModelForSentenceEmbeddingDP(
            model_name_or_path=args.model_name_or_path,
            pooling=args.pooling,
            normalize=args.normalize,
            add_pooler=args.add_pooler,
            n_experts=args.n_experts,
            topk=args.topk,
            residual_pooler=args.residual_pooler,
        ).load_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        
        self.prompt = ""
    
    def encode(self, sentences, **kwargs) -> np.ndarray:
        input_texts = [self.prompt + s for s in sentences]
        return self._do_encode(input_texts)
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.prompt['query'] + q for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.prompt['corpus'] + doc for doc in corpus]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        encoded_embeds = []
        batch_size = 64 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = self.tokenizer(
                batch_input_texts,
                max_length=512,
                padding=True,
                pad_to_multiple_of=8,
                return_token_type_ids=False,
                truncation=True,
                return_tensors='pt'
            )

            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                embeds = self.encoder(**batch_dict)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)
    
    def set_prompt(self, prompt):
        self.prompt = prompt


if __name__ == "__main__":
    args = get_args()
    # model = DenseEncoder(args)
    args.task_types = [t for t in args.task_types if t.strip()]
    evaluation = MTEB(
        task_types=args.task_types or None,
        task_langs=['en'] if not args.multilingual else None
    )

    for task_cls in evaluation.tasks:
        task_name: str = task_cls.description['name']
        task_type: str = task_cls.description['type']

        if task_name not in TASK_LIST:
            continue

        # disable l2 normalize for classification tasks, as it achieves slightly better results
        if task_type == 'Classification':
            args.normalize = False
        else:
            args.normalize = True
        logger.info('Set l2_normalize to {}'.format(args.normalize))

        model = DenseEncoder(args)
        if args.instruct: # e5
            task_def: str = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
            # prompt: str = get_detailed_instruct(task_def)
            prompt = "{}: ".format(task_def)
            model.set_prompt(prompt=prompt)
            logger.info('Set prompt: {}'.format(prompt))
        elif args.instruction_path: # instructor
            instructions = json.load(open(args.instruction_path))
            prompt = instructions[task_type][task_name]
            model.set_prompt(prompt=prompt)
            logger.info('Set prompt: {}'.format(prompt))
            

        sub_eval = MTEB(tasks=[task_name], task_langs=['en'] if not args.multilingual else None)
        logger.info('Running evaluation for task: {}, type: {}'.format(task_name, task_type))
        eval_splits = ["test"] if "test" in task_cls.description["eval_splits"] else task_cls.description["eval_splits"]
        sub_eval.run(
            model, eval_splits=eval_splits,
            output_folder=args.output_dir
        )
