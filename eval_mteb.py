import os
import torch
import torch.nn.functional as F
import tqdm
import json
import numpy as np
import argparse

from transformers import AutoTokenizer
from typing import List
from mteb import MTEB

from models import AutoModelForSentenceEmbedding

from utils import logger, move_to_cuda



def get_args():
    parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
    parser.add_argument('--task-types', nargs='+', default=[], help='task types to evaluate')
    parser.add_argument('--output-dir', default='', type=str, metavar='N', help='output directory')
    parser.add_argument('--multilingual', action='store_true', help='whether to use multilingual model')

    parser.add_argument('--model-name-or-path', default='tmp-outputs/', type=str, metavar='N', help='which model to use')
    parser.add_argument('--pooling', default='mean', help='pool type')
    parser.add_argument('--normalize', action='store_true', help='normalize embeddings?')
    parser.add_argument('--add-pooler', default='dense', type=str, help='projection head type')
    parser.add_argument('--n-experts', default=8, type=int, help='number of experts')
    parser.add_argument('--residual-pooler', action='store_true', help='add residual conntection to pooler')

    args = parser.parse_args()

    logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    assert args.pooling in ['mean', 'last', 'weightedmean'], 'pool_type should be cls / avg / last'
    os.makedirs(args.output_dir, exist_ok=True)
    return args


class DenseEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = AutoModelForSentenceEmbedding(
            model_name_or_path=args.model_name_or_path,
            pooling=args.pooling,
            normalize=args.normalize,
            add_pooler=args.add_pooler,
            n_experts=args.n_experts,
            residual_pooler=args.residual_pooler,
        ).load_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    @torch.no_grad()
    def encode(self, sentences, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        input_texts: List[str] = sentences

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
                embeds = self.encoder.module.encode(batch_dict)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)


if __name__ == "__main__":
    args = get_args()
    model = DenseEncoder(args)
    args.task_types = [t for t in args.task_types if t.strip()]
    evaluation = MTEB(
        task_types=args.task_types or None,
        task_langs=['en'] if not args.multilingual else None
    )

    for task_cls in evaluation.tasks:
        task_name: str = task_cls.description['name']
        task_type: str = task_cls.description['type']

        # disable l2 normalize for classification tasks, as it achieves slightly better results
        # if task_type == 'Classification':
        #     logger.info('Set l2_normalize to False for classification task')
        #     model.l2_normalize = False
        # else:
        #     model.l2_normalize = True
        #     logger.info('Set l2_normalize to {}'.format(model.l2_normalize))

        sub_eval = MTEB(tasks=[task_name], task_langs=['en'] if not args.multilingual else None)
        logger.info('Running evaluation for task: {}, type: {}'.format(task_name, task_type))
        eval_splits = ["test"] if "test" in task_cls.description["eval_splits"] else task_cls.description["eval_splits"]
        sub_eval.run(
            model, eval_splits=eval_splits,
            output_folder=args.output_dir
        )
