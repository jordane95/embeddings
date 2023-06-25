import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer
from transformers.trainer_utils import has_length
from transformers.trainer_pt_utils import ShardSampler

import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.distributed as dist

from loss import SimpleContrastiveLoss, DistributedContrastiveLoss

import logging
logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


TRAINING_ARGS_NAME = "training_args.bin"


class EmbeddingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(EmbeddingTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = 1.0
        if self.args.negatives_x_device and dist.is_initialized():
            self._dist_loss_scale_factor = dist.get_world_size() if self.args.loss_scale<=0 else self.args.loss_scale
        logger.info(f"Using loss scale: {self._dist_loss_scale_factor}")
        self._warmup_steps = self.args.get_warmup_steps(self.args.max_steps)
        logger.info(f"Warmup steps: {self._warmup_steps}")

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            return SequentialSampler(self.train_dataset)
        else:
            return ShardSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        disable_x_device = self.args.contrastive_warmup and (self.state.global_step <= self._warmup_steps)
        negatives_x_device = self.args.negatives_x_device and not disable_x_device
        return model(
            **inputs,
            temperature=self.args.temperature,
            negatives_x_device=negatives_x_device,
            loss_scale=self._dist_loss_scale_factor if negatives_x_device else 1.0,
            full_contrastive_loss=self.args.full_contrastive_loss,
        ).loss

    def training_step(self, *args):
        disable_x_device = self.args.contrastive_warmup and (self.state.global_step <= self._warmup_steps)
        negatives_x_device = self.args.negatives_x_device and not disable_x_device
        loss_scale_factor = self._dist_loss_scale_factor if negatives_x_device else 1.0
        return super(EmbeddingTrainer, self).training_step(*args) / loss_scale_factor


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.d_reps
    else:
        return x.q_reps


class GCTrainer(EmbeddingTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls(temperature=self.args.temperature)

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_d_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        queries, documents = {'query': inputs['query']}, {'doc': inputs['doc']}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, documents, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor
