import logging
import os
import sys
import json

import torch
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from arguments import (
    ModelArguments,
    DataArguments,
    EmbeddingTrainingArguments as TrainingArguments,
)

from data import (
    InfiniteMultipleIterableDataset,
    QDCollator,
)
from models import AutoModelForSentenceEmbedding
from trainer import (
    EmbeddingTrainer as Trainer,
    GCTrainer,
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    training_args.remove_unused_columns = False

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    model = AutoModelForSentenceEmbedding(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
    )

    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    data_config = json.load(open(data_args.data_config))

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_batch_size = training_args.per_device_train_batch_size * world_size
    train_dataset = InfiniteMultipleIterableDataset(
        train_dir=data_args.train_dir,
        data_config=data_config,
        batch_size=global_batch_size,
        query_field=data_args.query_column,
        doc_field=data_args.doc_column,
    )

    data_collator = QDCollator(
        tokenizer,
        max_q_len=data_args.q_max_len,
        max_d_len=data_args.d_max_len
    )

    # torch.autograd.set_detect_anomaly(True)
    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()  # TODO: resume training
    trainer.save_model()


if __name__ == "__main__":
    main()
