import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    embedding_dim: int = field(default=768)
    normalize: bool = field(default=False)
    pooling: str = field(default='mean')

    # peft
    bitfit: bool = field(default=False)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )

    data_config: str = field(default="config/data_config.json")

    mix_coefficient: float = field(default=0.0)

    query_column: Optional[str] = field(
        default="question",
        metadata={"help": "The name of the column in the datasets containing the questions."},
    )
    doc_column: Optional[str] = field(
        default="passage",
        metadata={"help": "The name of the column in the datasets containing the passages."},
    )

    q_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    d_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for document. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    add_prompt: bool = field(
        default=False, metadata={"help": "Prepend simple prompt to the text. e.g, 'query: this is a query', 'doc: this is a docc'."}
    )
    
    add_instruction: bool = field(
        default=False, metadata={"help": "Prepend detailed instructions for the data."}
    )

    mask_instruction_pooling: bool = field(
        default=True, metadata={"help": "Whether or not mask instruction tokens during pooling."}
    )

    finetune_data_path: str = field(default=None, metadata={"help": "Path to the json file for finetuning."})


@dataclass
class EmbeddingTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_d_chunk_size: int = field(default=32)

    temperature: float = field(default=1.0)
