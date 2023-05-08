import logging
from typing import Optional, Dict
from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import ModelOutput

from utils import dist_gather_tensor, full_contrastive_scores_and_labels

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    d_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = 'mean',
        normalize: bool = True,
        add_pooler: bool = False,
        embedding_dim: Optional[int] = None,
        **kwargs,
    ):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.lm = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        self.pooling = pooling
        self.normalize = normalize
        self.add_pooler = add_pooler
        self.pooler = nn.Linear(self.lm.config.hidden_size, embedding_dim or self.lm.config.hidden_size) if add_pooler else nn.Identity()

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def encode(self, texts):
        if texts is None:
            return None
        pooling_mask = texts.pop('pooling_mask') if "pooling_mask" in texts else texts['attention_mask']
        outputs = self.lm(**texts)
        last_hidden_state = outputs.last_hidden_state
        embeddings = self.pool_sentence_embedding(last_hidden_state, pooling_mask)
        embeddings = self.pooler(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()
    
    def pool_sentence_embedding(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling == 'mean':
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)[..., None]
        elif self.pooling == 'cls':
            return last_hidden_state[:, 0]

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        doc: Dict[str, Tensor] = None,
        temperature: float = 1.0,
        negatives_x_device: bool = False,
        loss_scale: float = 1.0,
    ):
        q_embeddings = self.encode(query) # (batch_size, embedding_dim)
        d_embeddings = self.encode(doc)

        if q_embeddings is None or d_embeddings is None: # for grad cache
            return EncoderOutput(
                q_reps=q_embeddings,
                d_reps=d_embeddings
            )

        if negatives_x_device and dist.is_initialized():
            q_embeddings = dist_gather_tensor(q_embeddings)
            d_embeddings = dist_gather_tensor(d_embeddings)


        scores, labels = full_contrastive_scores_and_labels(q_embeddings, d_embeddings, use_all_pairs=True)
        scores /= temperature

        loss = self.cross_entropy(scores, labels) * loss_scale
        # import pdb; pdb.set_trace();
        return EncoderOutput(
            q_reps=q_embeddings,
            d_reps=d_embeddings,
            scores=scores,
            loss=loss,
        )
    
    def gradient_checkpointing_enable(self):
        self.lm.gradient_checkpointing_enable()

    def save_pretrained(self, output_path):
        self.lm.save_pretrained(output_path)
        if self.add_pooler:
            torch.save(self.pooler.state_dict(), os.path.join(output_path, 'pooler.pt'))

    def load_pretrained(self, output_path):
        self.lm = self.lm.from_pretrained(output_path)
        if self.add_pooler:
            try:
                pooler_states = torch.load(os.path.join(output_path, 'pooler.pt'))
                self.pooer.load_state_dict(pooler_states)
            except FileNotFoundError:
                logger.info(f"Cannot find pooler.pt at {output_path}")
