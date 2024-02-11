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

from poolers import DensePooler, MoEPooler


POOLER_TYPE = {
    "dense": DensePooler,
    "moe": MoEPooler,
}


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
        embedding_dim: Optional[int] = None,
        add_pooler: str = None,
        peft: bool = False,
        n_experts: int = 8,
        residual_pooler: bool = False,
        **kwargs,
    ):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.lm = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        self.pooling = pooling
        self.normalize = normalize
        self.add_pooler = add_pooler
        self.config = self.lm.config

        if self.add_pooler:
            if self.add_pooler == 'dense':
                self.pooler = DensePooler(input_dim=self.config.hidden_size, output_dim=self.config.hidden_size)
            elif self.add_pooler == 'moe':
                self.pooler = MoEPooler(input_dim=self.config.hidden_size, output_dim=self.config.hidden_size, n_experts=n_experts)
            else:
                raise NotImplementedError(f"{self.add_pooler} type poolyer not supported!")
        else:
            self.pooler = nn.Identity()
        
        self.residual_pooler = residual_pooler
        
        if peft:
            # freeze lm, only tune pooler
            for name, param in self.lm.named_parameters():
                param.requires_grad = False

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def encode(self, texts):
        if texts is None:
            return None
        # import pdb; pdb.set_trace()
        pooling_mask = texts.pop('pooling_mask') if "pooling_mask" in texts else texts['attention_mask']
        outputs = self.lm(**texts)
        last_hidden_state = outputs.last_hidden_state
        # original embedding
        embeddings = self.pool_sentence_embedding(last_hidden_state, pooling_mask)
        # pooler embedding
        pooled_reps = self.pooler(last_hidden_state) # (bs, seq_len, emb_dim)
        pooled_embeddings = self.pool_sentence_embedding(pooled_reps, pooling_mask)

        if self.add_pooler:
            if self.residual_pooler:
                embeddings += pooled_embeddings
            else:
                embeddings = pooled_embeddings

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()
    
    def pool_sentence_embedding(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling == 'mean':
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)[..., None]
        elif self.pooling == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling == 'weightedmean':
            token_embeddings = last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                    torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float().to(token_embeddings.device)
                )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights
            
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == 'last':
            token_embeddings = last_hidden_state
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # argmin gives us the index of the first 0 in the attention mask; We get the last 1 index by subtracting 1
            gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1 # Shape [bs]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)
            
            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
            return embedding
        else:
            raise NotImplementedError(f"Currently do not support pooling method: {self.pooling}")

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        doc: Dict[str, Tensor] = None,
        temperature: float = 1.0,
        negatives_x_device: bool = False,
        loss_scale: float = 1.0,
        full_contrastive_loss: bool = True,
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


        scores, labels = full_contrastive_scores_and_labels(q_embeddings, d_embeddings, use_all_pairs=full_contrastive_loss)
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
            # torch.save(self.pooler.state_dict(), os.path.join(output_path, 'pooler.pt'))
            self.pooler.save(output_path)

    def load_pretrained(self, output_path):
        self.lm = self.lm.from_pretrained(output_path)
        if self.add_pooler:
            try:
                # pooler_states = torch.load(os.path.join(output_path, 'pooler.pt'))
                # self.pooer.load_state_dict(pooler_states)
                self.pooler.load(output_path)
            except FileNotFoundError:
                logger.info(f"Cannot find pooler.pt at {output_path}")



class AutoModelForEmbeddingTriple(AutoModelForSentenceEmbedding):
    def forward(
        self,
        query: Dict[str, Tensor] = None,
        pos: Dict[str, Tensor] = None,
        neg: Dict[str, Tensor] = None,
        temperature: float = 1.0,
        negatives_x_device: bool = False,
        loss_scale: float = 1.0,
        full_contrastive_loss: bool = True,
    ):
        q_embeddings = self.encode(query) # (batch_size, embedding_dim)
        p_embeddings = self.encode(pos)
        n_embeddings = self.encode(neg)

        if negatives_x_device and dist.is_initialized():
            q_embeddings = dist_gather_tensor(q_embeddings)
            p_embeddings = dist_gather_tensor(p_embeddings)
            n_embeddings = dist_gather_tensor(n_embeddings)
        
        d_embeddings = torch.cat([p_embeddings, n_embeddings])
        scores, labels = full_contrastive_scores_and_labels(q_embeddings, d_embeddings, use_all_pairs=full_contrastive_loss)
        scores /= temperature

        loss = self.cross_entropy(scores, labels) * loss_scale
        # import pdb; pdb.set_trace()
        return EncoderOutput(
            q_reps=q_embeddings,
            d_reps=d_embeddings,
            scores=scores,
            loss=loss,
        )


class AutoModelForEmbeddingMNKD(AutoModelForSentenceEmbedding):
    # mutiple negatives and knowledge distillation
    def __init__(self, *args, **kwargs):
        super(AutoModelForEmbeddingMNKD, self).__init__(*args, **kwargs)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        pos: Dict[str, Tensor] = None,
        negs: Dict[str, Tensor] = None,
        teacher_score: Tensor = None,
        temperature: float = 1.0,
        negatives_x_device: bool = False,
        loss_scale: float = 1.0,
        full_contrastive_loss: bool = True,
    ):
        q_embeddings = self.encode(query) # (batch_size, embedding_dim)
        p_embeddings = self.encode(pos) # (batch_size, embedding_dim)
        n_embeddings = self.encode(negs) # (batch_size * num_neg, embedding_dim)

        kl_loss = 0.0
        self.contrastive_loss_weight = 0.2
        if teacher_score is not None:
            batch_size, embedding_dim = q_embeddings.shape
            student_q = q_embeddings.view(batch_size, 1, embedding_dim)  # B 1 D
            student_p = p_embeddings.view(batch_size, 1, embedding_dim)  # B 1 D
            student_n = n_embeddings.view(batch_size, -1, embedding_dim) # B N D
            student_d = torch.cat([student_p, student_n], dim=1) # B 1+N D
            student_score = student_q @ student_d.transpose(-2, -1) # B 1 1+N
            student_score = student_score.squeeze(1)  # B 1+N

            inputs = F.log_softmax(student_score / temperature, dim=-1)
            target = F.softmax(teacher_score, dim=-1)
            kl_loss = self.kl(inputs, target)

        if negatives_x_device and dist.is_initialized():
            q_embeddings = dist_gather_tensor(q_embeddings)
            p_embeddings = dist_gather_tensor(p_embeddings)
            n_embeddings = dist_gather_tensor(n_embeddings)
        
        d_embeddings = torch.cat([p_embeddings, n_embeddings])
        scores, labels = full_contrastive_scores_and_labels(q_embeddings, d_embeddings, use_all_pairs=full_contrastive_loss)
        scores /= temperature

        loss = self.cross_entropy(scores, labels) * loss_scale

        if teacher_score is not None:
            loss = kl_loss + self.contrastive_loss_weight * loss

        # import pdb; pdb.set_trace()
        return EncoderOutput(
            q_reps=q_embeddings,
            d_reps=d_embeddings,
            scores=scores,
            loss=loss,
        )
