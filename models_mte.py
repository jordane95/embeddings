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



class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self.pooling = 'mean'
        self._config = {}

    def forward(self, token_embeddings):
        raise NotImplementedError('EncoderPooler is an abstract class')
    
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



    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class DensePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, normalize=False):
        super(DensePooler, self).__init__()
        self.normalize = normalize
        self.pooler = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'normalize': normalize}

    def forward(self, token_embeddings: Tensor = None, **kwargs):
        sentence_embeddings = self.pool_sentence_embedding(token_embeddings)
        rep = self.pooler(sentence_embeddings)
        if self.normalize:
            rep = nn.functional.normalize(rep, dim=-1)
        return rep


class ModularPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, normalize=False, n_experts: int = 32, top_k: int = 8):
        

        self.router = nn.Linear(input_dim, n_experts)
        self.experts = [
            nn.Linear(input_dim, output_dim)
            for _ in range(n_experts)
        ]
        

    def forward(self, token_embeddings: Tensor = None, **kwargs):
        expert_probs = self.router(token_embeddings) # (batch_size, seq_len, n_experts)
        token_experts = self.

        pass
    


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
        
        self.pooler = DensePooler(args)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def encode(self, texts):
        if texts is None:
            return None
        # import pdb; pdb.set_trace()
        pooling_mask = texts.pop('pooling_mask') if "pooling_mask" in texts else texts['attention_mask']
        outputs = self.lm(**texts)
        token_embeddings = outputs.last_hidden_state
        embeddings = self.pooler(token_embeddings)
        return embeddings.contiguous()
    
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
            torch.save(self.pooler.state_dict(), os.path.join(output_path, 'pooler.pt'))

    def load_pretrained(self, output_path):
        self.lm = self.lm.from_pretrained(output_path)
        if self.add_pooler:
            try:
                pooler_states = torch.load(os.path.join(output_path, 'pooler.pt'))
                self.pooer.load_state_dict(pooler_states)
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
