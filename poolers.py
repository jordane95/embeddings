import os
import json

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

import logging

logger = logging.getLogger(__name__)


class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}

    def forward(self, token_embeddings):
        raise NotImplementedError('EncoderPooler is an abstract class')

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

    def save(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class DensePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768):
        super(DensePooler, self).__init__()
        self.pooler = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, token_embeddings: Tensor = None, **kwargs):
        rep = self.pooler(token_embeddings)
        return rep



class Router(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super(Router, self).__init__()
        self.ff = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        router_logits = self.ff(x)
        router_probs = F.softmax(router_logits, dim=-1)
        return router_logits, router_probs


class MoEPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, n_experts: int = 8):
        super(MoEPooler, self).__init__()
        self.router = Router(input_dim, n_experts)
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for _ in range(n_experts)
        ])
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'n_experts': n_experts}


    def forward(self, token_embeddings: Tensor = None, **kwargs):
        expert_logits, expert_probs = self.router(token_embeddings) # (batch_size, seq_len, n_experts)
        expert_probs = expert_probs.unsqueeze(-1) # (batch_size, seq_len, n_experts, 1)
        expert_outputs = torch.stack([expert(token_embeddings) for expert in self.experts], dim=2) # (batch_size, seq_len, n_experts, output_dim)
        outputs = torch.sum(expert_probs * expert_outputs, dim=2) # (batch_size, seq_len, output_dim)
        return outputs
