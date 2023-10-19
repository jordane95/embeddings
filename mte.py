from models import AutoModelForSentenceEmbedding, AutoModelForEmbeddingMNKD


class ModularTextEmbedding(nn.Moduel):
    def __init__(self, *args, **kwargs):
        super(AutoModelForSentenceEmbedding, self).__init__(*args, **kwargs)

        self.n_experts = kwargs.get('n_experts')
        self.embedding_dim = kwargs.get('embedding_dim')

        self.router = nn.Linear(self.lm.config.hidden_size, self.n_experts) # routes each token to multiple experts

        self.experts = [
            nn.Linear(self.lm.config.hidden_size, self.embedding_dim or self.lm.config.hidden_size)
            for _ in range(self.n_experts)
        ]

    def encode(self, texts):
        if texts is None:
            return None
        # import pdb; pdb.set_trace()
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

