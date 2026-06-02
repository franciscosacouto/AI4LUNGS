import torch
import torch.nn as nn

class TextContextPrompter(nn.Module):
    def __init__(self, n_ctx=16, embedding_dim=512):
        super().__init__()
        self.n_ctx = n_ctx
        ctx_vectors = torch.empty(n_ctx, embedding_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

    def forward(self, prefix_embed, suffix_embed, content_embed):
        return torch.cat([
            prefix_embed, 
            self.ctx.unsqueeze(0).expand(prefix_embed.shape[0], -1, -1), 
            content_embed, 
            suffix_embed
        ], dim=1)

class ClinicalTextEncoder(nn.Module):
    def __init__(self, text_transformer_backbone, n_ctx=16):
        super().__init__()
        self.text_encoder = text_transformer_backbone
        self.prompt_learner = TextContextPrompter(n_ctx=n_ctx, embedding_dim=getattr(text_transformer_backbone, 'width', 512))

    def forward(self, sentences, tokenizer):
        # Your manual transformer residual block routing logic goes here...
        # It processes text tokens and returns a clean feature vector
        return text_emb