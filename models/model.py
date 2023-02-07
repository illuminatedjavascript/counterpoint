import torch
import torch.nn as nn
from .dataset import ChoraleDataset

    
class ChoraleBertConfig():
    """Default config class."""
    def __init__(self, dataset: ChoraleDataset):
        # Dataset properties
        self.vocab_len = len(dataset.key_to_token)
        self.seq_len = dataset[0][0].shape[0]
        self.pad_idx = dataset.token_to_key['<P>']
        self.device = dataset.device
        
        # Transformer model properties
        self.n_layers = 4
        self.n_heads = 6
        self.emb_dim = 30
        self.ff_dim = 2*self.emb_dim 
        

class ChoraleBertModel(nn.Module):
    def __init__(self, config: ChoraleBertConfig):
        super(ChoraleBertModel, self).__init__()
        self.config = config

        # Input layer
        self.pos_emb = nn.Embedding(config.seq_len, config.emb_dim)
        self.key_embed = nn.Embedding(config.vocab_len, config.emb_dim, padding_idx=config.pad_idx) # pad_idx required?

        # Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(config.emb_dim, config.n_heads, config.ff_dim, 
                                                        batch_first=True) # Add dropout?
        self.encoder = nn.TransformerEncoder(self.encoder_layer, config.n_layers) # Add norm (look at docs)?
        
    def forward(self, seq: torch.tensor): # Not tested
        """
        Args:
            seq: Tensor of shape (#batches, seq_len).
        Returns:
            logits: Tensor of shape (#batches, seq_len, vocab_len).
        
        """
        config = self.config

        # Embed and encode
        pos = torch.arange(0, config.seq_len, dtype=torch.long, device=config.device)
        pad_mask = (seq == config.pad_idx) # Shape (#batches, seq_len)
        emb = self.key_embed(seq) + self.pos_emb(pos)  # Shape (#batches, seq_len, emb_dim) + (1, seq_len, emb_dim)
        enc = self.encoder(src=emb, src_key_padding_mask=pad_mask) # Shape (#batches, seq_len, emb_dim)
        
        # Compute distributions using einsum where i: batch; j: seq_idx; k: emb_dim; l: vocab_idx
        logits = torch.einsum('ijk, lk -> ijl', enc, self.key_embed.weight) # Shape (#batches, seq_len, vocab_len)
    
        return logits


def test():
    pass

if __name__ == '__main__':
    test()