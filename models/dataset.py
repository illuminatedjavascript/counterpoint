import json
import random
import torch
from torch.utils import data


class ChoraleDataset(data.Dataset):
    """Dataset class for JSB Chorales.
    Args:
        load_path: path to (processed) dataset file.
        pitch_aug_range: range of random pitch augmentation.
        max_mask: maximum number of note tokens masked.
    """
    def __init__(self, load_path: str, pitch_aug_range: int = 6, max_mask: float = 0.7,
                 tt_split: float = 0.9, device: str = 'cpu'):
        self.pitch_aug_range = pitch_aug_range
        self.max_mask = max_mask
        self.device = device

        # Token information
        self.STATIC_TOKENS = ['<S>', '<E>', '<M>', '<T>', '<P>'] 
        self.pitch_range = (36 - pitch_aug_range, 88 + pitch_aug_range)
        self.token_list =  ['<S>', '<E>', '<M>', '<T>', '<P>', -1] + (
                          list(range(36 - pitch_aug_range - 1, 88 + pitch_aug_range + 1)))

        # Load data
        with open(load_path) as f:
            raw = json.load(f)
            
        # Generate key - token dicts
        self.key_to_token = {i: val for i, val in enumerate(self.token_list)}
        self.token_to_key = {val: i for i, val in enumerate(self.token_list)}

        # Test-train split
        tt_ind = round(tt_split * len(raw))
        self.train = raw[:tt_ind]
        self.test = raw[tt_ind:]
            
    def __len__(self):
        return len(self.train)
        
    def __getitem__(self, idx):
        pitch_aug = random.randint(-self.pitch_aug_range, self.pitch_aug_range) 
        src = self.train[idx].copy()
        tgt = self.train[idx].copy()
        
        return self._mask_and_aug(src, tgt, pitch_aug) 
        
    def get_test(self, n: int | None = None):
        """Returns the test set as (src, tgt)."""
        assert 1 < n and n < len(self.train), "Index out of range."

        src, tgt = self._mask_and_aug(self.test[0].copy(), self.test[0].copy(), 0)
        src = src.reshape(1, -1)
        tgt = tgt.reshape(1, -1)
        for i in range(1, n):
            temp_src, temp_tgt = self._mask_and_aug(self.test[i].copy(), self.test[i].copy(), 0)
            src = torch.cat((src, temp_src.reshape(1, -1)), dim=0)
            tgt = torch.cat((tgt, temp_tgt.reshape(1, -1)), dim=0)
            
        return src, tgt
        
    def _mask_and_aug(self, src, tgt, pitch_aug, max_mask=False):
        """Masks, augments, and encodes (src, tgt)"""
        if max_mask == True:
            mask_p = 0.7
        else:
            mask_p = random.uniform(0, self.max_mask)

        for i, token in enumerate(src):
            # Only mask/augment note tokens
            if token in self.STATIC_TOKENS:
                continue
            
            # Augment pitch if not silent (=-1)
            if token != -1:
                tgt[i] += pitch_aug
                src[i] += pitch_aug

            # Masking
            rng_mask = random.uniform(0, 1)
            if rng_mask < mask_p:
                src[i] = '<M>'
                    
        return self._encode(src), self._encode(tgt)
        
    def _encode(self, seq: list): 
        """Converts from list[str | int] to torch.tensor.
        Args:
            seq: unencoded sequence.
        Returns:
            seq_enc: src as torch.tensor.
        """
        return torch.tensor([*map(self.token_to_key.get, seq)], dtype=torch.long).to(self.device)

    def decode(self, seq_enc: torch.tensor):
        """Converts torch.tensor to list[str] .
        Args:
            seq_enc: encoded sequence.
        Returns:
            seq: decoded sequence.
        """
        return [self.key_to_token[key] for key in seq_enc.tolist()]

def test():
    test = ChoraleDataset('../data/processed/jsb16seq.json')
    
    print(test.get_test(5)[1].shape)

if __name__ == '__main__':
    test()