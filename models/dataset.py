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
        tt_split: ratio for test-train split.
        device: device for PyTorch tensors.
    """
    def __init__(self, load_path: str, pitch_aug_range: int = 6, max_mask: float = 0.9,
                 tt_split: float = 0.90, device: str = 'cpu'):
        self.pitch_aug_range = pitch_aug_range
        self.max_mask = max_mask
        self.device = device

        # Token information
        self.STATIC_TOKENS = ['<S>', '<E>', '<M>', '<T>', '<P>'] 
        self.token_list =  ['<S>', '<E>', '<M>', '<T>', '<P>', -1] + list(range(1, 127 + 1)) # Token range (1, 127)

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
        src, tgt = self.train[idx].copy(), self.train[idx].copy()
        mask_p = random.uniform(0, self.max_mask)
        pitch_aug = random.randint(-self.pitch_aug_range, self.pitch_aug_range) 
        
        return self._mask_and_aug(src, tgt, pitch_aug, mask_p) 
        
    def get_test(self, n: int = -1):
        """Returns a slice of the test set.
        Args:
            n: integer of how much of the test set to return, or -1 for entire set.
        Returns:
            src: torch.tensor (shape (n, seq_len)) of augmented, masked, and encoded chorales.
            tgt: torch.tensor (shape (n, seq_len)) of augmented and encoded chorales.
        """
        if n == -1:
            n = len(self.test) - 1
        elif n >= len(self.test):
            n = len(self.test) - 1

        assert 1 < n and n < len(self.test), "Index out of range."
            
        src_tensors = []
        tgt_tensors = []
        for i in range(n):
            mask_p = random.uniform(0, self.max_mask) # Recalculate each time
            src, tgt = self._mask_and_aug(self.test[i].copy(), self.test[i].copy(), 0, mask_p)
            src_tensors.append(src)
            tgt_tensors.append(tgt)
            
        return torch.stack(src_tensors, dim=0), torch.stack(tgt_tensors, dim=0)
    
    def sample_test(self):
        """Returns a random maximally masked (src, tgt) pair for sampling.
        Returns:
            src: torch.tensor of shape (seq_len).
            tgt: torch.tensor of shape (seq_len).
        """
        idx = random.randint(0, len(self.test) - 1)
        src, tgt = self.test[idx].copy(), self.test[idx].copy()
        
        # Mask everything but the top voice
        masked_voices = [2, 3, 4] # Add back 2
        for i, tok in enumerate(src):
            if i % 5 in masked_voices:
                src[i] = '<M>'
        
        return self._encode(src), self._encode(tgt)

    def _mask_and_aug(self, src, tgt, pitch_aug: int, mask_p: float):
        """Masks, augments, and encodes a tuple (src, tgt).
        Args:
            src: raw chorale sequence as a list.
            tgt: raw chorale sequence as a list.
            pitch_aug: integer to augment all pitches by.
            mask_p: percentage of src note tokens to mask.
        Returns: 
            src_enc: torch.tensor of masked, augmented and encoded src.
            tgt_enc: torch.tensor of masked, augmented and encoded src.
        """
        # Randomly selects 3 voices to mask
        masked_voices = (torch.multinomial(torch.ones(4)/4, 3, replacement=False) + 1).tolist()

        for i, token in enumerate(src):
            # Only mask/augment note tokens
            if token in self.STATIC_TOKENS:
                continue
            
            # Augment pitch if not silent (=-1)
            if token != -1:
                tgt[i] += pitch_aug
                src[i] += pitch_aug

            # Masking
            if i % 5 in masked_voices:
                rng_mask = random.uniform(0, 1)
                if rng_mask < mask_p: # So that mask_p is true
                    src[i] = '<M>'
            
        return self._encode(src), self._encode(tgt)
        
    def _encode(self, seq: list): 
        """Converts from list[str | int] to torch.tensor.
        Args:
            seq: unencoded sequence as a list.
        Returns:
            seq_enc: encoded sequence as a torch.tensor.
        """
        return torch.tensor([*map(self.token_to_key.get, seq)], dtype=torch.long).to(self.device)

    def decode(self, seq_enc: torch.tensor):
        """Converts torch.tensor to list[str] .
        Args:
            seq_enc: encoded sequence as a torch.tensor.
        Returns:
            seq: decoded sequence as a list.
        """
        return [self.key_to_token[key] for key in seq_enc.tolist()]
