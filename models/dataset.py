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
    def __init__(self, load_path: str, pitch_aug_range: int = 6, max_mask: float = 0.75,
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
        src, tgt = self.train[idx].copy(), self.train[idx].copy()
        mask_p = random.uniform(0, self.max_mask)
        pitch_aug = random.randint(-self.pitch_aug_range, self.pitch_aug_range) 
        
        return self._mask_and_aug(src, tgt, pitch_aug, mask_p) 
        
    def get_test(self, n: int = 100):
        """Returns a slice of the test set.
        Args:
            n: integer of how much of the test set to return.
        Returns:
            src: torch.tensor (shape (n, seq_len)) of augmented, masked, and encoded chorales.
            tgt: torch.tensor (shape (n, seq_len)) of augmented and encoded chorales.
        """
        assert 1 < n and n < len(self.train), "Index out of range."

        mask_p = random.uniform(0, self.max_mask)
        src, tgt = self._mask_and_aug(self.test[0].copy(), self.test[0].copy(), 0, mask_p)
        src = src.reshape(1, -1)
        tgt = tgt.reshape(1, -1)
        for i in range(1, n):
            mask_p = random.uniform(0, self.max_mask)
            temp_src, temp_tgt = self._mask_and_aug(self.test[i].copy(), self.test[i].copy(), 0, mask_p)
            src = torch.cat((src, temp_src.reshape(1, -1)), dim=0)
            tgt = torch.cat((tgt, temp_tgt.reshape(1, -1)), dim=0)
            
        return src, tgt
    
    def get_rand_test(self):
        """Returns a random maximally masked src, tgt pair for evaluation.
        Returns:
            src: torch.tensor of shape (seq_len).
            tgt: torch.tensor of shape (seq_len).
        """
        idx = random.randint(0, len(self.test))
        src, tgt = self.test[idx].copy(), self.test[idx].copy()

        return self._mask_and_aug(src, tgt, 0, 0.75)
        
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