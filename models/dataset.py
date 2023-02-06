import json
import random
import torch
from torch.utils import data


class ChoraleDataset(data.Dataset):
    """Dataset class for JSB Chorales.
    Args:
        load_path: path to (processed) dataset file.
        pitch_aug_range: range of random pitch augmentation.
        mask_p: probability that a token is masked.
        randtok_p: probability that a masked token is replace randomly.
    """
    def __init__(self, load_path: str, pitch_aug_range: int = 5, mask_p: float = 0.15, randtok_p: float = 0.1,
                 tt_split: float = 0.9, device: str = 'cpu'):
        self.pitch_aug_range = pitch_aug_range
        self.mask_p = mask_p
        self.randtok_p = randtok_p
        self.device = device

        # Token information
        self.STATIC_TOKENS = ['<S>', '<E>', '<M>', '<T>', '<P>', '-1'] 
        self.pitch_range = (21 - pitch_aug_range, 108 + pitch_aug_range)
        self.token_list =  ['<S>', '<E>', '<M>', '<T>', '<P>', '-1'] + (
                          list(range(21 - pitch_aug_range, 108 + pitch_aug_range + 1)))

        # Generate key - token dicts
        self.key_to_token = {i: val for i, val in enumerate(self.token_list)}
        self.token_to_key = {val: i for i, val in enumerate(self.token_list)}

        # Load data (+ test train split)
        with open(load_path) as f:
            raw = json.load(f)
        tt_ind = round(tt_split * len(raw))
        self.train = raw[:tt_ind]
        self.test = raw[tt_ind:]
            
    def __len__(self):
        return len(self.train)
        
    def __getitem__(self, idx):
        pitch_aug = random.randint(-self.pitch_aug_range, self.pitch_aug_range) # Not implemented yet
        tgt = self.train[idx].copy()
        src = self.train[idx].copy()
        
        # Masking
        for i, token in enumerate(src):
            # Only mask/augment note tokens
            if token in self.STATIC_TOKENS:
                continue
            
            # Augment pitch
            tgt[i] += pitch_aug
            src[i] += pitch_aug

            # BERT-style masking
            rng_mask = random.uniform(0, 1)
            if rng_mask < self.mask_p:
                rng_randtok = random.uniform(0, 1)
                if rng_randtok > self.randtok_p:
                    src[i] = '<M>'
                else:
                    src[i] = random.randint(*self.pitch_range)
                    
        return self._encode(src), self._encode(tgt)
        
    def _encode(self, seq: list): 
        """Converts from list[str] to torch.tensor.
        Args:
            seq: unencoded sequence.
        Returns:
            seq_enc: src as torch.tensor.
        """
        return torch.tensor([*map(self.token_to_key.get, seq)], dtype=torch.long).to(self.device)
        
    def _decode(self, seq_enc: torch.tensor):
        """Converts torch.tensor to list[str] .
        Args:
            seq_enc: encoded sequence.
        Returns:
            seq: decoded sequence.
        """
        return [*map(self.key_to_token.get, seq_enc.tolist())]

def test():
    test = ChoraleDataset('../data/processed/jsb16seq.json')
    print('Test __getitem__')
    for i in range(15):
        print(test[i][0])
        print(test[i][0].shape)
        print(test[i][1])
        print(test[i][1].shape)


if __name__ == '__main__':
    test()