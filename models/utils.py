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
        randtok_p: probability that a masked token is replace randomly."""
    def __init__(self, load_path: str, pitch_aug_range: int = 5, mask_p: float = 0.15, randtok_p: float = 0.1):
        self.pitch_aug_range = pitch_aug_range
        self.mask_p = mask_p
        self.randtok_p = randtok_p

        self.MASK_TOKEN = '<M>'
        self.STATIC_TOKENS = ['<S>', '<E>', '<T>', '<P>', 's-1', 'a-1', 't-1', 'b-1'] # REMOVE THE -1s
        self.pitch_range = (21 - pitch_aug_range, 108 + pitch_aug_range)

        with open(load_path) as f:
            self.raw = json.load(f)
            
    def __len__(self):
        return len(self.raw)
        
    def __getitem__(self, idx):
        pitch_aug = random.randint(-self.pitch_aug_range, self.pitch_aug_range) # Not implemented yet
        tgt = self.raw[idx].copy()
        src = self.raw[idx].copy()
        
        # Masking
        for i, token in enumerate(src):
            # Only mask/augment note tokens
            if token in self.STATIC_TOKENS:
                continue
            
            # Augment pitch
            tgt[i] = tgt[i][0] + str(pitch_aug + int(tgt[i][1:]))
            src[i] = src[i][0] + str(pitch_aug + int(src[i][1:]))

            # BERT-style masking
            rng_mask = random.uniform(0, 1)
            if rng_mask < self.mask_p:
                rng_randtok = random.uniform(0, 1)
                if rng_randtok > self.randtok_p:
                    src[i] = '<M>'
                else:
                    src[i] = src[i][0] + str(random.randint(*self.pitch_range))
        
        return src, tgt
        
    def _generate_dicts(self):
        # Implement token <-> int dictionaries

        raise(NotImplementedError)
        
    def _encode(self, src, tgt): 
        """Converts between list[str] and torch.tensor.
        Args:
            src: masked input.
            tgt: unmasked target.
        Returns:
            enc_src: src as torch.tensor.
            enc_tgt: tgt as torch.tensor."""

        raise(NotImplementedError)


def test():
    test = ChoraleDataset('../data/processed/jsb16seq.json')
    for i in range(15):
        print(test[0][0])

if __name__ == '__main__':
    test()