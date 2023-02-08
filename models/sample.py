import os
import torch
import json
from midiutil import MIDIFile
from .model import ChoraleBertModel
from .dataset import ChoraleDataset

# Does this need to be a class? or can it be a function?
class Sampler(): # NOT WORKING
    def __init__(self, model: ChoraleBertModel, dataset: ChoraleDataset,
                 load_path: str, save_path: str):
        self.model = model
        self.dataset = dataset
        self.save_path = save_path
        
        # Load prompts
        with open(load_path) as f:
            prompts = json.load(f)

        # Sample
        model.eval()
        samples = []
        for seq in prompts:
            seq_enc = dataset._encode(seq)
            sample_enc = gibbs_sample(model, dataset, seq_enc)
            print('sample_enc')
            print(sample_enc)
            sample_dec = dataset.decode(sample_enc)
            print('sample_dec')
            print(sample_dec)
            samples.append(sample_dec)

        # Save samples as json
        with open(os.path.join(save_path, f'samples.json'), "w") as f:
            json.dump(samples, f)
            
        # Save samples as MIDI
        for i, seq in enumerate(samples):
            to_midi(dataset, seq, os.path.join(save_path, f'sample_{i}.midi'))

def gibbs_sample(model: ChoraleBertModel, dataset: ChoraleDataset, seq: torch.tensor):
    num_step = 1000
    block_size = 3

    mask_key = dataset.token_to_key['<M>']
    uniform_dist = torch.where(seq == mask_key, 1., 0.)
    uniform_dist = uniform_dist / torch.linalg.norm(uniform_dist)

    for _ in range(num_step):
        idx = torch.multinomial(uniform_dist, block_size, replacement=False)
        seq[idx] = mask_key
        logits = model.forward(seq.reshape(1, -1)) # Shape (1, seq_len, vocab_len)
        probs = torch.nn.functional.softmax(logits[0, idx, :], dim=1) # Shape (block_size, vocab_len)
        
        for i in range(block_size):
            seq[idx[i]] = torch.multinomial(probs[i], 1)
            
    return seq


# Clean up this func
def to_midi(dataset, seq, save_path):
    track = 0
    channel = 0
    time = 0 
    volume = 100 
    midi_res = MIDIFile(1)
    
    for token in seq:
        if token in dataset.STATIC_TOKENS:
            if token == '<T>':
                time += 1
            continue
        
        midi_res.addNote(track, channel, pitch=int(token), duration=1, time=time, volume=volume)
    
    # Save
    with open(save_path, "wb") as f:
        midi_res.writeFile(f)