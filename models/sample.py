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
        samples = []
        for seq in prompts:
            seq_enc = dataset._encode(seq)
            sample_enc = gibbs_sample(model, dataset, seq_enc)
            #sample_enc = default_sample(model, seq_enc)
            samples.append(dataset.decode(sample_enc))

        # Save samples as json
        with open(os.path.join(save_path, f'samples.json'), "w") as f:
            json.dump(samples, f)
            
        # Save samples as MIDI
        for i, seq in enumerate(samples):
            to_midi(dataset, seq, os.path.join(save_path, f'sample_{i}.midi'))

def gibbs_sample(model, dataset, seq):
    raise(NotImplementedError)

# Clean up
def to_midi(dataset, seq, save_path):
    track = 0
    channel = 0
    time = 0 
    volume = 100 
    midi_res = MIDIFile(1)
    
    for token in seq:
        if token in dataset.STATIC_TOKENS:
            if token == '<T>':
                time += 2
            continue
        
        midi_res.addNote(track, channel, pitch=int(token), duration=2, time=time, volume=volume)
    
    # Save
    with open(save_path, "wb") as f:
        midi_res.writeFile(f)