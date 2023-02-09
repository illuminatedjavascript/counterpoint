import os
import random
import json
import torch
from midiutil import MIDIFile
from .model import ChoraleBertModel
from .dataset import ChoraleDataset

# Does this need to be a class? or can it be a function?
class Sampler(): 
    def __init__(self, model: ChoraleBertModel, dataset: ChoraleDataset,
                 save_path: str):
        self.model = model
        self.dataset = dataset
        self.save_path = save_path
        
        # Sample
        model.eval()
        samples = []
        
        # Samples from train set
        idx = random.randint(1, 1000)
        src, tgt = dataset[idx]
        src_dec = dataset.decode(src)
        
        tgt_dec = dataset.decode(tgt)
        sample_enc = gibbs_sample(model, dataset, src)
        sample_dec = dataset.decode(sample_enc)
        
        print('src')
        print(src_dec)
        print('tgt')
        print(tgt_dec)
        print('sample')
        print(sample_dec)

        samples.append(src_dec)
        samples.append(tgt_dec)
        samples.append(sample_dec)
        
        # Save samples as json
        with open(os.path.join(save_path, f'samples.json'), "w") as f:
            json.dump(samples, f)
            
        # Save samples as MIDI
        to_midi(dataset, tgt_dec, os.path.join(save_path, f'sample_bach.midi'))
        to_midi(dataset, sample_dec, os.path.join(save_path, f'sample_model.midi'))

def gibbs_sample(model: ChoraleBertModel, dataset: ChoraleDataset, seq: torch.tensor):
    num_step = 1000 # CHANGE
    block_size = 10
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


def to_midi(dataset: ChoraleDataset, seq, save_path): # NOT WORKING FOR MASKED SRC
    """THIS NEEDS TO BE CLEANED UP AND DOCUMENTED/COMMENTED BETTER."""
    assert '<P>' not in seq, "Sequence cannot contain pad token."
    midi_res = MIDIFile(1)
    midi_res.addProgramChange(0, 0, 0, 0)
    midi_res.addTempo(0, 0, 140)

    # Reformat into the difference channels
    tok_idx = 0
    chord_idx = 0
    seq_reformatted = [[], [], [], []]
    for token in seq:
        if token in dataset.STATIC_TOKENS:
            if token == '<T>':
                chord_idx += 1    
                tok_idx = 0
            elif token == '<M>':
                seq_reformatted[tok_idx].append(-1)
                tok_idx += 1
            continue
        
        seq_reformatted[tok_idx].append(int(token))
        tok_idx += 1
        
    seq_len = len(seq_reformatted[0]) - 1
        
    for channel in seq_reformatted:
        time_buffer = 0
        note_buffer = channel[0]
        
        for curr_time, note in enumerate(channel):
            if curr_time == 0:
                continue
                
            # Case: silence (-1) in the buffer
            if note_buffer == -1:
                if note == -1: # Case: current note is silence
                    pass # Do nothing
                else: # Case: current note in not silence
                    if curr_time == seq_len: # Case: last note
                        midi_res.addNote(0, 0, pitch=note_buffer, duration=1, time=time_buffer, volume=100) # Play final note
                    else: # Case: not the last note
                        note_buffer = note # Update buffer
                        time_buffer = curr_time # Update buffer

            else: # Case: silence is not in the buffer
                if note == -1: # Case: current note is silence
                    midi_res.addNote(0, 0, pitch=note_buffer, duration=(curr_time-time_buffer), time=time_buffer, volume=100) # Play note
                    note_buffer = note # Update buffer
                    time_buffer = curr_time # Update buffer
                elif note == note_buffer: # Case: current note is the same as buffer note
                    if curr_time == seq_len: # Case: current note is last
                        midi_res.addNote(0, 0, pitch=note_buffer, duration=((curr_time-time_buffer) + 1), time=time_buffer, volume=100) # Play final note
                    else: # Case: current note is not last:
                        pass # Do nothing
                elif note != note_buffer: # Case: current note is different from buffer note
                    midi_res.addNote(0, 0, pitch=note_buffer, duration=(curr_time-time_buffer), time=time_buffer, volume=100) # Play note
                    note_buffer = note
                    time_buffer = curr_time
                    
    # Save
    with open(save_path, "wb") as f:
        midi_res.writeFile(f)