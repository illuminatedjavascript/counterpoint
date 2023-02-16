import os
import torch
import numpy as np
from midiutil import MIDIFile
from .model import ChoraleBertModel
from .dataset import ChoraleDataset

# Does this need to be a class? or can it be a function?
class Sampler(): 
    """Placeholder entrypoint for calculating, processing, and saving samples.
    Args:
        model: ChoraleBertModel instance to be sampled from.
        dataset: ChoraleDataset instance to get prompts from.
    """
    def __init__(self, model: ChoraleBertModel, dataset: ChoraleDataset,
                 save_path: str):
        self.model = model
        self.dataset = dataset
        self.save_path = save_path
        
        model.eval()
        self._test_set(25)
        self._c_maj(10)
        
    def _test_set(self, num_samples: int):
        """Internal function for generating, processing, and saving samples from test set.
        Args:
            num_samples: number of samples required.
        """
        dataset = self.dataset
        model = self.model
        save_path = self.save_path
    
        for i in range(1, num_samples):
            # Get src not containing padding
            src, tgt = dataset.get_rand_test()
            src_dec = dataset.decode(src)
            tgt_dec = dataset.decode(tgt)
            while '<P>' in src_dec:
                src, tgt = dataset.get_rand_test()
                src_dec = dataset.decode(src)
                tgt_dec = dataset.decode(tgt)

            # Sample
            sample_enc = gibbs_sample(model, dataset, src)
            sample_dec = dataset.decode(sample_enc)

            # Save samples as MIDI
            to_midi(src_dec, os.path.join(save_path, f'{i}_source.midi'))
            to_midi(tgt_dec, os.path.join(save_path, f'{i}_original.midi'))
            to_midi(sample_dec, os.path.join(save_path, f'{i}_model.midi'))

            print(f'Completed {i}/{num_samples}')

    def _c_maj(self, num_samples: int):
        """Internal function for generating, processing, and saving samples from C-Major scale.
        Args:
            num_samples: number of samples required.
        """
        dataset = self.dataset
        model = self.model
        save_path = self.save_path

        # TODO: Store this in a separate file.
        cmaj = ["<S>", 60, '<M>', '<M>', '<M>', "<T>",
                       60, '<M>', '<M>', '<M>', "<T>",
                       60, '<M>', '<M>', '<M>', "<T>",
                       60, '<M>', '<M>', '<M>', "<T>",
                       62, '<M>', '<M>', '<M>', "<T>",
                       62, '<M>', '<M>', '<M>', "<T>",
                       62, '<M>', '<M>', '<M>', "<T>",
                       62, '<M>', '<M>', '<M>', "<T>",
                       64, '<M>', '<M>', '<M>', "<T>",
                       64, '<M>', '<M>', '<M>', "<T>",
                       64, '<M>', '<M>', '<M>', "<T>", 
                       64, '<M>', '<M>', '<M>', "<T>", 
                       65, '<M>', '<M>', '<M>', "<T>", 
                       65, '<M>', '<M>', '<M>', "<T>", 
                       65, '<M>', '<M>', '<M>', "<T>", 
                       65, '<M>', '<M>', '<M>', "<T>", 
                       67, '<M>', '<M>', '<M>', "<T>", 
                       67, '<M>', '<M>', '<M>', "<T>", 
                       67, '<M>', '<M>', '<M>', "<T>", 
                       67, '<M>', '<M>', '<M>', "<T>", 
                       69, '<M>', '<M>', '<M>', "<T>", 
                       69, '<M>', '<M>', '<M>', "<T>", 
                       69, '<M>', '<M>', '<M>', "<T>", 
                       69, '<M>', '<M>', '<M>', "<T>", 
                       71, '<M>', '<M>', '<M>', "<T>", 
                       71, '<M>', '<M>', '<M>', "<T>", 
                       71, '<M>', '<M>', '<M>', "<T>", 
                       71, '<M>', '<M>', '<M>', "<T>", 
                       72, '<M>', '<M>', '<M>', "<T>", 
                       72, '<M>', '<M>', '<M>', "<T>", 
                       72, '<M>', '<M>', '<M>', "<T>", 
                       72, '<M>', '<M>', '<M>', "<E>"]

        to_midi(cmaj, os.path.join(save_path, f'cmajor_source.midi'))
        src = dataset._encode(cmaj)

        # Sample and save
        for i in range(1, num_samples):
            sample_enc = gibbs_sample(model, dataset, src)
            sample_dec = dataset.decode(sample_enc)
            to_midi(sample_dec, os.path.join(save_path, f'cmajor_model_{i}.midi'))

            print(f'Completed {i}/{num_samples}')
                
        
# TODO: Add a more sophisticated gibbs sampling procedure 
def gibbs_sample(model: ChoraleBertModel, dataset: ChoraleDataset, seq: torch.tensor):
    """Generates samples according to a simplistic gibbs sampling procedure.
    Args:
        model: ChoraleBertModel instance to use to create samples.
        dataset: ChoraleDataset class to get encode decode functions from.
        seq: torch.tensor of encoded prompt to be harmonised.
    Returns:
        seq: torch.tensor of sequence harmonised using gibbs sampling.
    """
    # Gibbs sampling hyperparameters
    num_step = 1500
    block_size = 5
    temp = np.linspace(1, 0.01, num_step).tolist() # Linear temperature decrease

    seq = torch.clone(seq)
    mask_key = dataset.token_to_key['<M>']
    uniform_dist = torch.where(seq == mask_key, 1., 0.)
    uniform_dist = uniform_dist / torch.linalg.norm(uniform_dist)

    for j in range(num_step):
        curr_temp = temp[j]
        idx = torch.multinomial(uniform_dist, block_size, replacement=False)
        seq[idx] = mask_key
        logits = model.forward(seq.reshape(1, -1)) / curr_temp # Shape (1, seq_len, vocab_len)
        probs = torch.nn.functional.softmax(logits[0, idx, :], dim=1) # Shape (block_size, vocab_len)
        
        for i in range(block_size):
            seq[idx[i]] = torch.multinomial(probs[i], 1)
            
    return seq

def to_midi(seq: list, save_path: str):
    """Processes and saves an unencoded chorale sequence into midi.
    Args:
        seq: chorale as unencoded sequence.
        save_path: save path for midi file.
    """
    if '<P>' in seq:
        print('Sequence cannot contain <P>')
        return

    midi_res = MIDIFile(removeDuplicates=False, deinterleave=False) # Error without these options
    midi_res.addProgramChange(0, 0, 0, 0) # Piano = 1, Organ = 20
    midi_res.addTempo(0, 0, 160) # 160 BPM
    
    STATIC_TOKENS = ['<S>', '<E>', '<M>', '<T>', '<P>'] 

    # Reformat sequence into 4 different channels
    tok_idx = 0
    chord_idx = 0
    seq_reformatted = [[], [], [], []]
    for token in seq:
        if token in STATIC_TOKENS:
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
        
    # Create midi file according to channels of reformatted sequence
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
                    if curr_time == seq_len and note != -1: # Case: last note
                        midi_res.addNote(0, 0, pitch=note, duration=1, time=time_buffer, volume=100) # Play final note
                        
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