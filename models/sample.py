import os
import math
import json
import torch
import random
from midiutil import MIDIFile
from .model import ChoraleBertModel
from .dataset import ChoraleDataset


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
        #self._gen_fugues_random(12)
        #self._gen_fugues_prompt('./data/processed/fugue16sep64len_prompts.json', 16)
        self._test_set(10)
        
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
            src, tgt = dataset.sample_test()
            src_dec = dataset.decode(src)
            tgt_dec = dataset.decode(tgt)
            while '<P>' in src_dec:
                src, tgt = dataset.sample_test()
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

    def _gen_fugues_random(self, num_bars: int):
        """Internal function for generating a full length fugues.
        Args:
            num_bars: the number of bars to be added to the original prompt.
        """
        def _gen_fugue(model: ChoraleBertModel, dataset: ChoraleDataset, num_bars: int):
            """Given a the start of a fugue, complete it by adding num_bars more bars."""
            # Gibbs hyperparams
            alpha_max = 1. 
            alpha_min = 0.05
            num_steps = 200 
            neta = 0.75
            temp_max = 1.
            temp_min = 0.6

            TOKENS_PER_BAR = 5*16
            MASK_PER_BAR = 4*16
            MASKED_BAR = (['<T>'] + ['<M>']*4)*16
            PAD_BAR = (['<P>']*5)*16

            res =  MASKED_BAR*num_bars + ['<E>'] + PAD_BAR
            
            for n in range(num_steps):
                for _ in range(num_bars):
                    # Chose random bar number to resample
                    mid_bar_num = random.randint(0, num_bars-3)
                    
                    if mid_bar_num == num_bars-3: # Case: one pad bar included
                        total_to_mask = MASK_PER_BAR*3
                    else: # Case: no prompt or pad bars included 
                        total_to_mask = MASK_PER_BAR*4
                        
                    # Calculate Gibbs samplying params
                    mask_prob = max(alpha_min,
                                    alpha_max -
                                    ((n*(alpha_max - alpha_min)) / (neta*num_steps))
                                    )

                    temp = temp_max + n*((temp_min - temp_max)/(num_steps)) 
                    block_size = max(1, math.trunc(mask_prob * total_to_mask))
                    
                    # Get bar for resampling 
                    if mid_bar_num == num_bars-3: # Case: there is padding at the end
                        resample_bar = res[(mid_bar_num)*TOKENS_PER_BAR: (mid_bar_num + 4)*TOKENS_PER_BAR + 1]
                    else: # Case: no padding
                        resample_bar = res[(mid_bar_num)*TOKENS_PER_BAR: (mid_bar_num + 4)*TOKENS_PER_BAR] + ['<E>']
                    
                    resample_bar[0] = '<S>'
                    resample_bar_enc = dataset._encode(resample_bar)

                    # Gibbs step
                    gibbs_bar_enc = gibbs_step(model, dataset, temp, block_size, resample_bar_enc, mid_bar_num, num_bars)
                    gibbs_bar_dec = dataset.decode(gibbs_bar_enc)
                    
                    # Reformat
                    if mid_bar_num != 0:
                        gibbs_bar_dec[0] = '<T>' # Revert <S> back to <T>
                    gibbs_bar_dec.pop() # Remove <E> tag 

                    # Insert into res
                    res[(mid_bar_num)*TOKENS_PER_BAR: (mid_bar_num + 4)*TOKENS_PER_BAR] = gibbs_bar_dec
                
                print(f'{n+1}/{num_steps} steps completed.')

            return res
        
        def gibbs_step(model: ChoraleBertModel, dataset: ChoraleDataset,
                       temp: float, block_size: int, seq: torch.tensor,
                       mid_bar_num: int, num_bars: int):
            """Performs a single gibbs step according to the parameters. Note
            only works with 64len."""
            TOKENS_PER_BAR = 5*16
            CHORD_PER_BAR = 16
            EQ_CHORD = [0., 1., 1., 1., 1.]

            seq = torch.clone(seq) 
            mask_key = dataset.token_to_key['<M>']

            # Generate idx sampling distribution
            if mid_bar_num == num_bars-3:
                uniform_dist = torch.tensor(EQ_CHORD*(CHORD_PER_BAR*3) + [0.]*TOKENS_PER_BAR + [0.])
            else:
                uniform_dist = torch.tensor(EQ_CHORD*(CHORD_PER_BAR*4) + [0.]) 

            uniform_dist = uniform_dist / torch.linalg.norm(uniform_dist)
            
            # Gibbs sample step
            idx = torch.multinomial(uniform_dist, block_size, replacement=False)
            seq[idx] = mask_key
            logits = model.forward(seq.reshape(1, -1)) / temp # Shape (1, seq_len, vocab_len)
            probs = torch.nn.functional.softmax(logits[0, idx, :], dim=1) # Shape (block_size, vocab_len)
            
            for i in range(block_size):
                seq[idx[i]] = torch.multinomial(probs[i], 1)

            return seq

        dataset = self.dataset
        model = self.model
        save_path = self.save_path
        
        for i in range(10):
            res = _gen_fugue(model, dataset, num_bars)
            to_midi(res, os.path.join(save_path, f'f{i+1}.midi'))
            print(f'Finished {i+1}/{10}')
            
    def _gen_fugues_prompt(self, load_path: str, num_bars: int):
        """Internal function for generating a full length fugues.
        Args:
            num_bars: the number of bars to be added to the original prompt.
        """
        def _gen_fugue(model: ChoraleBertModel, dataset: ChoraleDataset, prompt: list, num_bars: int):
            """Given a the start of a fugue, complete it by adding num_bars more bars."""
            tokens_per_bar = 5*16
            MASKED_BAR = (['<T>'] + ['<M>']*4)*16
            PAD_BAR = (['<P>']*5)*16
            
            res = prompt.copy()[:3*tokens_per_bar]

            # First iter
            curr = res.copy()[:3*tokens_per_bar] + MASKED_BAR + ['<E>']
            curr_enc = dataset._encode(curr)
            curr_comp_enc = gibbs_sample(model, dataset, curr_enc)
            curr_comp_dec = dataset.decode(curr_comp_enc)
            res = res + curr_comp_dec[tokens_per_bar*3:tokens_per_bar*4]
            
            # Add all apart from last bar
            for i in range(1, num_bars-1):
                curr = ['<S>'] + res.copy()[i*tokens_per_bar + 1: (i+3)*tokens_per_bar] + MASKED_BAR + ['<E>']
                curr_enc = dataset._encode(curr)
                curr_comp_enc = gibbs_sample(model, dataset, curr_enc)
                curr_comp_dec = dataset.decode(curr_comp_enc)
                res = res + curr_comp_dec[tokens_per_bar*3:tokens_per_bar*4]
                
            # Add the ending 
            curr = res.copy()[-2*tokens_per_bar:] + MASKED_BAR + ['<E>'] + PAD_BAR
            curr_enc = dataset._encode(curr)
            curr_comp_enc = gibbs_sample(model, dataset, curr_enc)
            curr_comp_dec = dataset.decode(curr_comp_enc)
            res = res + curr_comp_dec[tokens_per_bar*2:tokens_per_bar*3]
            
            return res + ['<E>']

        dataset = self.dataset
        model = self.model
        save_path = self.save_path
        
        with open(load_path) as f:
            prompts = json.load(f)
            
        for i, prompt in enumerate(prompts):
            res = _gen_fugue(model, dataset, prompt, num_bars)
            to_midi(res, os.path.join(save_path, f'f{i+1}.midi'))
            print(f'Finished {i+1}/{len(prompts)}')
            

def gibbs_sample(model: ChoraleBertModel, dataset: ChoraleDataset, seq: torch.tensor):
    """Generates samples according to a simplistic gibbs sampling procedure.
    Args:
        model: ChoraleBertModel instance to use to create samples.
        dataset: ChoraleDataset class to get encode decode functions from.
        seq: torch.tensor of encoded prompt to be harmonised.
    Returns:
        seq: torch.tensor of sequence harmonised using gibbs sampling.
    """
    # Hyperparams from 'Counterpoint by Convolution' paper
    alpha_max = 1. 
    alpha_min = 0.05
    num_steps = 200
    neta = 0.75

    # Hyperparams for tempertature scaling
    temp_max = 1.
    temp_min = 0.6

    seq = torch.clone(seq) 
    mask_key = dataset.token_to_key['<M>']
    total_to_mask = torch.sum(seq == mask_key).item()
    uniform_dist = torch.where(seq == mask_key, 1., 0.)
    uniform_dist = uniform_dist / torch.linalg.norm(uniform_dist)
    
    # Gibbs sampling
    for n in range(num_steps):
        
        # Calc masking rate and temperature
        temp = temp_max + n*((temp_min - temp_max)/(num_steps)) 
        mask_prob = max(alpha_min,
                        alpha_max -
                        ((n*(alpha_max - alpha_min)) / (neta*num_steps))
                        )
        
        block_size = max(1, math.trunc(mask_prob * total_to_mask))
        idx = torch.multinomial(uniform_dist, block_size, replacement=False)
        seq[idx] = mask_key
        logits = model.forward(seq.reshape(1, -1)) / temp # Shape (1, seq_len, vocab_len)
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
        print('Sequences contains, <P>. Removing all such occurances...')
        print('Original:')
        print(seq)
        seq = [i for i in seq if i != '<P>']
        print('New:')
        print(seq)

    midi_res = MIDIFile(removeDuplicates=False, deinterleave=False) # Error without these options
    midi_res.addProgramChange(0, 0, 0, 0) # Piano = 0, Organ = 20
    midi_res.addTempo(0, 0, 240) # 240 BPM
    
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
