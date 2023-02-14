import os
import json
import mido
from midi_utils import pianoroll_to_midi, midi_to_pianoroll


class ProcessMIDI():
    """Processes JSB Fugue MIDI files into same format as the JSBChorales
    dataset.
    Args:
        load_dir: directory to load raw midi files from.
        midi_save_dir: directory to save quantised midi files.
        json_save_path: path to save raw json data.
        div: amount each beat should be quantised into.
    """
    def __init__(self, load_dir: str, midi_save_dir: str, json_save_path, div: int = 4):
        formatted_rolls = []

        for file in os.listdir(load_dir):
            if file.endswith(".midi"):
                # Load raw MIDI
                mid = mido.MidiFile(os.path.join(load_dir, file))
                
                # Special case that the Fugue has 5 voices (occurs for 1 fugue)
                if len(mid.tracks) > 5:
                    print(f'More than 4 voices for {mid.tracks[0][0]}')

                    #  Remove 2nd voice
                    mid.tracks.pop(2)
                    adj_file = file[:-5] + '_2rm.midi' # Adjust name

                    # Process fugue with second voice removed
                    quant_roll = midi_to_pianoroll(mid, div)
                    pianoroll_to_midi(quant_roll, os.path.join(midi_save_dir, adj_file), div) # With adjusted file name
                    formatted_rolls.append(self._format_pianoroll(quant_roll))

                    # Reload midi and remove 3rd voice
                    mid = mido.MidiFile(os.path.join(load_dir, file))
                    mid.tracks.pop(3)
                    adj_file = file[:-5] + '_3rm.midi' # Adjust name
                    
                    # Process fugue with second voice removed
                    quant_roll = midi_to_pianoroll(mid, div)
                    pianoroll_to_midi(quant_roll, os.path.join(midi_save_dir, adj_file), div) # With adjusted file name
                    formatted_rolls.append(self._format_pianoroll(quant_roll))
                    
                else:
                    # Quantise and save MIDI
                    quant_roll = midi_to_pianoroll(mid, div)
                    pianoroll_to_midi(quant_roll, os.path.join(midi_save_dir, file), div)

                    # Format quantised roll and append to list
                    formatted_rolls.append(self._format_pianoroll(quant_roll))
                
        # Save json
        self._save_data(formatted_rolls, json_save_path)

    def _format_pianoroll(self, roll: list):
        """Formats piano roll from shape (4, n) to shape (n, 4)."""
        res = [[a, b, c, d] for a, b, c, d in zip(*roll)]
        
        return res

    def _save_data(self, formatted_rolls: list, save_path: str):
        """Internal function for saving processed data.
        Args:
            save_path: path to save processed dataset.
        """
        with open(save_path, 'w') as f:
            json.dump(formatted_rolls, f)


def main():
    div = 8
    midi_save_dir = f'../data/raw/fugue/midi{div*4}sep'
    json_save_path = f'../data/raw/fugue/fugue{div*4}sep.json'
    if not os.path.exists(midi_save_dir):
        os.mkdir(midi_save_dir)

    proc = ProcessMIDI('../data/raw/fugue/raw_midi', midi_save_dir, json_save_path, div)

if __name__ == '__main__':
    main()