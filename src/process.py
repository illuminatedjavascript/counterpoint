import json

class JSBCtoSeq():
    """Loads, formats, and saves the raw JSB Chorales dataset into sequence form.
    Args
        load_path: path to load raw dataset.
        save_path: path to save processed dataset.
        seq_len: number of time steps to include in each data point.
    """
    def __init__(self, load_path: str, save_path: str = '', seq_len: int = 16):
        self.bar_len = 16
        self.PAD_TOKEN = '<P>'
        self.seq_len = seq_len
        
        raw = self._load_data(load_path)
        self.data = self._process_data(raw)
        self._save_data(save_path)

    def _process_data(self, raw):
        """Processes raw dataset into appropriately sized sequences.
        Args:
            raw: list of raw data.
        Returns:
            data: list of processed data.
        """
        raw = raw['test'] + raw['train'] + raw['valid']
        
        # Slice into seq_len chunks (sliding door with step_size = bar_length)
        bars = []
        for chorale in raw:
            bars = bars + [chorale[n*self.bar_len: (n*self.bar_len + self.seq_len)] for n in range(0, 1 + len(chorale)//self.bar_len)] 
            
        # Sequentialise and pad
        seq_data = []
        for bar in bars:
            if bar == []:
                continue
            seq = ['<S>']
            seq.append(bar[0][0])
            seq.append(bar[0][1])
            seq.append(bar[0][2])
            seq.append(bar[0][3])
            
            for chord in bar[1:]:
                seq.append('<T>')
                seq.append(chord[0])
                seq.append(chord[1])
                seq.append(chord[2])
                seq.append(chord[3])

            seq.append('<E>')
            
            # pad_num = #notes + #'<T>' + #'<S>' (=1) + #'<E>' (=1)
            pad_num = (self.seq_len*4) + (self.seq_len-1) + 2
            seq += [self.PAD_TOKEN]*(pad_num - len(seq))
            seq_data.append(seq)
        
        data = seq_data

        return data
    
    def _load_data(load_path: str):
        """Internal function for loading raw data.
        Args:
            load_path: path to raw dataset.
        Returns:
            raw: raw data as list.
        """
        with open(load_path) as f:
            raw = json.load(f)
        
        return raw
    
    def _save_data(self, save_path):
        """Internal function for saving processed data.
        Args:
            save_path: path to save processed dataset.
        """
        with open(save_path, 'w') as f:
            json.dump(self.data, f)


def main():
    seq_len = 32
    file = JSBCtoSeq('../data/raw/Jsb16thSeparated.json', f'../data/processed/jsb{seq_len}slide.json', seq_len)


if __name__ == '__main__':
    main()