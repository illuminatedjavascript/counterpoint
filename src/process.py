import json

class JSBCtoSeq():
    """Loads, formats, and saves the JSB Chorales dataset into sequence form."""
    def __init__(self, load_path: str, save_path: str = '', seq_len: int = 16):
        self.PAD_TOKEN = '<P>'
        self.seq_len = seq_len
        
        raw = self.load_data(load_path)
        self.data = self.process_data(raw)
        self.save_data(save_path)

    def process_data(self, raw):
        raw = raw['test'] + raw['train'] + raw['valid']
        
        # Slice into seq_len chunks
        bars = []
        for chorale in raw:
            bars = bars + [chorale[n*self.seq_len: (n+1)*self.seq_len] for n in range(0, 1 + len(chorale)//self.seq_len)] # Make into sliding door?
            
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
    
    @staticmethod
    def load_data(load_path: str):
        with open(load_path) as f:
            raw = json.load(f)
        
        return raw
    
    def save_data(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.data, f)


def main():
    seq_len = 32
    file = JSBCtoSeq('../data/raw/Jsb16thSeparated.json', f'../data/processed/jsb{seq_len}seq.json', seq_len)


if __name__ == '__main__':
    main()