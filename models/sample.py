from midiutil import MIDIFile

# Really shit change this
def to_midi(seq, dataset):
    track = 0
    channel = 0
    time = 0 # In beats
    volume = 100 # 0-127, as per the MIDI standard
    midi_res = MIDIFile(1)
    
    for token in seq:
        if token in dataset.STATIC_TOKENS:
            if token == '<T>':
                time += 2
            continue
        
        midi_res.addNote(track, channel, pitch=int(token), duration=2, time=time, volume=volume)
    
    with open("./data/output/major-scale.mid", "wb") as f:
        midi_res.writeFile(f)