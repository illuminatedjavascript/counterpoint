import mido
import midiutil

def midi_to_pianoroll(mid: mido.MidiFile, div: int = 4): # ADD TRACK ORDERING FOR VOICES
    """Loads a midi into pianorolls of according to tracks"""

    def _track_to_pianoroll(track: mido.MidiTrack, div):
        """Converts MidiTrack into pianoroll."""
        tick = int(mid.ticks_per_beat/(div))
        res = []

        # Skip starting meta messages
        idx = 0
        while track[idx].is_meta == True:
            idx += 1
        curr_msg = track[idx]

        # Parse MIDI track
        curr_time = 0
        next_msg_time = curr_msg.time

        # Process first message
        while curr_time < next_msg_time:
            res.append(-1)
            curr_time += tick
            
        prev_msg = curr_msg
        idx += 1
        curr_msg = track[idx]
        next_msg_time += curr_msg.time

        # Parsing the rest of the midi
        while True:
            # Break of end of track
            if curr_msg.is_meta == True:
                if curr_msg.type == 'end_of_track':
                    break
                else:
                    idx += 1
                    curr_msg = track[idx]
                    next_msg_time += curr_msg.time
                    continue
            
            # Change message appropriate
            if curr_time >= next_msg_time:
                idx += 1
                prev_msg = curr_msg
                curr_msg = track[idx]
                next_msg_time += curr_msg.time
                continue
            
            # Append token
            if prev_msg.velocity == 0:
                res.append(-1)
            else:
                res.append(prev_msg.note)
            
            curr_time += tick

        return res
    
    # Add voices to pianoroll
    pianorolls = [None, None, None, None]
    for i, track in enumerate(mid.tracks[1:]):
        roll = _track_to_pianoroll(track, div)
        pianorolls[i] = roll
        
    # Set roll length as length of soprano voice
    roll_len = len(pianorolls[0])

    # If voice is not present, fill with -1
    for i, roll in enumerate(pianorolls):
        if roll == None:
            pianorolls[i] = [-1]*roll_len

    # Special case that one of the voices is of the incorrect length (this occurs in 2 fugues)
    if len(pianorolls[3]) != roll_len:
        print(f'Short roll length for {mid.tracks[0][0]}')
        pianorolls[3] += [-1]*(roll_len - len(pianorolls[3])) # Pad roll length

    assert len(pianorolls[0]) == len(pianorolls[1]) == len(pianorolls[2]) == len(pianorolls[3]), 'Roll lengths incompatible'

    return pianorolls

# Clean up
def pianoroll_to_midi(pianoroll: list, save_path: str, div: int):
    """Converts a pianoroll of sequences into midi."""
    mid = midiutil.MIDIFile(removeDuplicates=False, deinterleave=False) # Error without these options
    mid.addProgramChange(0, 0, 0, 0) # Organ=20
    mid.addTempo(0, 0, 60*div) # 240 BPM

    seq_len = len(pianoroll[0])
    for seq in pianoroll:
        assert len(seq) == seq_len, 'Incompatible sequences.'
        
    # Create midi file according to channels of reformatted sequence
    for channel in pianoroll:
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
                        mid.addNote(0, 0, pitch=note, duration=1, time=time_buffer, volume=100) # Play final note
                        
                    else: # Case: not the last note
                        note_buffer = note # Update buffer
                        time_buffer = curr_time # Update buffer

            else: # Case: silence is not in the buffer
                if note == -1: # Case: current note is silence
                    mid.addNote(0, 0, pitch=note_buffer, duration=(curr_time-time_buffer), time=time_buffer, volume=100) # Play note
                    note_buffer = note # Update buffer
                    time_buffer = curr_time # Update buffer

                elif note == note_buffer: # Case: current note is the same as buffer note
                    if curr_time == seq_len: # Case: current note is last
                        mid.addNote(0, 0, pitch=note_buffer, duration=((curr_time-time_buffer) + 1), time=time_buffer, volume=100) # Play final note

                    else: # Case: current note is not last:
                        pass # Do nothing

                elif note != note_buffer: # Case: current note is different from buffer note
                    mid.addNote(0, 0, pitch=note_buffer, duration=(curr_time-time_buffer), time=time_buffer, volume=100) # Play note
                    note_buffer = note
                    time_buffer = curr_time

    # Save
    with open(save_path, "wb") as f:
        mid.writeFile(f)