from anticipation.convert import events_to_midi
from midi2audio import FluidSynth
file_path = '/home/xiruij/anticipation/datasets/finetune_subset/tokenized-events-song_v2-shuffled.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()
    line = lines[1234].strip()

events = line.split()
events = events[1:-1]
events = [int(event) for event in events]
mid = events_to_midi(events)
mid.save('generated.mid')
fs = FluidSynth()  # Create a FluidSynth instance for each process
fs.midi_to_audio('generated.mid', 'generated.wav')