import os
from multiprocessing import Pool
from midi2audio import FluidSynth
from anticipation.convert import events_to_midi
import time

# Define file paths
input_file = '/home/xiruij/anticipation/datasets/finetune_subset/test_v2.txt'
output_dir = '/home/xiruij/anticipation/datasets/finetune_subset/song_test_wav'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert events to WAV
def convert_events_to_wav(item):
    index, line, output_path = item
    events = line.split()
    events = events[1:-1]  # Remove first and last tokens
    events = [int(event) for event in events]
    
    # Convert events to MIDI
    mid = events_to_midi(events)
    
    # Create temporary MIDI file
    temp_midi_path = f"/tmp/temp_{index}.mid"
    mid.save(temp_midi_path)
    
    # Convert MIDI to WAV
    fs = FluidSynth()  # Create a FluidSynth instance for each process
    fs.midi_to_audio(temp_midi_path, output_path)
    
    # Clean up temporary MIDI file
    os.remove(temp_midi_path)
    
    return index, output_path

# Read the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Process each line and collect conversion tasks
conversion_tasks = []
for i, line in enumerate(lines):
    line = line.strip()
    if line:  # Skip empty lines
        # Create output filename with sequential numbering
        output_filename = f"{i:06d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Add to conversion tasks
        conversion_tasks.append((i, line, output_path))

# Process conversions in parallel
start_time = time.time()
num_workers = 8
print(f"Starting conversion of {len(conversion_tasks)} files using {num_workers} workers...")

# Use multiprocessing to convert files
with Pool(processes=num_workers) as pool:
    results = pool.map(convert_events_to_wav, conversion_tasks)

elapsed_time = time.time() - start_time
print(f"Conversion complete. Generated {len(results)} WAV files in {output_dir}")
print(f"Processing time: {elapsed_time:.2f} seconds")
