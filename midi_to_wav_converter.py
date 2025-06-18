import os
from multiprocessing import Pool
from midi2audio import FluidSynth
import time

# Define file paths
input_file = '/home/xiruij/anticipation/datasets/clap/test_unique.txt'
midi_dir = '/home/xiruij/anticipation/datasets/finetune_subset/song'
output_dir = '/home/xiruij/anticipation/datasets/clap/song_test'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert MIDI to WAV
def convert_midi_to_wav(item):
    index, midi_path, output_path = item
    fs = FluidSynth()  # Create a FluidSynth instance for each process
    fs.midi_to_audio(midi_path, output_path)
    return midi_path, output_path

# Read the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Process each line and collect conversion tasks
conversion_tasks = []
for i, line in enumerate(lines):
    # Extract the last word from the line
    last_word = line.strip().split()[-1]
    
    # Search for MIDI files that contain the last word
    matching_files = [f for f in os.listdir(midi_dir) if last_word in f and f.endswith('.mid')]
    
    # If matching files found, sort them and select the first one
    if matching_files:
        matching_files.sort()  # Sort alphabetically
        selected_midi = matching_files[0]
        midi_path = os.path.join(midi_dir, selected_midi)
        
        # Create output filename with sequential numbering
        output_filename = f"{i:06d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Add to conversion tasks
        conversion_tasks.append((i, midi_path, output_path))
    else:
        print(f"Warning: Could not find MIDI file for {last_word}")

# Process conversions in parallel
start_time = time.time()
num_workers = 8
print(f"Starting conversion of {len(conversion_tasks)} files using {num_workers} workers...")

# Use multiprocessing to convert files
with Pool(processes=num_workers) as pool:
    results = pool.map(convert_midi_to_wav, conversion_tasks)

# for midi_path, output_path in results:
#     print(f"Converted {os.path.basename(midi_path)} to {os.path.basename(output_path)}")

elapsed_time = time.time() - start_time
print(f"Conversion complete. Generated {len(results)} WAV files in {output_dir}")
print(f"Processing time: {elapsed_time:.2f} seconds")
