import os
from multiprocessing import Pool
from anticipation.convert import events_to_midi
import time
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Convert event sequences from a text file to MIDI files.')
args = parser.parse_args()

# Define file paths
test_file = '/home/xiruij/anticipation/datasets/finetune_subset/test_v2.txt'
train_file = '/home/xiruij/anticipation/datasets/finetune_subset/train_v2.txt'
output_dir = '/home/xiruij/anticipation/datasets/finetune_subset/song_midi_selected'

# Define specific line indices to extract (0-based)
test_indices = [3, 4, 9]
train_indices = [853, 1334, 1374]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert events to MIDI
def convert_events_to_midi(item):
    filename, line, output_path = item
    events = line.split()
    events = events[1:-1]  # Remove first and last tokens
    events = [int(event) for event in events]
    
    # Convert events to MIDI
    mid = events_to_midi(events)
    
    # Save MIDI file directly
    mid.save(output_path)
    
    return filename, output_path

# Read test file and extract specific lines
conversion_tasks = []

print("Reading test file...")
with open(test_file, 'r') as f:
    test_lines = f.readlines()

for idx in test_indices:
    if idx < len(test_lines):
        line = test_lines[idx].strip()
        if line:
            output_filename = f"test_{idx:06d}.mid"
            output_path = os.path.join(output_dir, output_filename)
            conversion_tasks.append((output_filename, line, output_path))

print("Reading train file...")
with open(train_file, 'r') as f:
    train_lines = f.readlines()

for idx in train_indices:
    if idx < len(train_lines):
        line = train_lines[idx].strip()
        if line:
            output_filename = f"train_{idx:06d}.mid"
            output_path = os.path.join(output_dir, output_filename)
            conversion_tasks.append((output_filename, line, output_path))

# Process conversions in parallel
start_time = time.time()
num_workers = 8
print(f"Starting conversion of {len(conversion_tasks)} files to MIDI using {num_workers} workers...")

# Use multiprocessing to convert files
with Pool(processes=num_workers) as pool:
    results = pool.map(convert_events_to_midi, conversion_tasks)

elapsed_time = time.time() - start_time
print(f"Conversion complete. Generated {len(results)} MIDI files in {output_dir}")
print(f"Processing time: {elapsed_time:.2f} seconds")

