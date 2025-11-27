import os
from multiprocessing import Pool
from midi2audio import FluidSynth
from anticipation.convert import events_to_midi
import time
import argparse
from pydub import AudioSegment
import tempfile
import pretty_midi

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Convert event sequences from a text file to audio files (WAV or MP3).')
parser.add_argument('--format', type=str, default='wav', choices=['wav', 'mp3'], help='Output audio format (wav or mp3).')
parser.add_argument('--improve-midi', action='store_true', help='Apply MIDI improvements (adjust velocity and instruments).')
args = parser.parse_args()

# Define file paths
input_file = '/home/xiruij/anticipation/datasets/finetune/train_v2.txt'
# input_file = '/home/xiruij/anticipation/datasets/finetune/generated_samples_prompted.txt'
output_dir = f'/home/xiruij/anticipation/datasets/finetune/song_train_{args.format}_improved'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to apply MIDI improvements
def improve_midi(midi_path):
    """
    Apply improvements to MIDI file:
    - Adjust click track (program 10) to be drums with specific pitches
    - Set harmony (program 0) velocity to 90
    - Set melody (program 66) velocity to 60
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    for instrument in midi.instruments:
        # Click track
        if instrument.program == 10:
            instrument.program = 0
            instrument.is_drum = True
            for note in instrument.notes:
                if note.pitch == 36:
                    # Downbeat
                    note.pitch = 34
                else:
                    note.pitch = 33
                note.velocity = 70
        elif instrument.program == 0:
            # Harmony
            for note in instrument.notes:
                note.velocity = 90
        elif instrument.program == 66:
            # Melody
            for note in instrument.notes:
                note.velocity = 60
    
    # Overwrite the original file with improved version
    midi.write(midi_path)

# Function to convert events to audio
def convert_events_to_audio(item):
    index, line, output_path = item
    events = line.split()
    events = events[1:-1]  # Remove first and last tokens
    events = [int(event) for event in events]
    
    # Convert events to MIDI
    mid = events_to_midi(events)
    
    # Create temporary MIDI file in a process-safe way
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi_file:
        temp_midi_path = temp_midi_file.name
        mid.save(temp_midi_path)
    
    # Apply MIDI improvements if enabled
    if args.improve_midi:
        improve_midi(temp_midi_path)

    fs = FluidSynth()

    try:
        if args.format == 'mp3':
            # Create a temporary path for the WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
                temp_wav_path = temp_wav_file.name
            
            # 1. Convert MIDI to WAV
            fs.midi_to_audio(temp_midi_path, temp_wav_path)
            
            # 2. Convert WAV to MP3
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(output_path, format="mp3")
            
            # 3. Clean up temporary WAV file
            os.remove(temp_wav_path)
        else: # 'wav' format
            # Convert MIDI directly to WAV
            fs.midi_to_audio(temp_midi_path, output_path)

    finally:
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
        output_filename = f"{i:06d}.{args.format}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Add to conversion tasks
        conversion_tasks.append((i, line, output_path))

# Process conversions in parallel
start_time = time.time()
num_workers = 8
print(f"Starting conversion of {len(conversion_tasks)} files to {args.format.upper()} using {num_workers} workers...")

# Use multiprocessing to convert files
with Pool(processes=num_workers) as pool:
    results = pool.map(convert_events_to_audio, conversion_tasks)

elapsed_time = time.time() - start_time
print(f"Conversion complete. Generated {len(results)} {args.format.upper()} files in {output_dir}")
print(f"Processing time: {elapsed_time:.2f} seconds")