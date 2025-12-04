import torch
import torchaudio
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, ClapModel
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import numpy as np

def get_audio_features_batch(audio_paths, model, feature_extractor, device, target_sample_rate=48000):
    """
    Extract audio features for a batch of audio files.
    Resamples audio if necessary.
    
    Args:
        audio_paths: List of audio file paths
        model: CLAP model
        feature_extractor: Audio feature extractor
        device: Computation device (cpu/cuda)
        target_sample_rate: Target sampling rate for audio processing
    
    Returns:
        Audio features tensor
    """
    processed_audios = []
    for audio_path in audio_paths:
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Convert stereo to mono if necessary
            if waveform.ndim > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to target sample rate if necessary
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
            processed_audios.append(waveform.squeeze().numpy())
        except Exception as e:
            print(f"Error loading or processing {audio_path}: {e}")
            # Add a dummy array of zeros if an audio file is corrupted
            # 1 second of silence as placeholder
            processed_audios.append(np.zeros(target_sample_rate))

    if not processed_audios:
        return torch.empty(0, model.config.projection_dim).to(device)

    inputs = feature_extractor(processed_audios, sampling_rate=target_sample_rate, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        audio_features = model.get_audio_features(**inputs)
    return audio_features

def main():
    parser = argparse.ArgumentParser(description="Calculate audio similarity using CLAP model")
    parser.add_argument("--file_type", default="mp3", choices=["wav", "mp3", "flac", "ogg"], 
                        help="Audio file format")
    parser.add_argument("--batch", type=int, default=16, 
                        help="Batch size for processing")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Training audio directory")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Test audio directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path")
    args = parser.parse_args()
    
    ext = f".{args.file_type}"
    train_audio_dir = args.train_dir
    test_audio_dir = args.test_dir
    output_file = args.output
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CLAP model and feature extractor...")
    model_id = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(model_id).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model.eval()

    print("Loading audio files...")
    # Get audio file paths and sort them to maintain consistent ordering
    train_files = sorted([os.path.join(train_audio_dir, f) for f in os.listdir(train_audio_dir) if f.endswith(ext)])
    test_files = sorted([os.path.join(test_audio_dir, f) for f in os.listdir(test_audio_dir) if f.endswith(ext)])

    if not train_files:
        print(f"No {ext} files found in {train_audio_dir}")
        return
    if not test_files:
        print(f"No {ext} files found in {test_audio_dir}")
        return

    print(f"Found {len(train_files)} train audio files.")
    print(f"Found {len(test_files)} test audio files.")

    batch_size = args.batch

    print("Extracting features for training set...")
    train_features_list = []
    for i in tqdm(range(0, len(train_files), batch_size)):
        batch_paths = train_files[i:i+batch_size]
        batch_features = get_audio_features_batch(batch_paths, model, feature_extractor, device)
        train_features_list.append(batch_features.cpu())
    
    if not train_features_list:
        print("No features extracted for training set. Exiting.")
        return
    train_features_all = torch.cat(train_features_list, dim=0)
    print(f"Train features shape: {train_features_all.shape}")

    print("Extracting features for test set...")
    test_features_list = []
    for i in tqdm(range(0, len(test_files), batch_size)):
        batch_paths = test_files[i:i+batch_size]
        batch_features = get_audio_features_batch(batch_paths, model, feature_extractor, device)
        test_features_list.append(batch_features.cpu())

    if not test_features_list:
        print("No features extracted for test set. Exiting.")
        return
    test_features_all = torch.cat(test_features_list, dim=0)
    print(f"Test features shape: {test_features_all.shape}")

    print("Calculating cosine similarity matrix...")
    # Normalize features for cosine similarity computation
    train_features_norm = F.normalize(train_features_all, p=2, dim=1)
    test_features_norm = F.normalize(test_features_all, p=2, dim=1)

    # Calculate cosine similarity: (N_train, D) @ (N_test, D).T = (N_train, N_test)
    similarity_matrix = torch.matmul(train_features_norm, test_features_norm.T)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    print(f"Saving similarity matrix to {output_file}...")
    torch.save(similarity_matrix, output_file)
    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    main()
