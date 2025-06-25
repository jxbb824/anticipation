import torch
import torchaudio
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, ClapModel
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np

def get_audio_features_batch(audio_paths, model, feature_extractor, device, target_sample_rate=48000):
    """
    Extracts audio features for a batch of audio files.
    Resamples audio if necessary.
    """
    processed_audios = []
    for audio_path in audio_paths:
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.ndim > 1 and waveform.shape[0] > 1: # Stereo to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
            processed_audios.append(waveform.squeeze().numpy())
        except Exception as e:
            print(f"Error loading or processing {audio_path}: {e}")
            # Add a dummy array of zeros if an audio file is corrupted or cannot be processed
            # This helps maintain the batch size and order, but introduces potential issues if not handled later
            processed_audios.append(np.zeros(target_sample_rate)) # 1 second of silence as placeholder

    if not processed_audios:
        return torch.empty(0, model.config.projection_dim).to(device)

    inputs = feature_extractor(processed_audios, sampling_rate=target_sample_rate, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        audio_features = model.get_audio_features(**inputs)
    return audio_features

def main():
    train_audio_dir = "/home/xiruij/anticipation/datasets/finetune_subset/song_train_wav"
    test_audio_dir = "/home/xiruij/anticipation/datasets/finetune_subset/song_test_wav"
    output_file = "/home/xiruij/anticipation/checkpoints_clap_new/audio_similarity_matrix.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CLAP model and feature extractor...")
    model_id = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(model_id).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model.eval()

    print("Loading datasets...")
    # Load datasets using audiofolder, which preserves order if files are sorted by name
    # Ensure files are .wav and get their paths, then sort them
    train_files = sorted([os.path.join(train_audio_dir, f) for f in os.listdir(train_audio_dir) if f.endswith(".wav")])
    # test_files = sorted([os.path.join(test_audio_dir, f) for f in os.listdir(test_audio_dir) if f.endswith(".wav")])[:100]
    test_files = sorted([os.path.join(test_audio_dir, f) for f in os.listdir(test_audio_dir) if f.endswith(".wav")])


    if not train_files:
        print(f"No .wav files found in {train_audio_dir}")
        return
    if not test_files:
        print(f"No .wav files found in {test_audio_dir}")
        return

    print(f"Found {len(train_files)} train audio files.")
    print(f"Found {len(test_files)} test audio files.")

    batch_size = 16 # Adjust based on your GPU memory

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
    # Normalize features
    train_features_norm = F.normalize(train_features_all, p=2, dim=1)
    test_features_norm = F.normalize(test_features_all, p=2, dim=1)

    # Calculate cosine similarity: (N, D) @ (M, D).T = (N, M)
    similarity_matrix = torch.matmul(train_features_norm, test_features_norm.T)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    print(f"Saving similarity matrix to {output_file}...")
    torch.save(similarity_matrix, output_file)
    print("Done.")

if __name__ == "__main__":
    main()
