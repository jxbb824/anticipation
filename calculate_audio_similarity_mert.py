import os
import argparse
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def load_waveform(path: str, target_sr: int) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return wav.squeeze().numpy()


@torch.no_grad()
def extract_batch_features(paths, model, processor, device):
    sr = processor.sampling_rate
    waves = []
    for p in paths:
        try:
            waves.append(load_waveform(p, sr))
        except Exception:
            waves.append(np.zeros(sr, dtype=np.float32))

    inputs = processor(
        waves, sampling_rate=sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    hs = torch.stack(model(**inputs, output_hidden_states=True).hidden_states)
    # hs: (L, B, T, D)
    mean_feats = hs.mean(2).cpu().permute(1, 0, 2)  # (B, L, D)
    max_feats = hs.max(2).values.cpu().permute(1, 0, 2)
    return mean_feats, max_feats


def aggregate_features(files, batch, model, processor, device):
    mean_list, max_list = [], []
    for i in tqdm(range(0, len(files), batch)):
        m, x = extract_batch_features(files[i : i + batch], model, processor, device)
        mean_list.append(m)
        max_list.append(x)
    mean_all = torch.cat(mean_list, 0)  # (N, L, D)
    max_all = torch.cat(max_list, 0)    # (N, L, D)
    return mean_all, max_all


def cosine_sim(a, b):
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return a_n @ b_n.T  # (N_train, N_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_type", default="mp3", choices=["wav", "mp3"])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    ext = f".{args.file_type}"

    train_dir = "/home/xiruij/anticipation/datasets/finetune/song_train_mp3"
    test_dir = "/home/xiruij/anticipation/datasets/finetune/song_test_mp3"
    out_path = "/home/xiruij/anticipation/checkpoints_subset_large/audio_similarity_all_layers.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    train_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(ext)])
    test_files  = sorted([os.path.join(test_dir,  f) for f in os.listdir(test_dir)  if f.endswith(ext)])[:500]

    if not train_files or not test_files:
        raise RuntimeError("No audio files found. Check paths and --file_type.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

    print("Extracting train features …")
    train_mean, train_max = aggregate_features(train_files, args.batch, model, processor, device)
    print("Extracting test features …")
    test_mean,  test_max  = aggregate_features(test_files,  args.batch, model, processor, device)

    L = train_mean.shape[1]
    N_train, N_test = train_mean.shape[0], test_mean.shape[0]
    sim_tensor = torch.empty(L, 2, N_train, N_test, dtype=torch.float32)

    for l in range(L):
        sim_tensor[l, 0] = cosine_sim(train_mean[:, l], test_mean[:, l])  # mean pooling
        sim_tensor[l, 1] = cosine_sim(train_max[:, l],  test_max[:, l])   # max  pooling

    torch.save(sim_tensor, out_path)
    print(f"Saved similarity tensor to {out_path} with shape {tuple(sim_tensor.shape)}")


if __name__ == "__main__":
    main()
