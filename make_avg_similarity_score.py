import argparse
import os
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="similarity tensor .pt (L×2×N×M)")
    ap.add_argument("--out", required=True, help="output .pt path (N×M)")
    args = ap.parse_args()

    sim = torch.load(args.sim, map_location="cpu")
    if not torch.is_tensor(sim):
        raise TypeError("Loaded object is not a torch.Tensor")

    if sim.ndim != 4 or sim.shape[1] < 1:
        raise ValueError(f"Expect 4-D (L,2,N,M) tensor, got {tuple(sim.shape)}")

    # pooling=mean 通道（索引0），在层维(L)对其求平均 → (N, M)
    sim_mean_pool = sim[:, 0, :, :]                 # (L, N, M)
    avg_score = sim_mean_pool.mean(dim=0).to(torch.float32)  # (N, M)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(avg_score, args.out)
    print(f"Saved averaged score tensor {tuple(avg_score.shape)} to: {args.out}")

if __name__ == "__main__":
    main()
