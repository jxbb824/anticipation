import torch
import os
import shutil
import pandas as pd
import numpy as np
import random
from pathlib import Path
import argparse

# Global selectors for slicing 4D similarity tensor: (layers, pooling, train, eval)
# If similarity is 2D, these are ignored.
SIMILARITY_LAYER_INDEX = -1  # e.g., last layer
SIMILARITY_POOLING_INDEX = 0  # e.g., 0 for mean, 1 for max (depends on your data)

# Default ranks specification for each group (TDA and SIMILARITY)
# Supports dynamic items relative to N (number of training samples): "N/2", "N-100", "N"
DEFAULT_RANKS_SPEC = [1, 10, 100, "N/2", "N-100", "N"]

def _detect_audio_extension(audio_dir):
    """
    Detect commonly used audio extension in a directory.
    Returns extension like '.wav', '.mp3', etc.
    """
    all_files = os.listdir(audio_dir)
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        if any(f.endswith(ext) for f in all_files):
            return ext
    return None

def _slice_similarity_2d(similarity_tensor):
    """
    Convert similarity tensor to 2D (train, eval) using the global selectors
    when it is 4D with shape (layers, pooling, train, eval). If it is already
    2D, return as-is.
    """
    if similarity_tensor.dim() == 4:
        layers, poolings, _, _ = similarity_tensor.shape
        layer_idx = SIMILARITY_LAYER_INDEX % layers
        pooling_idx = SIMILARITY_POOLING_INDEX % poolings
        return similarity_tensor[layer_idx, pooling_idx, :, :]
    if similarity_tensor.dim() == 2:
        return similarity_tensor
    raise ValueError(
        f"Unsupported similarity tensor dims: {similarity_tensor.dim()}D; expected 2D or 4D"
    )

def _resolve_ranks(rank_specs, num_train):
    """
    Resolve dynamic rank specs (like 'N/2', 'N-100', 'N') to concrete 1-based ranks.
    Clamps into [1, num_train].
    """
    resolved = []
    for spec in rank_specs:
        if isinstance(spec, int):
            r = spec
        elif isinstance(spec, str):
            s = spec.strip().upper()
            if s == 'N':
                r = num_train
            elif s == 'N/2' or s == 'N\u00f72':  # guard in case of unicode slash
                r = max(1, int(round(num_train / 2)))
            elif s.startswith('N-'):
                try:
                    k = int(s.split('-', 1)[1])
                except ValueError:
                    raise ValueError(f"Invalid rank spec: {spec}")
                r = max(1, num_train - k)
            else:
                try:
                    r = int(s)
                except ValueError:
                    raise ValueError(f"Invalid rank spec: {spec}")
        else:
            raise ValueError(f"Invalid rank spec type: {type(spec)}")
        r = max(1, min(num_train, r))
        resolved.append(r)
    return resolved

def _sorted_indices_desc(tensor_1d):
    """Return indices that would sort tensor values descending."""
    return torch.argsort(tensor_1d, descending=True)

def generate_pairs_for_subset(
    score_matrix_path,
    similarity_matrix_path,
    train_audio_dir,
    eval_audio_dir,
    output_subset_dir,
    subset_label,
    num_eval_samples=10,
    ranks_spec=DEFAULT_RANKS_SPEC,
    audio_extension=None,
    random_seed=42
):
    """
    Build 12 pairs for each selected eval sample (6 from TDA by influence score ranks,
    6 from SIMILARITY by similarity score ranks). Copies audio files into a structured
    directory and returns pair-level metadata and eval-sample-level metadata.

    - score_matrix: 2D tensor [num_train, num_eval]
    - similarity_matrix: 2D or 4D tensor; sliced to 2D [num_train, num_eval]
    - subset_label: 'test' or 'generation'
    """
    print(f"Loading influence score matrix: {score_matrix_path}")
    score_matrix = torch.load(score_matrix_path, map_location=torch.device('cpu'))
    if score_matrix.dim() != 2:
        raise ValueError(f"Score matrix must be 2D, got {score_matrix.dim()}D")

    print(f"Loading similarity matrix: {similarity_matrix_path}")
    similarity_tensor = torch.load(similarity_matrix_path, map_location=torch.device('cpu'))
    similarity_2d = _slice_similarity_2d(similarity_tensor)
    if similarity_2d.shape != score_matrix.shape:
        raise ValueError(
            f"Shape mismatch: score {score_matrix.shape} vs similarity {similarity_2d.shape}"
        )

    os.makedirs(output_subset_dir, exist_ok=True)

    # Determine audio extension if not provided
    if audio_extension is None:
        audio_extension = _detect_audio_extension(train_audio_dir)
        if audio_extension is None:
            raise ValueError("Cannot detect audio extension, please set audio_extension explicitly")
        print(f"Detected audio format: {audio_extension}")

    train_files = sorted([f for f in os.listdir(train_audio_dir) if f.endswith(audio_extension)])
    eval_files = sorted([f for f in os.listdir(eval_audio_dir) if f.endswith(audio_extension)])
    num_train = len(train_files)
    if num_train == 0:
        raise ValueError(f"No training audio files with extension {audio_extension} found in {train_audio_dir}")
    if len(eval_files) == 0:
        raise ValueError(f"No eval audio files with extension {audio_extension} found in {eval_audio_dir}")

    resolved_ranks = _resolve_ranks(ranks_spec, num_train)
    print(f"Using ranks for subset '{subset_label}': {resolved_ranks}")

    # Sample eval indices reproducibly
    num_eval_cols = score_matrix.shape[1]
    valid_eval_count = min(num_eval_cols, len(eval_files))
    if valid_eval_count < len(eval_files):
        print(
            f"Warning: found {len(eval_files)} eval files but only {num_eval_cols} columns in matrices; "
            f"restricting to first {valid_eval_count} files to match matrix columns."
        )
    rng = random.Random(random_seed)
    k = min(num_eval_samples, valid_eval_count)
    if k <= 0:
        raise ValueError("No valid eval samples available after restricting to matrix columns.")
    eval_indices_all = list(range(valid_eval_count))
    rng.shuffle(eval_indices_all)
    selected_eval_indices = sorted(eval_indices_all[:k])
    print(f"Selected {k} eval samples for subset '{subset_label}' (within first {valid_eval_count}) with seed {random_seed}")

    # Metadata
    pairs_records = []
    eval_records = []

    # Iterate selected eval samples
    for local_idx, eval_idx in enumerate(selected_eval_indices, 1):
        eval_filename = eval_files[eval_idx]
        eval_dir = os.path.join(output_subset_dir, f"sample_{subset_label}_{local_idx:03d}")
        os.makedirs(eval_dir, exist_ok=True)

        # Copy eval audio
        eval_src = os.path.join(eval_audio_dir, eval_filename)
        eval_dst = os.path.join(eval_dir, eval_filename)
        shutil.copy2(eval_src, eval_dst)

        # Column vectors for this eval sample
        tda_col = score_matrix[:, eval_idx]
        sim_col = similarity_2d[:, eval_idx]

        # Pre-compute sorted indices for both criteria
        tda_sorted = _sorted_indices_desc(tda_col)
        sim_sorted = _sorted_indices_desc(sim_col)

        # Eval-level record
        eval_records.append({
            'subset': subset_label,
            'eval_local_id': local_idx,
            'eval_original_index': eval_idx,
            'eval_filename': eval_filename,
            'eval_path': eval_dst,
            'num_train': num_train,
            'ranks_spec': ",".join(str(x) for x in ranks_spec),
            'ranks_resolved': ",".join(str(x) for x in resolved_ranks),
            'similarity_layer_index': SIMILARITY_LAYER_INDEX,
            'similarity_pooling_index': SIMILARITY_POOLING_INDEX
        })

        # For each group (TDA and SIMILARITY), pick the training samples at given ranks
        for group_name, sorted_idx_tensor in (("TDA", tda_sorted), ("SIMILARITY", sim_sorted)):
            for rank_requested, rank_resolved in zip(ranks_spec, resolved_ranks):
                zero_based = rank_resolved - 1
                if zero_based >= len(sorted_idx_tensor):
                    continue  # should not happen due to clamping, but be safe
                train_idx = int(sorted_idx_tensor[zero_based].item())
                train_filename = train_files[train_idx]

                # Copy train audio
                base, ext = os.path.splitext(train_filename)
                out_train_name = f"group_{group_name}_rank_{rank_resolved:05d}_{base}{ext}"
                train_src = os.path.join(train_audio_dir, train_filename)
                train_dst = os.path.join(eval_dir, out_train_name)
                shutil.copy2(train_src, train_dst)

                # Add pair record
                pairs_records.append({
                    'subset': subset_label,
                    'group': group_name,
                    'rank_requested': str(rank_requested),
                    'rank_resolved': rank_resolved,
                    'eval_local_id': local_idx,
                    'eval_original_index': eval_idx,
                    'eval_filename': eval_filename,
                    'eval_path': eval_dst,
                    'train_index': train_idx,
                    'train_filename': train_filename,
                    'train_path': train_dst,
                    'influence_score': float(tda_col[train_idx].item()),
                    'similarity_score': float(sim_col[train_idx].item()),
                    'similarity_layer_index': SIMILARITY_LAYER_INDEX,
                    'similarity_pooling_index': SIMILARITY_POOLING_INDEX,
                    'num_train': num_train
                })

    # Save subset metadata
    pairs_df = pd.DataFrame(pairs_records)
    eval_df = pd.DataFrame(eval_records)
    subset_pairs_csv = os.path.join(output_subset_dir, f"pairs_metadata_{subset_label}.csv")
    subset_eval_csv = os.path.join(output_subset_dir, f"eval_samples_{subset_label}.csv")
    pairs_df.to_csv(subset_pairs_csv, index=False)
    eval_df.to_csv(subset_eval_csv, index=False)

    print(f"Finished subset '{subset_label}'. Pairs saved to: {subset_pairs_csv}")
    return pairs_df, eval_df

def generate_experiment_pairs(
    test_score_matrix_path,
    test_similarity_matrix_path,
    generation_score_matrix_path,
    generation_similarity_matrix_path,
    train_audio_dir,
    test_audio_dir,
    generation_audio_dir,
    output_dir,
    num_test_eval_samples=10,
    num_generation_eval_samples=10,
    ranks_spec=DEFAULT_RANKS_SPEC,
    audio_extension=None,
    random_seed=42
):
    """
    Generate 20 eval samples (10 test + 10 generation) and 12 pairs per eval sample
    (6 TDA + 6 SIMILARITY). Writes structured directories and comprehensive metadata.
    Returns the combined pairs DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, 'test')
    gen_dir = os.path.join(output_dir, 'generation')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    test_pairs_df, test_eval_df = generate_pairs_for_subset(
        score_matrix_path=test_score_matrix_path,
        similarity_matrix_path=test_similarity_matrix_path,
        train_audio_dir=train_audio_dir,
        eval_audio_dir=test_audio_dir,
        output_subset_dir=test_dir,
        subset_label='test',
        num_eval_samples=num_test_eval_samples,
        ranks_spec=ranks_spec,
        audio_extension=audio_extension,
        random_seed=random_seed
    )

    gen_pairs_df, gen_eval_df = generate_pairs_for_subset(
        score_matrix_path=generation_score_matrix_path,
        similarity_matrix_path=generation_similarity_matrix_path,
        train_audio_dir=train_audio_dir,
        eval_audio_dir=generation_audio_dir,
        output_subset_dir=gen_dir,
        subset_label='generation',
        num_eval_samples=num_generation_eval_samples,
        ranks_spec=ranks_spec,
        audio_extension=audio_extension,
        random_seed=random_seed + 1  # different seed for generation subset
    )

    combined_pairs = pd.concat([test_pairs_df, gen_pairs_df], ignore_index=True)
    combined_pairs_csv = os.path.join(output_dir, 'pairs_metadata_all.csv')
    combined_pairs.to_csv(combined_pairs_csv, index=False)

    combined_eval = pd.concat([test_eval_df, gen_eval_df], ignore_index=True)
    combined_eval_csv = os.path.join(output_dir, 'eval_samples_all.csv')
    combined_eval.to_csv(combined_eval_csv, index=False)

    print("\nAll done!")
    print(f"- Combined pairs metadata: {combined_pairs_csv}")
    print(f"- Combined eval samples metadata: {combined_eval_csv}")
    print(f"- Output directory: {output_dir}")
    return combined_pairs

def extract_influential_samples(
    score_matrix_path,
    similarity_matrix_path,
    train_audio_dir,
    test_audio_dir,
    output_dir,
    num_test_samples=30,
    ranks_to_extract=[1, 5, 10, 100, 1000],
    audio_extension=None,
    random_seed=42
):
    """
    Randomly select num_test_samples test samples and extract corresponding influential training samples.

    Args:
        score_matrix_path: Path to influence score matrix (2D tensor [train, test])
        similarity_matrix_path: Path to similarity matrix (2D [train, test] or 4D [layers, poolings, train, test])
                                If 4D, automatically uses last layer mean (0) and max (1) pooling for reporting
        train_audio_dir: Directory containing training audio files
        test_audio_dir: Directory containing test audio files
        output_dir: Output directory
        num_test_samples: Number of test samples to process
        ranks_to_extract: List of 1-based ranks to extract by influence score
        audio_extension: Audio extension (e.g., '.wav', '.mp3'); auto-detected if None
        random_seed: Random seed for reproducible selection (default 42)
    """
    print(f"Loading score matrix: {score_matrix_path}")
    score_matrix = torch.load(score_matrix_path, map_location=torch.device('cpu'))

    print(f"Loading similarity matrix: {similarity_matrix_path}")
    similarity_matrix = torch.load(similarity_matrix_path, map_location=torch.device('cpu'))

    # Handle multi-dimensional similarity matrices (shape: (layers, pooling_types, train, test))
    # Extract the last layer with both mean and max pooling for reference
    similarity_mean = None
    similarity_max = None

    if similarity_matrix.dim() == 4:
        original_shape = similarity_matrix.shape
        print(f"Detected 4D similarity tensor, shape: {original_shape}")
        print("Auto-extracting last layer (index -1): mean pooling (0) and max pooling (1)...")
        similarity_mean = similarity_matrix[-1, 0, :, :]  # Last layer, mean pooling
        similarity_max = similarity_matrix[-1, 1, :, :]   # Last layer, max pooling
        print(f"Extracted similarity shapes: mean={similarity_mean.shape}, max={similarity_max.shape}")
    elif similarity_matrix.dim() == 2:
        # 2D matrix, use as-is
        print(f"Detected 2D similarity matrix, shape: {similarity_matrix.shape}")
        similarity_mean = similarity_matrix
        similarity_max = similarity_matrix
    else:
        raise ValueError(f"Unsupported similarity matrix dims: {similarity_matrix.dim()}D, expected 2D or 4D")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect audio extension if not provided
    if audio_extension is None:
        all_files = os.listdir(train_audio_dir)
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            if any(f.endswith(ext) for f in all_files):
                audio_extension = ext
                print(f"Detected audio format: {audio_extension}")
                break
        if audio_extension is None:
            raise ValueError("Could not auto-detect audio format. Please specify audio_extension explicitly")

    # Build file lists (sorted for determinism)
    train_files = sorted([f for f in os.listdir(train_audio_dir) if f.endswith(audio_extension)])
    all_test_files = sorted([f for f in os.listdir(test_audio_dir) if f.endswith(audio_extension)])

    # Seed and select random test samples
    random.seed(random_seed)
    num_test_samples = min(num_test_samples, len(all_test_files))

    # Shuffle indices and take first N
    test_indices = list(range(len(all_test_files)))
    random.shuffle(test_indices)
    selected_test_indices = sorted(test_indices[:num_test_samples])  # sort to process in index order

    print(f"Selected {num_test_samples} test samples with seed {random_seed}")
    print(f"Selected test indices: {selected_test_indices[:10]}..." if len(selected_test_indices) > 10 else f"Selected test indices: {selected_test_indices}")

    # Prepare metadata
    metadata = []

    print(f"Start processing {num_test_samples} randomly selected test samples...")

    # Process each selected test sample
    for sample_num, original_test_idx in enumerate(selected_test_indices, 1):
        test_file = all_test_files[original_test_idx]
        print(f"Processing test sample {sample_num}/{num_test_samples}: {test_file} (original index: {original_test_idx})")

        # Create output dir for this test sample
        test_sample_dir = os.path.join(output_dir, f"test_sample_{sample_num}")
        os.makedirs(test_sample_dir, exist_ok=True)

        # Copy test sample into output directory, preserving original filename
        test_file_path = os.path.join(test_audio_dir, test_file)
        test_output_path = os.path.join(test_sample_dir, test_file)
        shutil.copy2(test_file_path, test_output_path)

        # Column for this test sample (by original index)
        score_col = score_matrix[:, original_test_idx]

        # Similarity columns (mean and max) by original index
        similarity_mean_col = similarity_mean[:, original_test_idx]
        similarity_max_col = similarity_max[:, original_test_idx]

        # Sort by influence score (descending)
        sorted_indices = torch.argsort(score_col, descending=True)

        # Prepare metadata record for this test sample
        sample_record = {
            'test_sample_id': sample_num,
            'test_sample_original_index': original_test_idx,
            'test_sample_filename': test_file,
            'test_sample_path': test_output_path
        }

        # For each requested rank, copy the corresponding training sample
        for rank in ranks_to_extract:
            if rank - 1 < len(sorted_indices):
                train_idx = sorted_indices[rank - 1].item()
                train_file = train_files[train_idx]

                # Similarity scores (mean and max)
                similarity_mean_score = similarity_mean_col[train_idx].item()
                similarity_max_score = similarity_max_col[train_idx].item()

                # Influence score
                influence_score = score_col[train_idx].item()

                # Copy training sample, prefixing rank in filename
                train_file_path = os.path.join(train_audio_dir, train_file)
                train_file_basename = os.path.splitext(train_file)[0]
                train_file_ext = os.path.splitext(train_file)[1]
                output_train_filename = f"rank_{rank:05d}_{train_file_basename}{train_file_ext}"
                output_train_path = os.path.join(test_sample_dir, output_train_filename)
                shutil.copy2(train_file_path, output_train_path)

                # Append to record
                sample_record[f'rank_{rank}_train_index'] = train_idx
                sample_record[f'rank_{rank}_train_filename'] = train_file
                sample_record[f'rank_{rank}_train_path'] = output_train_path
                sample_record[f'rank_{rank}_influence_score'] = influence_score
                sample_record[f'rank_{rank}_similarity_mean'] = similarity_mean_score
                sample_record[f'rank_{rank}_similarity_max'] = similarity_max_score

        metadata.append(sample_record)

    # Write metadata CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"\nDone!")
    print(f"Randomly selected and extracted {num_test_samples} test samples and their influential training samples")
    print(f"Random seed: {random_seed}")
    print(f"Metadata saved to: {metadata_csv_path}")
    print(f"All audio files saved to: {output_dir}")
    print(f"\nNotes:")
    print(f"- Test sample keeps its original filename")
    print(f"- Training sample filename format: rank_XXXXX_<original>{audio_extension}")
    print(f"- metadata.csv includes:")
    print(f"  * influence_score: influence score")
    print(f"  * similarity_mean: similarity score (mean pooling)")
    print(f"  * similarity_max: similarity score (max pooling)")

    return metadata_df

def main():
    global SIMILARITY_LAYER_INDEX, SIMILARITY_POOLING_INDEX
    parser = argparse.ArgumentParser(description="Generate paired audio samples for evaluation (TDA vs Similarity).")
    parser.add_argument("--test_score", required=False, help="Path to test subset influence score matrix (2D tensor)")
    parser.add_argument("--test_similarity", required=False, help="Path to test subset similarity tensor (2D or 4D)")
    parser.add_argument("--gen_score", required=False, help="Path to generation subset influence score matrix (2D tensor)")
    parser.add_argument("--gen_similarity", required=False, help="Path to generation subset similarity tensor (2D or 4D)")
    parser.add_argument("--train_audio_dir", required=False, help="Directory with training audio files")
    parser.add_argument("--test_audio_dir", required=False, help="Directory with test audio files")
    parser.add_argument("--gen_audio_dir", required=False, help="Directory with generation audio files")
    parser.add_argument("--output_dir", required=False, default="./extracted_pairs_output", help="Output directory")
    parser.add_argument("--num_test", type=int, default=10, help="Number of eval samples from test subset")
    parser.add_argument("--num_gen", type=int, default=10, help="Number of eval samples from generation subset")
    parser.add_argument("--ranks", default="1,10,100,N/2,N-100,N", help="Comma-separated ranks spec")
    parser.add_argument("--audio_ext", default=None, help="Audio extension to filter (e.g., .mp3). If omitted, auto-detect from train dir")
    parser.add_argument("--seed", type=int, default=114514, help="Random seed")
    parser.add_argument("--sim_layer", type=int, default=SIMILARITY_LAYER_INDEX, help="Layer index for 4D similarity tensor (first dim)")
    parser.add_argument("--sim_pooling", type=int, default=SIMILARITY_POOLING_INDEX, help="Pooling index for 4D similarity tensor (second dim)")

    args = parser.parse_args()

    # Allow overriding global selectors via CLI
    SIMILARITY_LAYER_INDEX = args.sim_layer
    SIMILARITY_POOLING_INDEX = args.sim_pooling

    # If all four matrices and dirs are provided, run the full 20*12 generation; else print guidance
    required_args = [
        args.test_score, args.test_similarity, args.gen_score, args.gen_similarity,
        args.train_audio_dir, args.test_audio_dir, args.gen_audio_dir
    ]
    if all(required_args):
        ranks_spec = [s.strip() for s in args.ranks.split(',') if s.strip()]
        # Convert numeric strings to int where possible
        parsed_ranks = []
        for x in ranks_spec:
            try:
                parsed_ranks.append(int(x))
            except ValueError:
                parsed_ranks.append(x)

        generate_experiment_pairs(
            test_score_matrix_path=args.test_score,
            test_similarity_matrix_path=args.test_similarity,
            generation_score_matrix_path=args.gen_score,
            generation_similarity_matrix_path=args.gen_similarity,
            train_audio_dir=args.train_audio_dir,
            test_audio_dir=args.test_audio_dir,
            generation_audio_dir=args.gen_audio_dir,
            output_dir=args.output_dir,
            num_test_eval_samples=args.num_test,
            num_generation_eval_samples=args.num_gen,
            ranks_spec=parsed_ranks,
            audio_extension=args.audio_ext,
            random_seed=args.seed
        )
    else:
        print("Missing required arguments for full experiment generation.")
        print("You can still use extract_influential_samples() by calling it programmatically, or rerun with:")
        print(
            "--test_score --test_similarity --gen_score --gen_similarity --train_audio_dir "
            "--test_audio_dir --gen_audio_dir --output_dir [--num_test 10 --num_gen 10 --ranks 1,10,100,N/2,N-100,N]"
        )

if __name__ == "__main__":
    main()
