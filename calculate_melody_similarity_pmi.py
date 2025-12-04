import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from numba import jit

from anticipation.convert import events_to_compound
from anticipation.vocab import AUTOREGRESS


def read_token_lines(path, is_generated=False):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_generated:
                # generated_samples.txt: no first/last tokens, keep all tokens as-is
                toks = np.fromstring(line, sep=' ', dtype=np.int64)
                if toks.size == 0:
                    toks = np.array([], dtype=np.int64)
            else:
                # standard format (train/test): drop first token and last token (file id)
                parts = line.split()
                if len(parts) >= 3:
                    toks = np.array([int(x) for x in parts[1:-1]], dtype=np.int64)
                else:
                    toks = np.array([], dtype=np.int64)
            lines.append(toks)
    return lines


def events_to_compound_matrix(tokens_np):
    tokens = tokens_np.tolist()
    comp = events_to_compound(tokens, debug=False)
    arr = np.array(comp, dtype=np.int64).reshape(-1, 5)
    # columns: time_ticks, duration_ticks, pitch(0-127), instrument(0-128), velocity
    return arr


def extract_melody_pitch_sequence(compound_arr, target_instr=66):
    # Filter target instrument rows and sort by time
    rows = compound_arr[compound_arr[:, 3] == target_instr]
    if rows.shape[0] == 0:
        return np.array([], dtype=np.int64)
    rows = rows[np.argsort(rows[:, 0])]
    # Extract pitch class sequence (ignore duration/time; paper counts identical pitch classes)
    pitches = rows[:, 2].astype(np.int64)
    pitch_classes = np.mod(pitches, 12)
    return pitch_classes


# Needleman–Wunsch with affine gap penalties
# GOP (gap open penalty) = 12, GEP (gap extension) = 6 (Savage et al.)
# Match = 1 if same pitch class; Mismatch = 0 (only identical counted)

@jit(nopython=True)
def needleman_wunsch_affine(seq_a, seq_b, gop=12, gep=6):
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return 0, 0  # matches, align_len
    if n == 0:
        return 0, m
    if m == 0:
        return 0, n

    # DP matrices: M (match), X (gap in A), Y (gap in B)
    NEG_INF = -10**9
    M = np.full((n+1, m+1), NEG_INF, dtype=np.int32)
    X = np.full((n+1, m+1), NEG_INF, dtype=np.int32)
    Y = np.full((n+1, m+1), NEG_INF, dtype=np.int32)
    # Also store aligned length and match counts for backtracking-free accumulation
    ML = np.zeros((n+1, m+1), dtype=np.int32)
    XL = np.zeros((n+1, m+1), dtype=np.int32)
    YL = np.zeros((n+1, m+1), dtype=np.int32)
    MM = np.zeros((n+1, m+1), dtype=np.int32)
    XM = np.zeros((n+1, m+1), dtype=np.int32)
    YM = np.zeros((n+1, m+1), dtype=np.int32)

    M[0, 0] = 0
    # Initialize first row/col with gaps
    for i in range(1, n+1):
        X[i, 0] = -(gop + (i-1)*gep)
        XL[i, 0] = i
    for j in range(1, m+1):
        Y[0, j] = -(gop + (j-1)*gep)
        YL[0, j] = j

    for i in range(1, n+1):
        ai = seq_a[i-1]
        for j in range(1, m+1):
            bj = seq_b[j-1]
            match_score = 1 if ai == bj else 0

            # Update M
            cand_m = M[i-1, j-1]
            cand_x = X[i-1, j-1]
            cand_y = Y[i-1, j-1]
            if cand_m >= cand_x and cand_m >= cand_y:
                best_idx = 0
                best_val = cand_m
            elif cand_x >= cand_y:
                best_idx = 1
                best_val = cand_x
            else:
                best_idx = 2
                best_val = cand_y
            M[i, j] = best_val + match_score
            if best_idx == 0:
                ML[i, j] = ML[i-1, j-1] + 1
                MM[i, j] = MM[i-1, j-1] + (1 if match_score == 1 else 0)
            elif best_idx == 1:
                ML[i, j] = XL[i-1, j-1] + 1
                MM[i, j] = XM[i-1, j-1] + (1 if match_score == 1 else 0)
            else:
                ML[i, j] = YL[i-1, j-1] + 1
                MM[i, j] = YM[i-1, j-1] + (1 if match_score == 1 else 0)

            # Update X (gap in A → insert in B)
            open_from_M = M[i-1, j] - gop
            extend_from_X = X[i-1, j] - gep
            if open_from_M >= extend_from_X:
                X[i, j] = open_from_M
                XL[i, j] = ML[i-1, j] + 1
                XM[i, j] = MM[i-1, j]
            else:
                X[i, j] = extend_from_X
                XL[i, j] = XL[i-1, j] + 1
                XM[i, j] = XM[i-1, j]

            # Update Y (gap in B → insert in A)
            open_from_M = M[i, j-1] - gop
            extend_from_Y = Y[i, j-1] - gep
            if open_from_M >= extend_from_Y:
                Y[i, j] = open_from_M
                YL[i, j] = ML[i, j-1] + 1
                YM[i, j] = MM[i, j-1]
            else:
                Y[i, j] = extend_from_Y
                YL[i, j] = YL[i, j-1] + 1
                YM[i, j] = YM[i, j-1]

    # Take best of three endings
    end_m = M[n, m]
    end_x = X[n, m]
    end_y = Y[n, m]
    if end_m >= end_x and end_m >= end_y:
        best = 0
    elif end_x >= end_y:
        best = 1
    else:
        best = 2
    
    if best == 0:
        matches = MM[n, m]
        alen = ML[n, m]
    elif best == 1:
        matches = XM[n, m]
        alen = XL[n, m]
    else:
        matches = YM[n, m]
        alen = YL[n, m]

    return int(matches), int(alen)


def compute_pid4_matrix(train_tokens, test_tokens, target_instr=66, gop=12, gep=6, transposition_invariant=False):
    train_compounds = [events_to_compound_matrix(t) for t in train_tokens]
    test_compounds = [events_to_compound_matrix(t) for t in test_tokens]

    train_melodies = [extract_melody_pitch_sequence(c, target_instr) for c in train_compounds]
    test_melodies = [extract_melody_pitch_sequence(c, target_instr) for c in test_compounds]

    N_train, N_test = len(train_melodies), len(test_melodies)
    pid4 = torch.zeros(N_train, N_test, dtype=torch.float32)

    for i in tqdm(range(N_train), desc='PID4 train rows'):
        a = train_melodies[i]
        for j in range(N_test):
            b = test_melodies[j]
            denom = float(len(a) + len(b)) / 2.0
            if len(a) == 0 or len(b) == 0 or denom == 0.0:
                pid4[i, j] = 0.0
                continue
            
            if transposition_invariant:
                # Try all 12 transpositions and take maximum
                best_pid4 = 0.0
                for t in range(12):
                    b_transposed = (b + t) % 12
                    matches, _ = needleman_wunsch_affine(a, b_transposed, gop=gop, gep=gep)
                    cur_pid4 = float(matches) / denom
                    if cur_pid4 > best_pid4:
                        best_pid4 = cur_pid4
                pid4[i, j] = best_pid4
            else:
                matches, _ = needleman_wunsch_affine(a, b, gop=gop, gep=gep)
                pid4[i, j] = float(matches) / denom
    return pid4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_file', type=str, required=True)
    ap.add_argument('--test_file', type=str, required=True)
    ap.add_argument('--out_path', type=str, required=True)
    ap.add_argument('--target_instr', type=int, default=66)
    ap.add_argument('--max_test', type=int, default=500)
    ap.add_argument('--gop', type=int, default=12)
    ap.add_argument('--gep', type=int, default=6)
    ap.add_argument('--test_is_generated', action='store_true',
                    help='If set, treat test_file as generated samples: prepend AUTOREGRESS and do not drop last token.')
    ap.add_argument('--transposition_invariant', action='store_true',
                    help='If set, try all 12 transpositions and take maximum PID4 (detects transposed melodies).')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    train_lines = read_token_lines(args.train_file, is_generated=False)
    test_lines = read_token_lines(args.test_file, is_generated=args.test_is_generated)[:args.max_test]

    pid4 = compute_pid4_matrix(train_lines, test_lines, target_instr=args.target_instr, gop=args.gop, gep=args.gep, transposition_invariant=args.transposition_invariant)
    torch.save(pid4, args.out_path)
    print(f'Saved PID4 matrix to {args.out_path} with shape {tuple(pid4.shape)}')


if __name__ == '__main__':
    main()
