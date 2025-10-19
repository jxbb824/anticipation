import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from anticipation.convert import events_to_compound
from anticipation.config import TIME_RESOLUTION, MAX_PITCH
from anticipation.vocab import TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET


def read_token_lines(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = np.fromstring(line, sep=' ', dtype=np.int64)
            # Remove the first and last token per sample
            if toks.size >= 3:
                toks = toks[1:-1]
            else:
                # If too short, produce empty to be filtered downstream
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
            cands = [M[i-1, j-1], X[i-1, j-1], Y[i-1, j-1]]
            best_idx = int(np.argmax(cands))
            M[i, j] = cands[best_idx] + match_score
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
    ends = [M[n, m], X[n, m], Y[n, m]]
    best = int(np.argmax(ends))
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


def _best_pmi_against_batch_gpu(a_pc, b_pc_list, gop, gep, device):
    """Compute best PMI of a_pc against a batch of b sequences on GPU (12 transpositions).
    a_pc: 1D numpy array of pitch classes (len n)
    b_pc_list: list of 1D numpy arrays (variable lengths)
    returns: 1D torch.float32 tensor of PMI for each b
    """
    n = int(len(a_pc))
    B = len(b_pc_list)
    if B == 0 or n == 0:
        return torch.zeros(B, dtype=torch.float32)

    lengths = torch.tensor([len(b) for b in b_pc_list], dtype=torch.int32, device=device)
    if int(lengths.max().item()) == 0:
        return torch.zeros(B, dtype=torch.float32, device='cpu')

    m_max = int(lengths.max().item())
    # Build batch matrix of b (B, m_max) with padding -1
    b_mat = torch.full((B, m_max), -1, dtype=torch.int16, device=device)
    for idx, b in enumerate(b_pc_list):
        if len(b) > 0:
            b_mat[idx, :len(b)] = torch.tensor(b, dtype=torch.int16, device=device)

    # Build 12 transpositions (B, T=12, m_max)
    T = 12
    shifts = torch.arange(T, dtype=torch.int16, device=device).view(1, T, 1)
    b_t = (b_mat.unsqueeze(1).repeat(1, T, 1) + shifts) % 12

    # Prepare DP arrays for previous row (scores, lengths, matches)
    NEG = torch.iinfo(torch.int32).min // 4
    M_prev = torch.full((B, T, m_max + 1), NEG, dtype=torch.int32, device=device)
    X_prev = torch.full_like(M_prev, NEG)
    Y_prev = torch.full_like(M_prev, NEG)
    ML_prev = torch.zeros_like(M_prev)
    XL_prev = torch.zeros_like(M_prev)
    YL_prev = torch.zeros_like(M_prev)
    MM_prev = torch.zeros_like(M_prev)
    XM_prev = torch.zeros_like(M_prev)
    YM_prev = torch.zeros_like(M_prev)

    # Initialize row 0
    M_prev[:, :, 0] = 0
    # Y_prev[0, j] = -(gop + (j-1)*gep)
    j_idx = torch.arange(m_max + 1, device=device, dtype=torch.int32)
    # For j>=1
    Y_prev[:, :, 1:] = -(gop + (j_idx[1:].view(1, 1, -1) - 1) * gep)
    YL_prev[:, :, 1:] = j_idx[1:]

    a_vec = torch.tensor(a_pc, dtype=torch.int16, device=device)

    for i in range(1, n + 1):
        ai = a_vec[i - 1]
        # Current row buffers
        M_cur = torch.full_like(M_prev, NEG)
        X_cur = torch.full_like(M_prev, NEG)
        Y_cur = torch.full_like(M_prev, NEG)
        ML_cur = torch.zeros_like(M_prev)
        XL_cur = torch.zeros_like(M_prev)
        YL_cur = torch.zeros_like(M_prev)
        MM_cur = torch.zeros_like(M_prev)
        XM_cur = torch.zeros_like(M_prev)
        YM_cur = torch.zeros_like(M_prev)

        # j = 0 initialization
        # X[i,0]
        open_from_M = M_prev[:, :, 0] - gop
        extend_from_X = X_prev[:, :, 0] - gep
        x_take_open = open_from_M >= extend_from_X
        X_cur[:, :, 0] = torch.where(x_take_open, open_from_M, extend_from_X)
        XL_cur[:, :, 0] = torch.where(x_take_open, ML_prev[:, :, 0] + 1, XL_prev[:, :, 0] + 1)
        XM_cur[:, :, 0] = torch.where(x_take_open, MM_prev[:, :, 0], XM_prev[:, :, 0])
        # M_cur[:, :, 0] stays NEG; Y_cur[:, :, 0] stays NEG

        # Precompute match tensor for j>=1: (B, T, m_max)
        bj = b_t  # (B, T, m_max)
        match = (bj == ai).to(torch.int32)
        # For padded positions (-1), match==False already

        # Compute M for j>=1
        candM = M_prev[:, :, :-1]
        candX = X_prev[:, :, :-1]
        candY = Y_prev[:, :, :-1]
        base = torch.stack([candM, candX, candY], dim=0)  # (3, B, T, m_max)
        best_idx = torch.argmax(base, dim=0)  # (B, T, m_max)
        best_prev = torch.gather(base, 0, best_idx.unsqueeze(0)).squeeze(0)
        M_cur[:, :, 1:] = best_prev + match
        # lengths and matches propagate
        ML_src = torch.stack([ML_prev[:, :, :-1], XL_prev[:, :, :-1], YL_prev[:, :, :-1]], dim=0)
        MM_src = torch.stack([MM_prev[:, :, :-1], XM_prev[:, :, :-1], YM_prev[:, :, :-1]], dim=0)
        ML_cur[:, :, 1:] = torch.gather(ML_src, 0, best_idx.unsqueeze(0)).squeeze(0) + 1
        MM_cur[:, :, 1:] = torch.gather(MM_src, 0, best_idx.unsqueeze(0)).squeeze(0) + match

        # Compute X for all j (vertical gaps)
        open_from_M_all = M_prev - gop
        extend_from_X_all = X_prev - gep
        x_take_open_all = open_from_M_all >= extend_from_X_all
        X_cur = torch.where(x_take_open_all, open_from_M_all, extend_from_X_all)
        XL_cur = torch.where(x_take_open_all, ML_prev + 1, XL_prev + 1)
        XM_cur = torch.where(x_take_open_all, MM_prev, XM_prev)

        # Compute Y by left-to-right scan (horizontal gaps)
        # Start from j=1..m_max
        # Initialize column 0 already NEG; iterate j
        for j in range(1, m_max + 1):
            open_from_M = M_cur[:, :, j - 1] - gop
            extend_from_Y = Y_cur[:, :, j - 1] - gep
            take_open = open_from_M >= extend_from_Y
            Y_cur[:, :, j] = torch.where(take_open, open_from_M, extend_from_Y)
            YL_cur[:, :, j] = torch.where(take_open, ML_cur[:, :, j - 1] + 1, YL_cur[:, :, j - 1] + 1)
            YM_cur[:, :, j] = torch.where(take_open, MM_cur[:, :, j - 1], YM_cur[:, :, j - 1])

        # Roll to next row
        M_prev, X_prev, Y_prev = M_cur, X_cur, Y_cur
        ML_prev, XL_prev, YL_prev = ML_cur, XL_cur, YL_cur
        MM_prev, XM_prev, YM_prev = MM_cur, XM_cur, YM_cur

    # Gather end results at j = lengths for each sample
    # Build gather indices (B, 1, 1)
    idx = lengths.view(B, 1, 1).to(torch.int64)
    # Repeat across T and collapse batch dims with gather along last dim
    M_end = torch.gather(M_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)
    X_end = torch.gather(X_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)
    Y_end = torch.gather(Y_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)
    ML_end = torch.gather(ML_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)
    XL_end = torch.gather(XL_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)
    YL_end = torch.gather(YL_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)
    MM_end = torch.gather(MM_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)
    XM_end = torch.gather(XM_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)
    YM_end = torch.gather(YM_prev, 2, idx.repeat(1, 12, 1)).squeeze(-1)

    # Choose best end state by score
    end_scores = torch.stack([M_end, X_end, Y_end], dim=0)  # (3, B, T)
    best_state = torch.argmax(end_scores, dim=0)  # (B, T)

    end_len = torch.where(best_state == 0, ML_end, torch.where(best_state == 1, XL_end, YL_end))
    end_match = torch.where(best_state == 0, MM_end, torch.where(best_state == 1, XM_end, YM_end))

    # PMI per transposition and per sample (B, T)
    pmi_bt = end_match.clamp(min=0).to(torch.float32) / end_len.clamp(min=1).to(torch.float32)
    # Take best across 12 transpositions
    pmi_b = pmi_bt.max(dim=1).values  # (B,)
    return pmi_b.detach().to('cpu')


def _best_pmi_train_batch_gpu(a_list, b_pc_list, gop, gep, device):
    """Compute PMI for a batch of train melodies vs a batch of test melodies on GPU.
    a_list: list of 1D numpy arrays (pitch classes) length A
    b_pc_list: list of 1D numpy arrays (pitch classes) length B
    returns: tensor (A, B) of PMI (max over 12 transpositions)
    """
    A = len(a_list)
    B = len(b_pc_list)
    if A == 0 or B == 0:
        return torch.zeros((A, B), dtype=torch.float32)

    a_lengths = torch.tensor([len(a) for a in a_list], dtype=torch.int32, device=device)
    b_lengths = torch.tensor([len(b) for b in b_pc_list], dtype=torch.int32, device=device)
    if int(a_lengths.max().item()) == 0 or int(b_lengths.max().item()) == 0:
        return torch.zeros((A, B), dtype=torch.float32)

    n_max = int(a_lengths.max().item())
    m_max = int(b_lengths.max().item())

    # Build padded matrices
    a_mat = torch.full((A, n_max), -1, dtype=torch.int16, device=device)
    for idx, a in enumerate(a_list):
        if len(a) > 0:
            a_mat[idx, :len(a)] = torch.tensor(a, dtype=torch.int16, device=device)

    b_mat = torch.full((B, m_max), -1, dtype=torch.int16, device=device)
    for idx, b in enumerate(b_pc_list):
        if len(b) > 0:
            b_mat[idx, :len(b)] = torch.tensor(b, dtype=torch.int16, device=device)

    # Transpositions for B: (B, T, m_max)
    T = 12
    shifts = torch.arange(T, dtype=torch.int16, device=device).view(1, T, 1)
    b_t = (b_mat.unsqueeze(1).repeat(1, T, 1) + shifts) % 12  # (B,T,m)

    # Expand to (A,B,T, m)
    b_t = b_t.unsqueeze(0).repeat(A, 1, 1, 1)  # (A,B,T,m)

    NEG = torch.iinfo(torch.int32).min // 4
    # DP states: (A,B,T, m_max+1)
    M_prev = torch.full((A, B, T, m_max + 1), NEG, dtype=torch.int32, device=device)
    X_prev = torch.full_like(M_prev, NEG)
    Y_prev = torch.full_like(M_prev, NEG)
    ML_prev = torch.zeros_like(M_prev)
    XL_prev = torch.zeros_like(M_prev)
    YL_prev = torch.zeros_like(M_prev)
    MM_prev = torch.zeros_like(M_prev)
    XM_prev = torch.zeros_like(M_prev)
    YM_prev = torch.zeros_like(M_prev)

    # init j=0 and first row
    M_prev[:, :, :, 0] = 0
    j_idx = torch.arange(m_max + 1, device=device, dtype=torch.int32)
    Y_prev[:, :, :, 1:] = -(gop + (j_idx[1:].view(1, 1, 1, -1) - 1) * gep)
    YL_prev[:, :, :, 1:] = j_idx[1:]

    for i in range(1, n_max + 1):
        ai = a_mat[:, i - 1]  # (A,)
        valid_i = (i <= a_lengths).view(A, 1, 1, 1)

        M_cur = torch.full_like(M_prev, NEG)
        X_cur = torch.full_like(M_prev, NEG)
        Y_cur = torch.full_like(M_prev, NEG)
        ML_cur = torch.zeros_like(M_prev)
        XL_cur = torch.zeros_like(M_prev)
        YL_cur = torch.zeros_like(M_prev)
        MM_cur = torch.zeros_like(M_prev)
        XM_cur = torch.zeros_like(M_prev)
        YM_cur = torch.zeros_like(M_prev)

        # X at j=0
        open_from_M = M_prev[:, :, :, 0] - gop
        extend_from_X = X_prev[:, :, :, 0] - gep
        take_open = open_from_M >= extend_from_X
        X_cur[:, :, :, 0] = torch.where(take_open, open_from_M, extend_from_X)
        XL_cur[:, :, :, 0] = torch.where(take_open, ML_prev[:, :, :, 0] + 1, XL_prev[:, :, :, 0] + 1)
        XM_cur[:, :, :, 0] = torch.where(take_open, MM_prev[:, :, :, 0], XM_prev[:, :, :, 0])

        # match for j>=1
        # Broadcast ai over (A,B,T,m)
        match = (b_t == ai.view(A, 1, 1, 1)).to(torch.int32)

        # M for j>=1 using best of prev states at j-1
        candM = M_prev[:, :, :, :-1]
        candX = X_prev[:, :, :, :-1]
        candY = Y_prev[:, :, :, :-1]
        base = torch.stack([candM, candX, candY], dim=0)
        best_idx = torch.argmax(base, dim=0)
        best_prev = torch.gather(base, 0, best_idx.unsqueeze(0)).squeeze(0)
        M_cur[:, :, :, 1:] = best_prev + match
        ML_src = torch.stack([ML_prev[:, :, :, :-1], XL_prev[:, :, :, :-1], YL_prev[:, :, :, :-1]], dim=0)
        MM_src = torch.stack([MM_prev[:, :, :, :-1], XM_prev[:, :, :, :-1], YM_prev[:, :, :, :-1]], dim=0)
        ML_cur[:, :, :, 1:] = torch.gather(ML_src, 0, best_idx.unsqueeze(0)).squeeze(0) + 1
        MM_cur[:, :, :, 1:] = torch.gather(MM_src, 0, best_idx.unsqueeze(0)).squeeze(0) + match

        # Invalidate M for padded (i beyond length)
        M_cur = torch.where(valid_i, M_cur, torch.full_like(M_cur, NEG))
        ML_cur = torch.where(valid_i, ML_cur, torch.zeros_like(ML_cur))
        MM_cur = torch.where(valid_i, MM_cur, torch.zeros_like(MM_cur))

        # X for all j
        open_from_M_all = M_prev - gop
        extend_from_X_all = X_prev - gep
        take_open_all = open_from_M_all >= extend_from_X_all
        X_cur = torch.where(take_open_all, open_from_M_all, extend_from_X_all)
        XL_cur = torch.where(take_open_all, ML_prev + 1, XL_prev + 1)
        XM_cur = torch.where(take_open_all, MM_prev, XM_prev)

        # Y by left-to-right scan (kept simple for correctness)
        for j in range(1, m_max + 1):
            open_from_M = M_cur[:, :, :, j - 1] - gop
            extend_from_Y = Y_cur[:, :, :, j - 1] - gep
            take_open = open_from_M >= extend_from_Y
            Y_cur[:, :, :, j] = torch.where(take_open, open_from_M, extend_from_Y)
            YL_cur[:, :, :, j] = torch.where(take_open, ML_cur[:, :, :, j - 1] + 1, YL_cur[:, :, :, j - 1] + 1)
            YM_cur[:, :, :, j] = torch.where(take_open, MM_cur[:, :, :, j - 1], YM_cur[:, :, :, j - 1])

        # Mask Y for invalid i (when A exhausted, cannot insert gaps in B)
        Y_cur = torch.where(valid_i, Y_cur, torch.full_like(Y_cur, NEG))
        YL_cur = torch.where(valid_i, YL_cur, torch.zeros_like(YL_cur))
        YM_cur = torch.where(valid_i, YM_cur, torch.zeros_like(YM_cur))

        # next row
        M_prev, X_prev, Y_prev = M_cur, X_cur, Y_cur
        ML_prev, XL_prev, YL_prev = ML_cur, XL_cur, YL_cur
        MM_prev, XM_prev, YM_prev = MM_cur, XM_cur, YM_cur

    # Gather ends at b-length per pair
    idx = b_lengths.view(1, B, 1, 1).to(torch.int64)
    M_end = torch.gather(M_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)
    X_end = torch.gather(X_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)
    Y_end = torch.gather(Y_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)
    ML_end = torch.gather(ML_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)
    XL_end = torch.gather(XL_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)
    YL_end = torch.gather(YL_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)
    MM_end = torch.gather(MM_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)
    XM_end = torch.gather(XM_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)
    YM_end = torch.gather(YM_prev, 3, idx.repeat(A, 1, 12, 1)).squeeze(-1)

    end_scores = torch.stack([M_end, X_end, Y_end], dim=0)  # (3, A, B, T)
    best_state = torch.argmax(end_scores, dim=0)            # (A, B, T)
    end_len = torch.where(best_state == 0, ML_end, torch.where(best_state == 1, XL_end, YL_end))
    end_match = torch.where(best_state == 0, MM_end, torch.where(best_state == 1, XM_end, YM_end))
    pmi_abt = end_match.clamp(min=0).to(torch.float32) / end_len.clamp(min=1).to(torch.float32)
    pmi_ab = pmi_abt.max(dim=2).values  # (A, B)
    return pmi_ab.detach().to('cpu')


def compute_pmi_matrix(train_tokens, test_tokens, target_instr=66, gop=12, gep=6, device=None, batch_test=32, batch_train=8):
    train_compounds = [events_to_compound_matrix(t) for t in train_tokens]
    test_compounds = [events_to_compound_matrix(t) for t in test_tokens]

    train_melodies = [extract_melody_pitch_sequence(c, target_instr) for c in train_compounds]
    test_melodies = [extract_melody_pitch_sequence(c, target_instr) for c in test_compounds]

    N_train, N_test = len(train_melodies), len(test_melodies)
    pmi = torch.zeros(N_train, N_test, dtype=torch.float32)

    use_gpu = device is not None and device.startswith('cuda') and torch.cuda.is_available()

    if use_gpu:
        for s in tqdm(range(0, N_train, batch_train), desc='PMI train rows'):
            e = min(s + batch_train, N_train)
            a_batch = [train_melodies[k] for k in range(s, e) if train_melodies[k].size > 0]
            idx_map = [k for k in range(s, e) if train_melodies[k].size > 0]
            if len(a_batch) == 0:
                continue
            # process test in sub-batches to control memory
            for start in range(0, N_test, batch_test):
                end = min(start + batch_test, N_test)
                b_batch = test_melodies[start:end]
                pmi_block = _best_pmi_train_batch_gpu(a_batch, b_batch, gop, gep, device)
                # write back
                for ii, ridx in enumerate(idx_map):
                    pmi[ridx, start:end] = pmi_block[ii]
    else:
        for i in tqdm(range(N_train), desc='PMI train rows'):
            a = train_melodies[i]
            if a.size == 0:
                continue
            for j in range(N_test):
                b = test_melodies[j]
                if b.size == 0:
                    pmi[i, j] = 0.0
                    continue
                best_pmi = 0.0
                for t in range(12):
                    b_shift = (b + t) % 12
                    matches, alen = needleman_wunsch_affine(a, b_shift, gop=gop, gep=gep)
                    cur = float(matches) / float(max(alen, 1))
                    if cur > best_pmi:
                        best_pmi = cur
                pmi[i, j] = best_pmi
    return pmi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_file', type=str, default='/home/xiruij/anticipation/datasets/finetune/train_v2.txt')
    ap.add_argument('--test_file', type=str, default='/home/xiruij/anticipation/datasets/finetune/test_v2.txt')
    ap.add_argument('--out_path', type=str, default='/home/xiruij/anticipation/checkpoints_subset_large/melody_similarity_pmi.pt')
    ap.add_argument('--target_instr', type=int, default=66)  # Tenor Sax per user
    ap.add_argument('--max_test', type=int, default=500)
    ap.add_argument('--gop', type=int, default=12)
    ap.add_argument('--gep', type=int, default=6)
    ap.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--batch_test', type=int, default=64)
    ap.add_argument('--batch_train', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    train_lines = read_token_lines(args.train_file)
    test_lines = read_token_lines(args.test_file)[:args.max_test]

    pmi = compute_pmi_matrix(train_lines, test_lines, target_instr=args.target_instr, gop=args.gop, gep=args.gep, device=args.device, batch_test=args.batch_test, batch_train=args.batch_train)
    torch.save(pmi, args.out_path)
    print(f'Saved PMI matrix to {args.out_path} with shape {tuple(pmi.shape)}')


if __name__ == '__main__':
    main()
