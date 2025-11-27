import os
import shutil
import argparse
import json
import pandas as pd


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_audio(src: str, dst: str):
    safe_mkdir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def load_pairs_dataframe(pairs_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pairs_csv)
    required_cols = [
        'subset', 'group', 'rank_requested', 'rank_resolved', 'eval_local_id',
        'eval_original_index', 'eval_filename', 'eval_path', 'train_index',
        'train_filename', 'train_path'
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in pairs CSV: {c}")
    return df


def build_question_indices(seed_order_csv: str, default_total: int = 40):
    # If a precomputed order is needed, we can derive from CSV; else 0..39
    if seed_order_csv and os.path.exists(seed_order_csv):
        order_df = pd.read_csv(seed_order_csv, header=None)
        order = order_df[0].tolist()
        if len(order) != default_total:
            raise ValueError("Seed order length mismatch")
        return order
    return list(range(default_total))


def prepare_ui_bundle(
    extracted_root: str,
    output_ui_dir: str,
    mapping_output_path: str,
    question_count: int = 40
):
    pairs_csv = os.path.join(extracted_root, 'pairs_metadata_all.csv')
    eval_csv = os.path.join(extracted_root, 'eval_samples_all.csv')
    if not os.path.exists(pairs_csv):
        raise FileNotFoundError(f"pairs_metadata_all.csv not found at {pairs_csv}")
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"eval_samples_all.csv not found at {eval_csv}")

    pairs_df = load_pairs_dataframe(pairs_csv)

    # Select eval samples; each eval yields TWO questions (TDA and SIMILARITY)
    # For 40 total questions, take 10 evals from 'test' and 10 from 'generation'.
    test_pairs = pairs_df[pairs_df['subset'] == 'test']
    gen_pairs = pairs_df[pairs_df['subset'] == 'generation']

    def pick_first_k_eval(eval_subset_df: pd.DataFrame, k: int):
        eval_ids = sorted(eval_subset_df['eval_local_id'].unique())[:k]
        return eval_ids

    # Each eval contributes 2 questions per subset overall
    evals_per_subset = max(1, question_count // 4)
    test_eval_ids = pick_first_k_eval(test_pairs, evals_per_subset)
    gen_eval_ids = pick_first_k_eval(gen_pairs, evals_per_subset)

    # Build mapping for q indices 0..39 to concrete files
    mapping = {
        'questions': []
    }

    # Create UI directory structure
    safe_mkdir(output_ui_dir)
    audio_root = os.path.join(output_ui_dir, 'audio')
    safe_mkdir(audio_root)

    # Copy static UI files if exist in repo
    repo_ui_dir = os.path.join(os.path.dirname(__file__), 'human_eval_ui')
    if not os.path.isdir(repo_ui_dir):
        raise FileNotFoundError(f"Missing human_eval_ui template at {repo_ui_dir}")

    for fname in ['index.html', 'main.js']:
        shutil.copy2(os.path.join(repo_ui_dir, fname), os.path.join(output_ui_dir, fname))
    # Copy consent.pdf if present alongside the template
    consent_src = os.path.join(repo_ui_dir, 'consent.pdf')
    if os.path.exists(consent_src):
        shutil.copy2(consent_src, os.path.join(output_ui_dir, 'consent.pdf'))

    # Helper to extract 6 option rows for a given subset, eval_local_id, and group
    def get_group_six_rows(subset_name: str, eval_id: int, group_name: str):
        subset_rows = pairs_df[
            (pairs_df['subset'] == subset_name)
            & (pairs_df['eval_local_id'] == eval_id)
            & (pairs_df['group'] == group_name)
        ]
        subset_rows = subset_rows.sort_values('rank_resolved')
        return subset_rows.head(6)

    q_index = 0
    for subset_name, eval_ids in [('test', test_eval_ids), ('generation', gen_eval_ids)]:
        for eval_id in eval_ids:
            # Two questions per eval: one TDA and one SIMILARITY
            for group_name in ['TDA', 'SIMILARITY']:
                rows = get_group_six_rows(subset_name, eval_id, group_name)
                if len(rows) < 6:
                    continue

                row0 = rows.iloc[0]
                test_src = row0['eval_path']
                q_dir = os.path.join(audio_root, f"q_{str(q_index).zfill(3)}")
                safe_mkdir(q_dir)
                test_dst = os.path.join(q_dir, 'test.mp3')
                copy_audio(test_src, test_dst)

                option_info = []
                for i in range(6):
                    r = rows.iloc[i]
                    opt_src = r['train_path']
                    opt_dst = os.path.join(q_dir, f"opt_{i}.mp3")
                    copy_audio(opt_src, opt_dst)
                    option_info.append({
                        'subset': r['subset'],
                        'group': r['group'],
                        'rank_requested': r['rank_requested'],
                        'rank_resolved': int(r['rank_resolved']),
                        'eval_local_id': int(r['eval_local_id']),
                        'eval_original_index': int(r['eval_original_index']),
                        'train_index': int(r['train_index'])
                    })

                mapping['questions'].append({
                    'q_index': q_index,
                    'subset': subset_name,
                    'group': group_name,
                    'eval_local_id': int(row0['eval_local_id']),
                    'eval_original_index': int(row0['eval_original_index']),
                    'test_path': os.path.relpath(test_dst, start=os.path.dirname(mapping_output_path)),
                    'options': option_info
                })
                q_index += 1
                if q_index >= question_count:
                    break
            if q_index >= question_count:
                break
        if q_index >= question_count:
            break

    # Save mapping outside UI folder (as requested)
    safe_mkdir(os.path.dirname(mapping_output_path))
    with open(mapping_output_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Prepared UI at: {output_ui_dir}")
    print(f"Mapping saved to: {mapping_output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare self-contained human evaluation UI folder')
    parser.add_argument('--extracted_root', required=True, help='Path to extracted pairs output root')
    parser.add_argument('--output_ui_dir', required=True, help='Destination folder for the self-contained UI')
    parser.add_argument('--mapping_out', required=True, help='Path to save mapping.json OUTSIDE the UI folder')
    parser.add_argument('--questions', type=int, default=40, help='Number of questions to include (default 40)')
    args = parser.parse_args()

    prepare_ui_bundle(
        extracted_root=args.extracted_root,
        output_ui_dir=args.output_ui_dir,
        mapping_output_path=args.mapping_out,
        question_count=args.questions
    )


if __name__ == '__main__':
    main()


