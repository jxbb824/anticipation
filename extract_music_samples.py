import torch
import os
import shutil
import pandas as pd
import numpy as np
import random
from pathlib import Path

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
    随机选择num_test_samples个测试样本，提取对应的高影响力训练样本
    
    Args:
        score_matrix_path: 影响力得分矩阵路径
        similarity_matrix_path: 相似度矩阵路径 (支持2D或4D张量)
                               - 4D: (layers, pooling_types, train_samples, test_samples)
                                 自动提取最后一层的mean和max pooling
                               - 2D: (train_samples, test_samples) 直接使用
                               用于记录每个排名训练样本与测试样本的实际相似度
        train_audio_dir: 训练音频文件目录
        test_audio_dir: 测试音频文件目录
        output_dir: 输出目录
        num_test_samples: 要提取的测试样本数量
        ranks_to_extract: 要提取的排名列表
        audio_extension: 音频文件扩展名 (如'.wav', '.mp3')，默认None表示自动检测
        random_seed: 随机种子，用于可重复的随机选择（默认42）
    """
    print(f"加载得分矩阵: {score_matrix_path}")
    score_matrix = torch.load(score_matrix_path, map_location=torch.device('cpu'))
    
    print(f"加载相似度矩阵: {similarity_matrix_path}")
    similarity_matrix = torch.load(similarity_matrix_path, map_location=torch.device('cpu'))
    
    # Handle multi-dimensional similarity matrices (shape: (layers, pooling_types, a, b))
    # Extract the last layer with both mean and max pooling
    similarity_mean = None
    similarity_max = None
    
    if similarity_matrix.dim() == 4:
        original_shape = similarity_matrix.shape
        print(f"检测到4D相似度矩阵，形状: {original_shape}")
        print(f"自动提取最后一层 (索引 -1) 的 mean pooling (索引 0) 和 max pooling (索引 1)...")
        similarity_mean = similarity_matrix[-1, 0, :, :]  # Last layer, mean pooling
        similarity_max = similarity_matrix[-1, 1, :, :]   # Last layer, max pooling
        print(f"提取后的相似度矩阵形状: mean={similarity_mean.shape}, max={similarity_max.shape}")
    elif similarity_matrix.dim() == 2:
        # 2D矩阵，直接使用
        print(f"检测到2D相似度矩阵，形状: {similarity_matrix.shape}")
        similarity_mean = similarity_matrix
        similarity_max = similarity_matrix
    else:
        raise ValueError(f"不支持的相似度矩阵维度: {similarity_matrix.dim()}D，期望2D或4D")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 自动检测音频文件扩展名（如果未指定）
    if audio_extension is None:
        all_files = os.listdir(train_audio_dir)
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            if any(f.endswith(ext) for f in all_files):
                audio_extension = ext
                print(f"自动检测到音频格式: {audio_extension}")
                break
        if audio_extension is None:
            raise ValueError("无法自动检测音频格式，请手动指定 audio_extension 参数")
    
    # 获取音频文件列表（排序保证一致性）
    train_files = sorted([f for f in os.listdir(train_audio_dir) if f.endswith(audio_extension)])
    all_test_files = sorted([f for f in os.listdir(test_audio_dir) if f.endswith(audio_extension)])
    
    # 设置随机种子并随机选择测试样本
    random.seed(random_seed)
    num_test_samples = min(num_test_samples, len(all_test_files))
    
    # 创建索引列表并随机打乱，然后选择前N个
    test_indices = list(range(len(all_test_files)))
    random.shuffle(test_indices)
    selected_test_indices = sorted(test_indices[:num_test_samples])  # 排序以便按索引顺序处理
    
    print(f"使用随机种子 {random_seed} 随机选择了 {num_test_samples} 个测试样本")
    print(f"选中的测试样本索引: {selected_test_indices[:10]}..." if len(selected_test_indices) > 10 else f"选中的测试样本索引: {selected_test_indices}")
    
    # 准备元数据
    metadata = []
    
    print(f"开始处理 {num_test_samples} 个随机选择的测试样本...")
    
    # 为每个选中的测试样本处理
    for sample_num, original_test_idx in enumerate(selected_test_indices, 1):
        test_file = all_test_files[original_test_idx]
        print(f"处理测试样本 {sample_num}/{num_test_samples}: {test_file} (原始索引: {original_test_idx})")
        
        # 创建该测试样本的目录
        test_sample_dir = os.path.join(output_dir, f"test_sample_{sample_num}")
        os.makedirs(test_sample_dir, exist_ok=True)
        
        # 复制测试样本到输出目录，保留原文件名
        test_file_path = os.path.join(test_audio_dir, test_file)
        test_output_path = os.path.join(test_sample_dir, test_file)
        shutil.copy2(test_file_path, test_output_path)
        
        # 获取该测试样本的得分列（使用原始索引）
        score_col = score_matrix[:, original_test_idx]
        
        # 获取相似度列（使用原始索引，同时获取mean和max）
        similarity_mean_col = similarity_mean[:, original_test_idx]
        similarity_max_col = similarity_max[:, original_test_idx]
        
        # 按得分排序（降序）
        sorted_indices = torch.argsort(score_col, descending=True)
        
        # 准备该测试样本的元数据记录
        sample_record = {
            'test_sample_id': sample_num,
            'test_sample_original_index': original_test_idx,
            'test_sample_filename': test_file,
            'test_sample_path': test_output_path
        }
        
        # 为每个需要的排名提取训练样本
        for rank in ranks_to_extract:
            if rank - 1 < len(sorted_indices):
                train_idx = sorted_indices[rank - 1].item()
                train_file = train_files[train_idx]
                
                # 获取相似度得分（mean和max）
                similarity_mean_score = similarity_mean_col[train_idx].item()
                similarity_max_score = similarity_max_col[train_idx].item()
                
                # 获取影响力得分
                influence_score = score_col[train_idx].item()
                
                # 复制训练样本到输出目录，保留原文件名但添加rank前缀
                train_file_path = os.path.join(train_audio_dir, train_file)
                train_file_basename = os.path.splitext(train_file)[0]
                train_file_ext = os.path.splitext(train_file)[1]
                output_train_filename = f"rank_{rank:05d}_{train_file_basename}{train_file_ext}"
                output_train_path = os.path.join(test_sample_dir, output_train_filename)
                shutil.copy2(train_file_path, output_train_path)
                
                # 添加到记录
                sample_record[f'rank_{rank}_train_index'] = train_idx
                sample_record[f'rank_{rank}_train_filename'] = train_file
                sample_record[f'rank_{rank}_train_path'] = output_train_path
                sample_record[f'rank_{rank}_influence_score'] = influence_score
                sample_record[f'rank_{rank}_similarity_mean'] = similarity_mean_score
                sample_record[f'rank_{rank}_similarity_max'] = similarity_max_score
        
        metadata.append(sample_record)
    
    # 创建并保存元数据CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)
    
    print(f"\n处理完成!")
    print(f"随机选择并提取了 {num_test_samples} 个测试样本及其对应的影响力训练样本")
    print(f"随机种子: {random_seed}")
    print(f"元数据已保存至: {metadata_csv_path}")
    print(f"所有音频文件已保存至: {output_dir}")
    print(f"\n说明:")
    print(f"- 测试样本保留原文件名")
    print(f"- 训练样本文件名格式: rank_XXXXX_原文件名{audio_extension}")
    print(f"- metadata.csv 包含:")
    print(f"  * influence_score: 影响力得分")
    print(f"  * similarity_mean: 相似度得分 (mean pooling)")
    print(f"  * similarity_max: 相似度得分 (max pooling)")
    
    return metadata_df

def main():
    # 定义路径
    # score_matrix_path = "/home/xiruij/anticipation/checkpoints_clap_new/score_LoGra.pt"
    # similarity_matrix_path = "/home/xiruij/anticipation/checkpoints_clap_new/audio_similarity_matrix.pt"
    # train_audio_dir = "/home/xiruij/anticipation/datasets/finetune_subset/song_train_wav"
    # test_audio_dir = "/home/xiruij/anticipation/datasets/finetune_subset/song_test_wav"
    # output_dir = "/home/xiruij/anticipation/extracted_music_samples"
    
    score_matrix_path = "/home/xiruij/anticipation/checkpoints_subset_large/score_LoGra_4096.pt"
    similarity_matrix_path = "/home/xiruij/anticipation/checkpoints_subset_large/audio_similarity_all_layers.pt"
    train_audio_dir = "/home/xiruij/anticipation/datasets/finetune/song_train_mp3"
    test_audio_dir = "/home/xiruij/anticipation/datasets/finetune/song_test_mp3"
    output_dir = "/home/xiruij/anticipation/extracted_music_samples_final_test"
    # 执行提取
    extract_influential_samples(
        score_matrix_path,
        similarity_matrix_path,
        train_audio_dir,
        test_audio_dir,
        output_dir,
        num_test_samples=12,
        ranks_to_extract=[1, 10, 100, 14000, 27900, 28000]
    )

if __name__ == "__main__":
    main()
