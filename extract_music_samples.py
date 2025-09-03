import torch
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

def extract_influential_samples(
    score_matrix_path,
    similarity_matrix_path,
    train_audio_dir,
    test_audio_dir,
    output_dir,
    num_test_samples=30,
    ranks_to_extract=[1, 5, 10, 100, 1000]
):
    """
    为前num_test_samples个测试样本提取对应的高影响力训练样本
    
    Args:
        score_matrix_path: 影响力得分矩阵路径
        similarity_matrix_path: 相似度矩阵路径
        train_audio_dir: 训练音频文件目录
        test_audio_dir: 测试音频文件目录
        output_dir: 输出目录
        num_test_samples: 要提取的测试样本数量
        ranks_to_extract: 要提取的排名列表
    """
    print(f"加载得分矩阵: {score_matrix_path}")
    score_matrix = torch.load(score_matrix_path, map_location=torch.device('cpu'))
    
    print(f"加载相似度矩阵: {similarity_matrix_path}")
    similarity_matrix = torch.load(similarity_matrix_path, map_location=torch.device('cpu'))
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取音频文件列表
    train_files = sorted([f for f in os.listdir(train_audio_dir) if f.endswith('.wav')])
    test_files = sorted([f for f in os.listdir(test_audio_dir) if f.endswith('.wav')])
    
    # 限制测试样本数量
    num_test_samples = min(num_test_samples, len(test_files))
    test_files = test_files[:num_test_samples]
    
    # 准备元数据
    metadata = []
    
    print(f"开始处理前{num_test_samples}个测试样本...")
    
    # 为每个测试样本处理
    for test_idx, test_file in enumerate(test_files):
        print(f"处理测试样本 {test_idx+1}/{num_test_samples}: {test_file}")
        
        # 创建该测试样本的目录
        test_sample_dir = os.path.join(output_dir, f"test_sample_{test_idx+1}")
        os.makedirs(test_sample_dir, exist_ok=True)
        
        # 复制测试样本到输出目录
        test_file_path = os.path.join(test_audio_dir, test_file)
        shutil.copy2(test_file_path, os.path.join(test_sample_dir, "test_sample.wav"))
        
        # 获取该测试样本的得分列
        score_col = score_matrix[:, test_idx]
        
        # 获取相似度列
        similarity_col = similarity_matrix[:, test_idx]
        
        # 按得分排序（降序）
        sorted_indices = torch.argsort(score_col, descending=True)
        
        # 准备该测试样本的元数据记录
        sample_record = {
            'test_sample_id': test_idx + 1,
            'test_sample_filename': test_file,
            'test_sample_path': os.path.join(test_sample_dir, "test_sample.wav")
        }
        
        # 为每个需要的排名提取训练样本
        for rank in ranks_to_extract:
            if rank - 1 < len(sorted_indices):
                train_idx = sorted_indices[rank - 1].item()
                train_file = train_files[train_idx]
                
                # 获取相似度得分
                similarity_score = similarity_col[train_idx].item()
                
                # 复制训练样本到输出目录
                train_file_path = os.path.join(train_audio_dir, train_file)
                output_train_path = os.path.join(test_sample_dir, f"rank_{rank}.wav")
                shutil.copy2(train_file_path, output_train_path)
                
                # 添加到记录
                sample_record[f'rank_{rank}_train_filename'] = train_file
                sample_record[f'rank_{rank}_train_path'] = output_train_path
                sample_record[f'rank_{rank}_similarity_score'] = similarity_score
        
        metadata.append(sample_record)
    
    # 创建并保存元数据CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)
    
    print(f"\n处理完成!")
    print(f"提取了{num_test_samples}个测试样本及其对应的影响力训练样本")
    print(f"元数据已保存至: {metadata_csv_path}")
    print(f"所有音频文件已保存至: {output_dir}")
    
    return metadata_df

def main():
    # 定义路径
    score_matrix_path = "/home/xiruij/anticipation/checkpoints_clap_new/score_LoGra.pt"
    similarity_matrix_path = "/home/xiruij/anticipation/checkpoints_clap_new/audio_similarity_matrix.pt"
    train_audio_dir = "/home/xiruij/anticipation/datasets/finetune_subset/song_train_wav"
    test_audio_dir = "/home/xiruij/anticipation/datasets/finetune_subset/song_test_wav"
    output_dir = "/home/xiruij/anticipation/extracted_music_samples"
    
    # 执行提取
    extract_influential_samples(
        score_matrix_path,
        similarity_matrix_path,
        train_audio_dir,
        test_audio_dir,
        output_dir,
        num_test_samples=30,
        ranks_to_extract=[1, 10, 100, 1000, 3000]
    )

if __name__ == "__main__":
    main()
