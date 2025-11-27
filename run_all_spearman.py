import torch
from spearman import calculate_one
import spearman

def calculate_with_custom_loss(path, loss_path):
    original_loss = "/home/xiruij/anticipation/checkpoints_subset_large/gt_gen_prompted.pt"
    
    spearman.calculate_one.__globals__['loss_list'] = torch.load(loss_path, map_location=torch.device('cpu')).detach()
    
    score = torch.load(path, map_location=torch.device('cpu'))
    
    nodes_str = [f"./checkpoints_subset_large/{i}/train_index.csv" for i in range(30)]
    full_nodes = [i for i in range(28000)]
    
    node_list = []
    for node_str in nodes_str:
        numbers = spearman.read_nodes(node_str)
        index = [full_nodes.index(number) for number in numbers]
        node_list.append(index)
    
    loss_list = torch.load(loss_path, map_location=torch.device('cpu')).detach()
    
    approx_output = []
    for i in range(len(nodes_str)):
        score_approx_0 = score[node_list[i], :]
        sum_0 = torch.sum(score_approx_0, axis=0)
        approx_output.append(sum_0)
    
    res = 0
    counter = 0
    from scipy.stats import spearmanr
    import numpy as np
    
    for i in range(500):
        tmp = spearmanr(np.array([approx_output[k][i] for k in range(len(approx_output))]),
                        np.array([loss_list[k][i].numpy() for k in range(len(loss_list))])).statistic
        if np.isnan(tmp):
            continue
        res += tmp
        counter += 1
    
    return res/counter

if __name__ == "__main__":
    base_path = "/home/xiruij/anticipation/checkpoints_subset_large"
    lds_path = f"{base_path}/lds_matrices"
    
    configs = [
        ("MERT", "Test", f"{lds_path}/lds_masked_audio_similarity_all_layers.pt", f"{base_path}/gt.pt"),
        ("MERT", "Generation", f"{lds_path}/lds_masked_audio_similarity_all_layers_gen_prompted.pt", f"{base_path}/gt_gen_prompted.pt"),
        ("CLAP", "Test", f"{base_path}/audio_similarity_clap.pt", f"{base_path}/gt.pt"),
        ("CLAP", "Generation", f"{base_path}/audio_similarity_clap_gen_prompted.pt", f"{base_path}/gt_gen_prompted.pt"),
        ("PMI", "Test", f"{base_path}/melody_similarity_pmi_ti.pt", f"{base_path}/gt.pt"),
        ("PMI", "Generation", f"{base_path}/melody_similarity_pmi_ti_gen_prompted.pt", f"{base_path}/gt_gen_prompted.pt"),
        ("TRAK", "Test", f"{base_path}/score_TRAK_8192_test.pt", f"{base_path}/gt.pt"),
        ("TRAK", "Generation", f"{base_path}/score_TRAK_8192_generated_prompted.pt", f"{base_path}/gt_gen_prompted.pt"),
        ("LoGra", "Test", f"{base_path}/score_LoGra_4096.pt", f"{base_path}/gt.pt"),
        ("LoGra", "Generation", f"{base_path}/score_LoGra_4096_gen_prompted.pt", f"{base_path}/gt_gen_prompted.pt"),
        ("Random", "Test", "random_test", f"{base_path}/gt.pt"),
        ("Random", "Generation", "random_gen", f"{base_path}/gt_gen_prompted.pt"),
    ]
    
    print(f"{'Model':<15} {'Type':<15} {'Spearman':>10}")
    print("-" * 45)
    
    for model, data_type, score_path, loss_path in configs:
        if score_path.startswith("random"):
            random_matrix = torch.rand(28000, 500)
            temp_path = "/tmp/random_matrix_temp.pt"
            torch.save(random_matrix, temp_path)
            result = calculate_with_custom_loss(temp_path, loss_path)
        else:
            result = calculate_with_custom_loss(score_path, loss_path)
        print(f"{model:<15} {data_type:<15} {result:>10.3f}")

