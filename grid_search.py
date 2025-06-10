import torch
from transformers import AutoModelForCausalLM, default_data_collator
import numpy as np
from torch.utils.data import DataLoader
import os
import argparse
import random
import time
import itertools
from dattri.algorithm.trak import TRAKAttributor
from dattri.task import AttributionTask
from score import TextDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for TRAK attribution hyperparameters.')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training data.')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Path to validation data.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing checkpoints and for saving results.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for dataloaders.')
    parser.add_argument('--num_checkpoints', type=int, default=3,
                        help='Number of checkpoints to use.')
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--reg_values', type=str, default="1e-1, 1, 10, 100, 1000",
                        help='Comma-separated list of regularization values to try.')
    parser.add_argument('--proj_dims', type=str, default="8192",
                        help='Comma-separated list of projection dimensions to try.')
    parser.add_argument('--eval_samples', type=int, default=500,
                        help='Number of evaluation samples to use.')
    parser.add_argument('--task_id', type=int, default=None,
                        help='Task ID for Slurm array jobs. If provided, only run the corresponding hyperparameter combination.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = TextDataset(args.train_file)
    eval_dataset = TextDataset(args.valid_file, num_samples=args.eval_samples)
    
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        print(f"Error: Dataset is empty or failed to load. Exiting.")
        return
    
    print(f"Training dataset length: {len(train_dataset)}.")
    print(f"Evaluation dataset length: {len(eval_dataset)}.")
    
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Check if output directory exists
    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist. Exiting.")
        return
    
    # Get checkpoints
    checkpoints = [os.path.join(args.output_dir, str(i)) for i in range(args.num_checkpoints)]
    
    if not os.path.isdir(checkpoints[0]):
        print(f"Error: Checkpoint directory {checkpoints[0]} does not exist. Exiting.")
        return
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained('/home/xiruij/anticipation/finetune_subset_output/final_model', attn_implementation="eager").to(device)
    model.eval()
    
    # Define loss and probability functions
    def f(params, batch):
        outputs = torch.func.functional_call(model, params, batch["input_ids"].to(device),
                                             kwargs={"attention_mask": batch["attention_mask"].to(device),
                                                     "labels": batch["labels"].to(device)})
        logp = -outputs.loss
        return logp - torch.log(1 - torch.exp(logp))

    def m(params, batch):
        outputs = torch.func.functional_call(model, params, batch["input_ids"].to(device),
                                             kwargs={"attention_mask": batch["attention_mask"].to(device),
                                                     "labels": batch["labels"].to(device)})
        p = torch.exp(-outputs.loss)
        return p
    
    # Define checkpoint loading function
    def checkpoints_load_func(model, checkpoint):
        model = AutoModelForCausalLM.from_pretrained(checkpoint, attn_implementation="eager").to(device)
        model.eval()
        return model
    
    # Create grid search parameter space
    reg_values = [float(x) for x in args.reg_values.split(',')]
    proj_dims = [int(x) for x in args.proj_dims.split(',')]
    
    print(f"Available regularization values: {reg_values}")
    print(f"Available projection dimensions: {proj_dims}")
    
    # Create task
    task = AttributionTask(loss_func=f, model=model,
                           checkpoints=checkpoints,
                           checkpoints_load_func=checkpoints_load_func)
    
    # Generate all parameter combinations
    all_combinations = list(itertools.product(reg_values, proj_dims))
    print(f"Total of {len(all_combinations)} parameter combinations")
    
    # Select specific combination to run based on task_id
    if args.task_id is not None:
        task_id = args.task_id
        if task_id < 0 or task_id >= len(all_combinations):
            print(f"Warning: task_id {task_id} out of range [0, {len(all_combinations) - 1}], using modulo operation")
            task_id = task_id % len(all_combinations)
        
        combinations_to_run = [all_combinations[task_id]]
        print(f"Running only task {task_id}: {combinations_to_run[0]}")
    else:
        combinations_to_run = all_combinations
        print("Running all parameter combinations")
    
    grid_search_dir = os.path.join(args.output_dir, "grid_search")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Execute only selected parameter combinations
    for reg, proj_dim in combinations_to_run:
        print(f"\nEvaluating configuration: reg={reg}, proj_dim={proj_dim}")
        
        projector_kwargs = {
            "device": device,
            "proj_dim": proj_dim,
            "use_half_precision": False,
        }
        
        start_time = time.time()
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=device,
            projector_kwargs=projector_kwargs,
            regularization=reg,
        )
        
        print("Caching training dataloader...")
        attributor.cache(train_dataloader)
        
        print("Computing attribution scores...")
        with torch.no_grad():
            score = attributor.attribute(eval_dataloader)
        
        elapsed_time = time.time() - start_time
        
        # Save results for this configuration
        config_name = f"reg_{reg}_dim_{proj_dim}"
        if args.task_id is not None:
            config_name = f"task_{args.task_id}_{config_name}"
            
        output_score_file = os.path.join(grid_search_dir, f"{config_name}_score.pt")
        torch.save(score, output_score_file)
        
        print(f"Results saved to {output_score_file}")
        print(f"Score shape: {score.shape}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
    
    print("\nGrid search completed.")
    
    # Print summary of configurations run
    print("\nGrid search summary:")
    for reg, proj_dim in combinations_to_run:
        config_name = f"reg_{reg}_dim_{proj_dim}"
        if args.task_id is not None:
            config_name = f"task_{args.task_id}_{config_name}"
        print(f"Configuration: {config_name}")

if __name__ == "__main__":
    main()