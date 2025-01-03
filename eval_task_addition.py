import json
import os

from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from eval_single_task import evaluate
from task_vectors import NonLinearTaskVector




def compute_metrics_after_scaling(pretrained_path, fine_tuned_paths, datasets, args):
    """
    Compute metrics after scaling for each task.
    """
    
    metrics_after_scaling = {}
    for i, dataset in enumerate(datasets):
        print(f"Evaluating metrics after scaling for dataset: {dataset}")

        print(f"Pretrained path: {pretrained_path}")
        print(f"Fine-tuned path: {fine_tuned_paths}")
        
        for path in fine_tuned_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")
            
        # Load task vector
        task_vector = NonLinearTaskVector(pretrained_path, fine_tuned_paths[i])
        print(f"Created task vector for {dataset} using checkpoint: {fine_tuned_paths[i]}")
        
        # Find the best scaling coefficient
        scaling_info = evaluate_task_vector(task_vector, pretrained_path, args)
        best_alpha = max(scaling_info, key=lambda a: scaling_info[a]["avg_normalized_top1"])

        # Apply best scaling coefficient
        scaled_model = task_vector.apply_to(pretrained_path, scaling_coef=best_alpha)

        # Evaluate metrics
        args.eval_datasets = [dataset]  # Evaluate only for this task
        test_metrics = evaluate(scaled_model, args)
        args.split = True
        train_metrics = evaluate(scaled_model, args)

        # Compute log-trace of Fisher Information Matrix
        # log_trace_fim = compute_log_trace_fim(scaled_model, args)

        # Save results
        metrics_after_scaling[dataset] = {
            "test_accuracy": test_metrics[dataset + ":top1"],
            "train_accuracy": train_metrics[dataset + ":top1"]
            # "log_trace_fim": log_trace_fim
        }

        print(f"Dataset: {dataset}, Test Acc: {test_metrics[dataset + ':top1']:.2f}, "
              f"Train Acc: {train_metrics[dataset + ':top1']:.2f}")

    return metrics_after_scaling

def evaluate_task_addition(pretrained_path, task_vectors, best_scalings, args):
    """
    Evaluate multi-task model using combined task vectors with optimal scaling coefficients.
    """ 
    # Combine scaled task vectors
    print(f"Combining task vectors for multi-task evaluation.")

    # Initialize combined_task_vector as None
    combined_task_vector = None

    # Loop through datasets and scale each task vector
    for i, dataset in enumerate(args.eval_datasets):
        print(f"Scaling task vector for dataset: {dataset}")
        scaled_vector = task_vectors[i] * best_scalings[dataset]
        print(f"Scaled vector for {dataset}: {scaled_vector}")

    # Accumulate the scaled task vectors
    if combined_task_vector is None:
        combined_task_vector = scaled_vector
    else:
        combined_task_vector += scaled_vector
    multi_task_model = combined_task_vector.apply_to(pretrained_path, scaling_coef=1.0)
    print("Multi-task model successfully created.")
    
    # Evaluate the multi-task model on all datasets
    results = {}
    total_absolute_accuracy = 0.0
    total_normalized_accuracy = 0.0

    for dataset in args.eval_datasets:
        args.eval_datasets = [dataset]
        metrics = evaluate(multi_task_model, args)

        # Compute individual metrics
        absolute_accuracy = metrics[dataset + ":top1"]
        normalized_accuracy = absolute_accuracy / args.finetuning_accuracies[dataset]

        results[dataset] = {
            "absolute_accuracy": absolute_accuracy,
            "normalized_accuracy": normalized_accuracy,
        }

        total_absolute_accuracy += absolute_accuracy
        total_normalized_accuracy += normalized_accuracy

    # Compute averages across all tasks
    num_tasks = len(args.eval_datasets)
    results["avg_absolute_accuracy"] = total_absolute_accuracy / num_tasks
    results["avg_normalized_accuracy"] = total_normalized_accuracy / num_tasks

    return results

def main():
    data_location = 'Task_Arithmetic_Datasets'
    model = 'ViT-B-32-quickgelu'
    args = parse_arguments()
    args.data_location = data_location
    args.lr = 1e-4
    args.batch_size = 32
    args.model = model
    args.control_dataset = None
    args.save = f'checkpoints'
    
    args.finetuning_accuracies = {
            'DTDVal': 0.9876,
            'DTD': 0.9723,
        }

    # Datasets and corresponding checkpoints
    eval_datasets = ["DTD"]
    
    args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
    
    fine_tuned_path = []
    task_vectors = []
    for dataset in eval_datasets:
        pretrained_checkpoint = f"checkpoints/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"checkpoints/{dataset}Val/finetuned.pt"
        
        # Store the fine-tuned checkpoint path
        fine_tuned_path.append(finetuned_checkpoint)

        # Create the task vector for this dataset
        task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        task_vectors.append(task_vector)
    

    # Step 1: Compute metrics after scaling
    metrics_after_scaling = compute_metrics_after_scaling(pretrained_checkpoint, fine_tuned_path, eval_datasets, args)
    
    print("\nMetrics After Scaling:")
    for dataset, metrics in metrics_after_scaling.items():
        print(f"{dataset}: {metrics}")


  
    # Step 2: Find optimal coefficients on validation set
    best_scalings = {}
    
    for i, dataset in enumerate(eval_datasets):
        task_vector = task_vectors[i]
        val_metrics = evaluate_task_vector(task_vector, pretrained_checkpoint, args)

        # Find optimal scaling coefficient
        optimal_alpha = max(val_metrics, key=lambda a: val_metrics[a]["avg_normalized_top1"])
        best_scalings[dataset] = optimal_alpha
        print(f"Best scaling for {dataset}: alpha={optimal_alpha}")

    # Step 3: Evaluate multi-task model on test set
    args.eval_datasets = [dataset for dataset in eval_datasets]  # Use test datasets
    
    test_results = evaluate_task_addition(
            pretrained_checkpoint, 
            task_vectors, 
            best_scalings, 
            args
    )
    print("=" * 100)
    # Print multi-task results
    print("\nMulti-task results:")
    for dataset, metrics in test_results.items():
        if isinstance(metrics, dict):
            print(f"{dataset}: Absolute Accuracy = {metrics['absolute_accuracy']:.4f}, "
                  f"Normalized Accuracy = {metrics['normalized_accuracy']:.4f}")
    print(f"Avg Absolute Accuracy: {multi_task_results['avg_absolute_accuracy']:.4f}")
    print(f"Avg Normalized Accuracy: {multi_task_results['avg_normalized_accuracy']:.4f}")


if __name__ == "__main__":
    main()