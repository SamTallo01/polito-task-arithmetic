import json
import os

from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import NonLinearTaskVector

def function(args):

    eval_datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    task_vectors = []

    for dataset in eval_datasets:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

    task_vector = sum(task_vectors)

    args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
    val_metrics = {}
    # We use the validation set to choose the optimal coefficient.
    if args.opt_coeff is not None:

        optimal_coef = args.opt_coeff
        val_metrics[optimal_coef] = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        float(optimal_coef),
    )
    else:
        val_metrics = evaluate_task_vector(
            task_vector,
            pretrained_checkpoint,
            args,
        )

    if args.opt_coeff is None:
        optimal_coef = find_optimal_coef(
            val_metrics,
            metric="avg_normalized_top1",
            minimize=False,
        )

    print("=&" * 50)
    print(f"Optimal coefficient: {optimal_coef}")
    print("=&" * 50)
    print()
    print()


    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset for dataset in eval_datasets]
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        float(optimal_coef),
    )

    print("\n\nTest datasets metrics:")
    for dataset in eval_datasets:
        print("=" * 100)
        print(f"Test absolute accuracy for {dataset}: {test_metrics[dataset + ':top1']}")
        print(f"Test normalized accuracy for {dataset}: {test_metrics[dataset + ':normalized_top1']}")

    print("=" * 100)
    print()
    print()
    print("=" * 100)
    print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
    print(f"Test absolute accuracy: {test_metrics['avg_top1']}")

    print("=" * 100)
    print()
    train_eval_datasets = [dataset + "Val" for dataset in eval_datasets]

    print("\n\nTraining datasets metrics:")
    for key, metrics in val_metrics.items():
        print(f"Metrics for key {key}:")
        for dataset in train_eval_datasets:
            if f"{dataset}:top1" in metrics and f"{dataset}:normalized_top1" in metrics:
                print("=" * 100)
                print(f"Training absolute accuracy for {dataset}: {metrics[dataset + ':top1']}")
                print(f"Training normalized accuracy for {dataset}: {metrics[dataset + ':normalized_top1']}")

    print("=" * 100)
    print()
    print()
    for key, metrics in val_metrics.items():
        print(f"Metrics for key {key}:")
        print(f"Training normalized accuracy: {metrics.get('avg_normalized_top1', 'Not Available')}")
        print(f"Training absolute accuracy: {metrics.get('avg_top1', 'Not Available')}")
        print("=" * 100)
    
    return

if __name__ == '__main__':

    data_location = 'Task_Arithmetic_Datasets'
    model = 'ViT-B-32-quickgelu'

    args = parse_arguments()
    args.lr = 1e-4

    args.data_location = data_location
    args.model = model
    args.save = f'checkpoints'                        
    args.opt_coeff = 0.3

    args.finetuning_accuracies = {
        'DTDVal': 0.9876,
        'DTD': 0.9723,
        'EuroSATVal': 0.9975,
        'EuroSAT': 0.9815,  
        'GTSRBVal': 0.9999,
        'GTSRB': 0.9754,
        'MNISTVal': 0.9975,
        'MNIST': 0.9944,
        'RESISC45Val': 0.9985,
        'RESISC45': 0.9382,
        'SVHNVal': 0.9695,
        'SVHN': 0.9633,
    }
    
    function(args)


# for single_dataset in eval_datasets:
#         print("\n" + "=" * 100)
#         print(f"Evaluating {single_dataset} dataset with optimal coefficient (α = {optimal_coef:.2f}).")
#         print("=" * 100)

#         single_task_vector = NonLinearTaskVector(
#             f"{args.save}/{single_dataset}Val/zeroshot.pt",
#             f"{args.save}/{single_dataset}Val/finetuned.pt"
#         )

#         image_encoder = single_task_vector.apply_to(pretrained_checkpoint, scaling_coef=optimal_coef)

#         for split in ["test", "train"]:
#             eval_dataset = single_dataset if split == "test" else f"{single_dataset}Val"
#             accuracy = eval_single_dataset(image_encoder, eval_dataset, args)["top1"]
#             print("-" * 50)  # Add separator for test/val results
#             print(f"{split.capitalize()} Accuracy for {eval_dataset} with α = {optimal_coef:.2f}: {accuracy:.4f}")
#             print("-" * 50)