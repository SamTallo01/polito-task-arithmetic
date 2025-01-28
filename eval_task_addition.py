import json
import os

from utils import find_optimal_coef, train_diag_fim_logtr
from modeling import ImageClassifier, ImageEncoder
from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import NonLinearTaskVector
from eval_single_task import eval_single_dataset
from heads import get_classification_head


def function(args):
    # Define datasets to evaluate
    eval_datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    # List to store the task vectors for each dataset
    task_vectors = []               

    # Create task vector for each dataset
    for dataset in eval_datasets:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

    # Sum all task vecotrs to create a multi-task vector
    task_vector = sum(task_vectors)

    # We use the validation set to choose the optimal coefficient.
    args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
    val_metrics = {}
    
    # Check if an optimal coefficient is provided
    if args.opt_coeff is not None:
        optimal_coef = args.opt_coeff
        val_metrics[optimal_coef] = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        float(optimal_coef),
    )
    else:
        # Evaluate task vector for various coefficients to find the best one
        val_metrics = evaluate_task_vector(
            task_vector,
            pretrained_checkpoint,
            args,
        )

    # Find the optimal coefficient if not provided
    if args.opt_coeff is None:
        optimal_coef = find_optimal_coef(
            val_metrics,
            metric="avg_normalized_top1",
            minimize=False,
        )

    results_file = "task_addition_results.txt"

    def save_to_file(content, file_path):
        with open(file_path, "a") as f:
            f.write(content + "\n")

    # Display and save the optimal coefficient
    separator = "==" * 50
    save_to_file("\n"+separator, results_file)
    save_to_file(f"Optimal coefficient: {optimal_coef}", results_file)
    save_to_file(separator, results_file)
    save_to_file("\n\n", results_file)

    # Evaluate on the test set with the optimal coefficient.
    args.split = False
    args.eval_datasets = [dataset for dataset in eval_datasets]
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        float(optimal_coef),
    )

    # Test datasets metrics
    save_to_file("=" * 100, results_file)
    save_to_file("Test datasets metrics:", results_file)
    save_to_file("=" * 100, results_file)
    for dataset in eval_datasets:
        save_to_file(f"Test absolute accuracy for {dataset}: {test_metrics[dataset + ':top1']}", results_file)
        save_to_file(f"Test normalized accuracy for {dataset}: {test_metrics[dataset + ':normalized_top1']}", results_file)
        save_to_file("-" * 100, results_file)

    save_to_file("=" * 100, results_file)
    save_to_file("", results_file)
    save_to_file("=" * 100, results_file)
    save_to_file(f"Test average absolute accuracy: {test_metrics['avg_top1']}", results_file)
    save_to_file(f"Test average normalized accuracy: {test_metrics['avg_normalized_top1']}", results_file)
    save_to_file("=" * 100, results_file)
    save_to_file("\n\n", results_file)

    # Training dataset metrics
    train_eval_datasets = [dataset + "Val" for dataset in eval_datasets]
    save_to_file("=" * 100, results_file)
    save_to_file("Training datasets metrics:", results_file)
    save_to_file("=" * 100, results_file)

    for key, metrics in val_metrics.items():
        save_to_file(f"Metrics for key {key}:", results_file)
        save_to_file("=" * 100, results_file)
        for dataset in train_eval_datasets:
            if f"{dataset}:top1" in metrics and f"{dataset}:normalized_top1" in metrics:
                save_to_file(f"Training absolute accuracy for {dataset}: {metrics[dataset + ':top1']}", results_file)
                save_to_file(f"Training normalized accuracy for {dataset}: {metrics[dataset + ':normalized_top1']}", results_file)
                save_to_file("-" * 100, results_file)

    save_to_file("\n\n", results_file)
    for key, metrics in val_metrics.items():
        save_to_file("=" * 100, results_file)
        save_to_file(f"Average metrics for key {key}:", results_file)
        save_to_file("=" * 100, results_file)
        save_to_file(f"Training average absolute accuracy: {metrics.get('avg_top1', 'Not Available')}", results_file)
        save_to_file(f"Training average normalized accuracy: {metrics.get('avg_normalized_top1', 'Not Available')}", results_file)
        save_to_file("-" * 100, results_file)


    # Evaluating a single task with the optimal coefficient
    for single_dataset in eval_datasets:
        single_dataset_header = "\n\n" + "=" * 100 + f"\nEvaluating {single_dataset} dataset with optimal coefficient (alpha = {optimal_coef:.2f}).\n" + "=" * 100
        save_to_file(single_dataset_header, results_file)
        
        pretrained_checkpoint = f"{args.save}/{single_dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{single_dataset}Val/finetuned.pt"
        
        single_task_vector = NonLinearTaskVector(
            pretrained_checkpoint,
            finetuned_checkpoint
        )
        # Apply the optimal coefficient to the task vector
        image_encoder = single_task_vector.apply_to(pretrained_checkpoint, scaling_coef=optimal_coef)

        samples_nr = 200
        classification_head = get_classification_head(args, single_dataset)
        model = ImageClassifier(image_encoder, classification_head)

        # Evaluate both test and train splits
        for split in ["test", "train"]:
            if split == "test":
                args.split = False  # test split
                eval_dataset = single_dataset
            else:
                args.split = True  # Use training split
                eval_dataset = f"{single_dataset}Val"
                logdet_hF = train_diag_fim_logtr(args, model, eval_dataset, samples_nr)

            # Evaluate and display accuracy
            accuracy = eval_single_dataset(image_encoder, eval_dataset, args)["top1"]
            accuracy_result = (f"{split.capitalize()} Accuracy for {eval_dataset} with alpha = {optimal_coef:.2f}: {accuracy:.4f}\n" + "-" * 100)
            save_to_file(accuracy_result, results_file)
            if split == "train":
                logdet_result = (f"Log-det of the Fisher Information Matrix for {eval_dataset} with alpha = {optimal_coef:.2f}: {logdet_hF:.4f}\n" + "-" * 100)
                save_to_file(logdet_result, results_file)


if __name__ == '__main__':

    data_location = 'Task_Arithmetic_Datasets'
    model = 'ViT-B-32-quickgelu'

    args = parse_arguments()
    args.lr = 1e-4

    args.data_location = data_location
    args.model = model
    args.save = f'checkpoints_batch_size_32'                        
    #args.opt_coeff = 0.3        # Optional optimal coefficient in order to skip the search

    # Define fine-tuning accuracies for datasets
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
