import json
import os

from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import NonLinearTaskVector

def function(args):

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}"
    else:
        args.save = f"checkpoints"

    eval_datasets = ["DTD"]
    task_vectors = []

    for dataset in eval_datasets:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

    task_vector = sum(task_vectors)

    args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
    args.control_dataset = None

    # We use the validation set to choose the optimal coefficient.
    val_metrics = evaluate_task_vector(
        task_vector,
        pretrained_checkpoint,
        args,
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric="avg_normalized_top1",
        minimize=False,
    )

    print("=&" * 50)
    print(f"Optimal coefficient: {optimal_coef}")
    print("=&" * 50)

    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset for dataset in eval_datasets]
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        float(optimal_coef),
    )

    print("=" * 100)
    print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
    print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
    print("=" * 100)

if __name__ == '__main__':
    data_location = 'Task_Arithmetic_Datasets'
    model = 'ViT-B-32-quickgelu'
    datasets = ['DTD'] 
    epochs = {
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SVHN': 4,
    }

    for dataset in datasets:

        args = parse_arguments()
        args.lr = 1e-4

        args.epochs = epochs[dataset]
        args.data_location = data_location
        args.train_dataset = dataset + 'Val'
        args.batch_size = 32
        args.model = model

        args.save = f'checkpoints'  
        args.eval_datasets = dataset + 'Val'  
        args.finetuning_mode = f"standard"
        args.finetuning_accuracies = {
            'DTDVal': 0.9876,
            'DTD': 0.9723,
            #'EuroSATVal': 0.9975,
        }
        function(args)
