import numpy as np
from modeling import ImageEncoder
from eval_single_task import evaluate


def evaluate_task_vector_at_coef(task_vector, pretrained_checkpoint, args, scaling_coef):
    """
    Evaluate a task vector at a specific scaling coefficient.
    """
    # Apply scaling coefficient to the task vector
    image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    print("Image encoder successfully created.")
    metrics = evaluate(image_encoder, args)

    if metrics is None:
        raise ValueError("`evaluate` returned None. Check if the evaluation failed.")
    
    metrics = add_normalized_accuracy(metrics, args)
    # Compute average metrics
    avg_normalized_top1 = np.mean(
        [metrics[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    avg_top1 = np.mean(
        [metrics[dataset + ":top1"] for dataset in args.eval_datasets]
    )
    metrics["avg_normalized_top1"] = avg_normalized_top1
    metrics["avg_top1"] = avg_top1

    return metrics


def evaluate_task_vector(task_vector, pretrained_checkpoint, args):
    info = {}
    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print("-" * 100)
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        print("-" * 100)
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef
        )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results









