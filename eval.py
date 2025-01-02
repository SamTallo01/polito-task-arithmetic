import numpy as np
from modeling import ImageEncoder
from eval_single_task import evaluate

def evaluate_task_vector_at_coef(task_vector, pretrained_checkpoint, args, scaling_coef):
    image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    coef_info = evaluate(image_encoder, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean([coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets])
    coef_info["avg_top1"] = np.mean([coef_info[dataset + ":top1"] for dataset in args.eval_datasets])

    return coef_info


def evaluate_task_vector(task_vector, pretrained_checkpoint, args):
    info = {}
    for scaling_coef in np.linspace(0.0, 0.3, args.n_eval_points):
        print("=" * 100)
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        print("=" * 100)
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef,
        )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results


def nonlinear_advantage(acc_linear, acc_nonlinear, num_classes):
    err_linear = 1 - acc_linear
    err_nonlinear = 1 - acc_nonlinear
    return (err_linear - err_nonlinear) * num_classes / (num_classes - 1)







