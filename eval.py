import numpy as np

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
    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print()
        print("=" * 100)
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        print("=" * 100)
        print()
        print()

        info[scaling_coef] = evaluate_task_vector_at_coef(task_vector,pretrained_checkpoint,args,scaling_coef,)

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:

        results[dataset_name + ":normalized_top1"] = (results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name])

    return results