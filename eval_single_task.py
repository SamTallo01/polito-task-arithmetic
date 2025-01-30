import tqdm
import torch
import utils
from datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from datasets.registry import get_dataset
from utils import train_diag_fim_logtr
from args import parse_arguments
import csv

def eval_single_dataset(image_encoder, dataset_name, args):

    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=args.split, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%")

    return metrics


def evaluate(image_encoder, args):

    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (args.eval_datasets)

    for dataset_name in eval_datasets:
        print("=" * 100)
        print("Evaluating on", dataset_name)
        print("=" * 100)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]
        if dataset_name.endswith('Val') and args.split:
            #Log-det of the Fisher Information Matrix
            classification_head = get_classification_head(args, dataset_name)
            model = ImageClassifier(image_encoder, classification_head)
            samples_nr = 200
            logdet_hF = train_diag_fim_logtr(args, model,  dataset_name , samples_nr)
            print(f"{'='*100}\nLog-det of the Fisher Information Matrix: {logdet_hF}\n{'='*100}")

    return per_dataset_results


def save_results_to_file(results, file_path, batch_size, learning_rate, weight_decay):

    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if the file is new
        if not file_exists:
            writer.writerow(["Dataset", "Split", "top1", "logdet_hF", "Batch_Size", "Learning_Rate", "Weight_Decay"])

        # Write rows
        for dataset, splits in results.items():
            for split, metrics in splits.items():
                writer.writerow([
                    dataset,
                    split,
                    metrics.get("top1", None),
                    metrics.get("logdet_hF", None),
                    batch_size,
                    learning_rate,
                    weight_decay
                ])

def eval_single_task(args):

    eval_datasets = args.eval_datasets
    samples_nr = 200  # How many per-example gradients to accumulate

    # Extracting learning rate and weight decay from args
    learning_rate = args.lr
    weight_decay = getattr(args, 'weight_decay', None)  # Assuming weight_decay exists in args

    #Initialize the result
    results = {}
    for dataset in eval_datasets:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        train_dataset = dataset + "Val"

        # Training set
        image_encoder = ImageEncoder.load(pretrained_checkpoint)
        print("=" * 100)
        print(f"Evaluating on training set of {dataset}.")
        print("=" * 100)
        args.split = True
        train_metrics = eval_single_dataset(image_encoder, train_dataset, args)

        # Log-det of the Fisher Information Matrix
        classification_head = get_classification_head(args, train_dataset)
        model = ImageClassifier(image_encoder, classification_head)
        logdet_hF = train_diag_fim_logtr(args, model, train_dataset, samples_nr)
        print(f"{'='*100}\nLog-det of the Fisher Information Matrix: {logdet_hF}\n{'='*100}")

        # Save training results
        if dataset not in results:
            results[dataset] = {}
        results[dataset]["train"] = {"top1": train_metrics["top1"], "logdet_hF": logdet_hF}

        # Test set
        args.split = False
        print("=" * 100)
        print(f"Evaluating on test set of {dataset}.")
        print("=" * 100)

        test_metrics = eval_single_dataset(image_encoder, dataset, args)

        # Save test results
        results[dataset]["test"] = {"top1": test_metrics["top1"]}

    for dataset in eval_datasets:
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        train_dataset = dataset + "Val"

        # Training set
        image_encoder = ImageEncoder.load(finetuned_checkpoint)
        print("=" * 100)
        print(f"Evaluating on training set of finetuned {dataset}.")
        print("=" * 100)
        args.split = True
        train_metrics = eval_single_dataset(image_encoder, train_dataset, args)

        # Log-det of the Fisher Information Matrix
        classification_head = get_classification_head(args, train_dataset)
        model = ImageClassifier(image_encoder, classification_head)
        logdet_hF = train_diag_fim_logtr(args, model, train_dataset, samples_nr)
        print(f"{'='*100}\nLog-det of the Fisher Information Matrix: {logdet_hF}\n{'='*100}")

        # Save finetuned training results
        if dataset not in results:
            results[dataset] = {}
            
        results[dataset]["finetuned_train"] = {"top1": train_metrics["top1"], "logdet_hF": logdet_hF}

        # Test set
        args.split = False
        print("=" * 100)
        print(f"Evaluating on test set of finetuned {dataset}.")
        print("=" * 100)
        test_metrics = eval_single_dataset(image_encoder, dataset, args)

        # Save finetuned test results
        results[dataset]["finetuned_test"] = {"top1": test_metrics["top1"]}

        # Save results to file immediately after processing the dataset
        save_results_to_file(results, args.results_db, args.batch_size, learning_rate, weight_decay)


def eval_single_checkpoint(args):

    if not args.load:
        print("No checkpoint specified. Please provide a checkpoint via 'args.load'.")
        return

    checkpoint_path = args.load
    dataset = args.eval_datasets[0]  # Prendiamo il primo dataset per la valutazione, dato che il checkpoint è specifico
    train_dataset = dataset + "Val"

    # Load the image encoder from the checkpoint
    image_encoder = ImageEncoder.load(checkpoint_path)
    print("=" * 100)
    print(f"Evaluating on dataset {dataset} using checkpoint {checkpoint_path}.")
    print("=" * 100)

    # Evaluation on the training set
    args.split = True
    train_metrics = eval_single_dataset(image_encoder, train_dataset, args)

    # Log-det of the Fisher Information Matrix
    samples_nr = 200
    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    logdet_hF = train_diag_fim_logtr(args, model, train_dataset, samples_nr)
    print(f"{'='*100}\nLog-det of the Fisher Information Matrix: {logdet_hF}\n{'='*100}")

    # Save the training set results
    results = {}
    results[dataset] = {}
    results[dataset]["train"] = {"top1": train_metrics["top1"], "logdet_hF": logdet_hF}

    # Run the evaluation for the test set
    args.split = False
    print("=" * 100)
    print(f"Evaluating on test set of {dataset}.")
    print("=" * 100)
    test_metrics = eval_single_dataset(image_encoder, dataset, args)

    # Save the test set results
    results[dataset]["test"] = {"top1": test_metrics["top1"]}

    # Save the results to file
    save_results_to_file(results, args.results_db, args.batch_size, args.lr, getattr(args, 'weight_decay', None))


if __name__ == '__main__':
    import os
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

    args = parse_arguments()
    args.lr = 1e-4                                  # Is also saved in the result file if specified
    args.epochs = epochs
    args.data_location = data_location
    args.batch_size = 32                            # Is also saved in the result file if specified
    args.model = model
    args.weight_decay = 0.0                         # Is also saved in the result file if specified

    args.save = f'checkpoints'
    args.eval_datasets = datasets
    args.results_db = f'results.csv'                # Results saved to this CSV file

    #Example of how to load a checkpoint
    # args.load = f'checkpoints/DTDVal/finetuned.pt'

    # If a checkpoint is specified, evaluate it otherwise evaluate all datasets finetuned and zeroshot 
    # that are in the folder specified by args.save
    if args.load:
        eval_single_checkpoint(args)
    else:
        eval_single_task(args)