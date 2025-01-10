import tqdm
import torch
import utils
from datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from datasets.registry import get_dataset
from utils import train_diag_fim_logtr
from args import parse_arguments

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
            # Log-det of the Fisher Information Matrix
            classification_head = get_classification_head(args, dataset_name)
            model = ImageClassifier(image_encoder, classification_head)
            samples_nr = 200
            logdet_hF = train_diag_fim_logtr(args, model,  dataset_name , samples_nr)
            print(f"{'='*100}\nLog-det of the Fisher Information Matrix: {logdet_hF}\n{'='*100}")

    return per_dataset_results



def eval_single_task(args):

    eval_datasets = args.eval_datasets
    samples_nr = 200 # How many per-example gradients to accumulate
    
    for dataset in eval_datasets:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        train_dataset = dataset + "Val"

        # Training set
        image_encoder = ImageEncoder.load(pretrained_checkpoint)
        print("=" * 100)
        print(f"Evaluating on training set of {dataset}.")
        args.split = True
        eval_single_dataset(image_encoder, train_dataset, args)

        # Log-det of the Fisher Information Matrix
        classification_head = get_classification_head(args, train_dataset)
        model = ImageClassifier(image_encoder, classification_head)
        logdet_hF = train_diag_fim_logtr(args, model,  train_dataset , samples_nr)
        print(f"{'='*100}\nLog-det of the Fisher Information Matrix: {logdet_hF}\n{'='*100}")

        # Test set
        print("=" * 100)
        args.split = False
        print(f"Evaluating on test set of {dataset}.")
        eval_single_dataset(image_encoder, dataset, args)


    print("=" * 100)
    print(f"Evaluating on the finetuned models")
    print("=" * 100)

    for dataset in eval_datasets:
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        train_dataset = dataset + "Val"

        # Training set
        image_encoder = ImageEncoder.load(finetuned_checkpoint)
        print("=" * 100)
        print(f"Evaluating on training set.")
        args.split = True
        eval_single_dataset(image_encoder, dataset + "Val", args)

        # Log-det of the Fisher Information Matrix
        classification_head = get_classification_head(args, train_dataset)
        model = ImageClassifier(image_encoder, classification_head)
        logdet_hF = train_diag_fim_logtr(args, model,  train_dataset , samples_nr)
        print(f"{'='*100}\nLog-det of the Fisher Information Matrix: {logdet_hF}\n{'='*100}")

        # Test set
        print("=" * 100)
        args.split = False
        print(f"Evaluating on test set.")
        eval_single_dataset(image_encoder, dataset, args)
    return



if __name__ == '__main__':

    data_location = 'Task_Arithmetic_Datasets'
    model = 'ViT-B-32-quickgelu'
    datasets = ['DTD', 'EuroSAT']
    epochs = {
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SVHN': 4,
    }

    args = parse_arguments()
    args.lr = 1e-4
    args.epochs = epochs
    args.data_location = data_location
    args.batch_size = 32
    args.model = model

    args.save = f'checkpoints'                          #checkpoint directory
    eval_datasets = datasets
    args.eval_datasets = eval_datasets
    
    args.split = True                                   # Used only for the eval function. 
                                                        # True: Train split | False: Val split
    args.results_db = f'results'                        # Used to save the results in a .csv file
    eval_single_task(args)
