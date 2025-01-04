import tqdm
import torch
import utils
from datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier
from datasets.registry import get_dataset


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
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
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

    return per_dataset_results