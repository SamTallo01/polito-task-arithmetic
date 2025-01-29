import time
import os   
import torch
from modeling import ImageClassifier, ImageEncoder
from datasets.registry import get_dataset
from heads import get_classification_head
from utils import train_diag_fim_logtr
from eval import evaluate
from datasets.common import get_dataloader, maybe_dictionarize
from args import parse_arguments

def eval_best_fim_checkpoint(args, model, epoch, best_fim):
    """
    Evaluate and update the best FIM checkpoint based on the diagonal FIM log-trace criterion
    """
    samples_nr = 200
    logdet_hF = train_diag_fim_logtr(args, model, args.train_dataset , samples_nr)
    
     # Update the best FIM checkpoint if the log-trace improves
    if logdet_hF > best_fim["value"]:
        # Delete the previous best checkpoint
        if best_fim["path"] and os.path.exists(best_fim["path"]):
            os.remove(best_fim["path"])
        
        best_fim["value"] = logdet_hF
        dataset_folder = os.path.join(args.save, args.train_dataset)
        best_fim["path"] = os.path.join(dataset_folder, f"best_fim_epoch_{epoch}.pt")
        model.image_encoder.save(best_fim["path"])
        print(f"New best FIM checkpoint: {best_fim['path']} (Log-Trace: {logdet_hF:.4f})")
    return best_fim

def eval_best_val_checkpoint(args, model, epoch, best_val):
    """
    Evaluate and update the best validation accuray criterion
    """
    image_encoder = model.image_encoder
    val_acc_metrics = evaluate(image_encoder, args)
    val_acc = val_acc_metrics[args.train_dataset + ":top1"]
    
    # Update the best validation accuray checkpoint if it improves
    if val_acc > best_val["value"]:
        # Delete the previous best checkpoint
        if best_val["path"] and os.path.exists(best_val["path"]):
            os.remove(best_val["path"])
            
        best_val["value"] = val_acc
        dataset_folder = os.path.join(args.save, args.train_dataset)
        best_val["path"] = os.path.join(dataset_folder, f"best_val_epoch_{epoch}.pt")
        image_encoder.save(best_val["path"])
        print(f"New best validation accuracy checkpoint saved at {best_val['path']} with accuracy {val_acc:.4f}")

    return best_val


def finetune(args):
    
    train_dataset = args.train_dataset
    print("Working on dataset: " + train_dataset)
    print('='*100)
    ckpdir = os.path.join(args.save, train_dataset)

    assert train_dataset is not None, "Please provide a training dataset."

    print('Building image encoder.')
    image_encoder = ImageEncoder(args)

    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )

    num_batches = len(dataset.train_loader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print('='*100)

    # CrossEntropyLoss is used as the loss function for the model
    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    
    #SGD is used as the optimizer for the model
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wd)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')
        # Save the zeroshot model
        if train_dataset.endswith('Val'):
            if not os.path.exists(zs_path):
                print(f"Saving zeroshot checkpoint at {zs_path}")
                image_encoder.save(zs_path)
            else:
                print(f"Zeroshot checkpoint already exists at {zs_path}, skipping save.")

    # # best_fim: Tracks the checkpoint with the highest diagonal FIM log-trace value.
    # best_fim = {"value": float("-inf"), "path": None}
    # # best_val_acc: Tracks the checkpoint with the highest validation accuracy.
    # best_val_acc = {"value": float("-inf"), "path": None}

    for epoch in range(args.epochs):
        model = model.to(device)
        model.train()
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
    
            optimizer.zero_grad()

            #This is to ensures the input batches used during training or evaluation are in a consistent format
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(device)
            labels = batch['labels'].to(device)

            data_time = time.time() - start_time

            #logit is for storing the output of the last layer and training the model
            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()
            
            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

            # #Evaluate and save the best FIM checkpoint
            # best_fim = eval_best_fim_checkpoint(args, model, epoch, best_fim)
            # print("=" * 100)

            # # Evaluate and save the best valisation accuracy checkpoint
            # best_val_acc = eval_best_val_checkpoint(args, model, epoch, best_val_acc)
            # print("=" * 100)

    # #Final evaluation
    # print("=" * 100)
    # print(f"Finetuning completed for {train_dataset}.")
    # print(f"Best FIM checkpoint: {best_fim['path']} with log-trace {best_fim['value']:.4f}")
    # print(f"Best validation accuracy checkpoint: {best_val_acc['path']} with accuracy {best_val_acc['value']:.4f}")
    # print("=" * 100)

    # return best_fim["path"], best_val_acc["path"]
    
    if args.save is not None:
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        print(f"Saving fine-tuned checkpoint at {ft_path}")
        image_encoder.save(ft_path)
        
    return zs_path, ft_path

if __name__ == '__main__':

    data_location = 'Task_Arithmetic_Datasets'
    model = 'ViT-B-32-quickgelu'
    datasets = ['DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SVHN']
    epochs = {
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SVHN': 4,
    }

    for dataset in datasets:
        print('='*100)
        print(f'Finetuning {model} on {dataset}')
        print('='*100)
        args = parse_arguments()
        args.lr = 1e-4
        args.epochs = epochs[dataset]
        args.data_location = data_location
        args.train_dataset = dataset + 'Val' 
        args.wd = 0.0
        args.batch_size = 32
        args.model = model
        args.save = f'checkpoints'                                                  # Checkpoint directory
        #args.split = True                                                          # Used for the evaluate for the best fim checkpoints and the best validation accuracy checkpoints                     
        finetune(args)
