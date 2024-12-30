import time
import os   
import torch
from modeling import ImageClassifier, ImageEncoder
from datasets.registry import get_dataset
from datasets.templates import get_templates
from heads import get_classification_head
from datasets.common import get_dataloader, maybe_dictionarize
from eval_single_task import evaluate

from args import parse_arguments


def finetune(args):
    
    train_dataset = args.train_dataset
    print("Working on dataset: " + train_dataset)
    print('='*100)
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if checkpoints already exist
    ft_path = (
        os.path.join(args.save, train_dataset, "finetuned.pt")
    )
    zs_path = (
        os.path.join(args.save, train_dataset, "zeroshot.pt")
    )

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith('pt'):
        print(f"Loading model from {args.load}")
        image_encoder = ImageEncoder.load(args.load)
    else:
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
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=0)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')
        # Salva il checkpoint zeroshot solo se non esiste già
        if train_dataset.endswith('Val'):
            if not os.path.exists(zs_path):
                print(f"Saving zeroshot checkpoint at {zs_path}")
                image_encoder.save(zs_path)
            else:
                print(f"Zeroshot checkpoint already exists at {zs_path}, skipping save.")

    if args.train is not None and args.train is True:
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
                
                #is a technique to prevent the gradients from becoming too large during backpropagation.

                optimizer.step()
                batch_time = time.time() - start_time

                if step % print_every == 0:
                    percent_complete = 100 * i / len(data_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )
    
    if args.train:
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        print(f"Saving fine-tuned checkpoint at {ft_path}")
        image_encoder.save(ft_path)
        
    # Evaluate
    image_encoder = model.image_encoder
    evaluate(image_encoder, args) 

    return zs_path, ft_path

if __name__ == '__main__':

    data_location = 'Task_Arithmetic_Datasets'
    model = 'ViT-B-32-quickgelu'
    datasets = ['SVHN']
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
        args.batch_size = 32
        args.model = model

        args.save = f'checkpoints'                      #checkpoint directory
        args.eval_datasets = dataset + 'Val'            # Use Val for train and val, remove for test + split = False
        
        #args.load = f'checkpoints/DTDVal/finetuned.pt'  # Used for loading a model

        args.split = True                               # Used only for the eval function. 
                                                        # True: Train split | False: Val split

        args.train = True                             # Used to train the model
        #args.results_db = f'results'                   # Used to save the results in a .csv file
        finetune(args)


# Appending "Val" to the name allows for accessing the train split when
# is_train=True is passed to get_dataloader(..) (the validation split is accessed by
# passing is_train=False). For the test split you just use the name of the dataset and you
# pass is_train=False. 
