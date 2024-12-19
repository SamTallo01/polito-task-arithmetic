import time
import os   
import torch
from modeling import ImageClassifier, ImageEncoder
from datasets.registry import get_dataset
from datasets.templates import get_templates
from heads import get_classification_head
from datasets.common import get_dataloader, maybe_dictionarize


from args import parse_arguments


def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if checkpoints already exist
    # zs_path = os.path.join(args.save, train_dataset, 'checkpoint_0.pt')  
    # ft_path = os.path.join(args.save, train_dataset, f'checkpoint_{args.epochs}.pt')
    # if os.path.exists(zs_path) and os.path.exists(ft_path):
    #     print(f'Skipping fine-tuning because {ft_path} exists.')
    #     return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

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
    
    #  This part of the code used for multi-gpu training
    # devices = list(range(torch.cuda.device_count()))
    # print('Using devices', devices)
    # model = torch.nn.DataParallel(model, device_ids=devices)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CrossEntropyLoss is used as the loss function for the model
    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    
    #SGD is used as the optimizer for the model
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0001)



    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f'zeroshot.pt')
        model.module.image_encoder.save(model_path)

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
    
            optimizer.zero_grad()
            #This is to ensures the input batches used during training or evaluation are in a consistent format
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            #logit is for storing the output of the last layer and training the model
            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()
            
            #is a technique to prevent the gradients from becoming too large during backpropagation.
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    # Evaluate (implementation is missing)
    image_encoder = model.module.image_encoder
    #evaluate(image_encoder, args) 

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')  
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
        return zs_path, ft_path




if __name__ == '__main__':
    data_location = '<your_data_location>'
    models = ['ViT-B-32']
    datasets = ['DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45','SVHN']
    epochs = {
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SVHN': 4,
    }

    for model in models:
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
            args.save = f'checkpoints/{model}'
            finetune(args)