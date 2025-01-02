import os
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset

def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu")
    if device is not None:
        model = model.to(device)
    return model

def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


class DotDict(dict): 
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def train_diag_fim_logtr(
            args,
            model,
            dataset_name: str,
            samples_nr: int = 2000):
    
    model.cuda()
    if not dataset_name.endswith('Val'):
        dataset_name += 'Val'

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=0
    )
    data_loader = torch.utils.data.DataLoader(
        dataset.train_dataset, 
        batch_size=args.batch_size, 
        num_workers=0, shuffle=False)
    
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    progress_bar = tqdm(total=samples_nr)
    seen_nr = 0

    while seen_nr < samples_nr:
        data_iterator = iter(data_loader)
        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data = next(data_loader)
        data = maybe_dictionarize(data)
        x, y = data['images'], data['labels']
        x, y = x.cuda(), y.cuda()

        logits = model(x)
        outdx = torch.distributions.Categorical(logits=logits).sample().unsqueeze(1).detach()
        samples = logits.gather(1, outdx)

        idx, batch_size = 0, x.size(0)

        for idx in range(batch_size):
            model.zero_grad()
            torch.autograd.backward(samples[idx], retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad and hasattr(param, 'grad') and param.grad is not None:
                    fim[name] += (param.grad * param.grad)
                    fim[name].detach_()
            seen_nr += 1
            progress_bar.update(1)
            if seen_nr >= samples_nr: break

    fim_trace = 0.0
    for name, grad2 in fim.items():
        fim_trace += grad2.sum()
    fim_trace = torch.log(fim_trace / samples_nr).item()

    return fim_trace

def find_optimal_coef(
    results,
    metric="avg_normalized_top1",
    minimize=False,
    control_metric=None,
    control_metric_threshold=0.0,
):
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef
