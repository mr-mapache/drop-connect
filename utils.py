import os
from uuid import UUID
from typing import Iterator, Tuple, Protocol
from typing import Dict
from typing import Optional
from typing import List

from matplotlib.pyplot import figure, show, savefig
from matplotlib.axes import Axes

import torch
from torch import argmax
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from metrics import Metrics
from metrics import Summary

class Criterion(Protocol):
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        ...

class Data(Protocol):
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        ...

def serialize(model: Module, optimizer: Optimizer, criterion: Criterion):
    return {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion
    
    }

def accuracy(predictions: Tensor, target: Tensor) -> float:
    return (predictions == target).float().mean().item()

def predictions(output: Tensor):
    return argmax(output, dim=1)

def train(model: Module, criterion: Criterion, optimizer: Optimizer, data: Data, metrics: Metrics, device: str):
    model.train()
    metrics.start('train')

    for batch, (input, target) in enumerate(data, start=1):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        metrics.update(batch, loss.item(), accuracy(predictions(output), target))

    metrics.stop()

def test(model: Module, criterion: Criterion, data: Data, metrics: Metrics, device: str):
    with torch.no_grad():
        model.eval()
        metrics.start('test')
        for batch, (input, target) in enumerate(data, start=1):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            metrics.update(batch, loss.item(), accuracy(predictions(output), target))

        metrics.stop()

def plot(metrics: Dict[str, Metrics], metric: str, ax: Optional[Axes] = None):
    if ax is None:
        plot = figure()
        ax = plot.add_subplot()

    for key, value in metrics.items():
        ax.plot(value.history[metric], label=key)
    
    ax.legend()
    ax.set_title(metric)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)

    if ax is None:
        show()


def run(model: Module, optimizer: Optimizer, criterion: Criterion, device: str, data: Dict[str, Data], summary: Summary, epochs: int = 30):
    summary.open()
    summary.add_text('model', str(model))
    summary.add_text('optimizer', str(optimizer))
    summary.add_text('criterion', str(criterion))

    for epoch in range(epochs):
        train(model, criterion, optimizer, data['train'], summary.metrics['train'], device)
        test(model, criterion, data['test'], summary.metrics['test'], device)

    summary.close()

    metrics_plot = figure(figsize=(10, 5))
    metrics_plot.suptitle(f'{summary.name}')
    ax = metrics_plot.add_subplot(1, 2, 1)
    plot(summary.metrics, 'loss', ax)

    ax = metrics_plot.add_subplot(1, 2, 2)
    plot(summary.metrics, 'accuracy', ax)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    savefig(f'plots/{summary.name}-{summary.id}.pdf', bbox_inches ="tight" )
    show()
