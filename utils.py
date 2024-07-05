from typing import Iterator, Tuple, Protocol
from typing import Dict
from typing import Optional
from typing import List

from matplotlib.pyplot import figure, show
from matplotlib.axes import Axes

import torch
from torch import argmax
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

class Criterion(Protocol):
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        ...

class Data(Protocol):
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        ...

class Metrics(Protocol):
    history: Dict[str, List]

    def start(self, mode: str):
        ...

    def update(self, batch: int, loss: float, accuracy: float):
        ...
    
    def stop(self):
        ...
    def reset(self):
        ...

class Summary(Protocol):
    metrics: Dict[str, Metrics]
    
    def open(self):
        ...

    def close(self):
        ...

    def add_text(self, tag: str, text: str):
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
    plot(summary.metrics, 'loss')
    plot(summary.metrics, 'accuracy')