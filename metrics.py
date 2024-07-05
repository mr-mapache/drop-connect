from uuid import UUID, uuid4
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from typing import Protocol, Optional

class Writer(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        ...
    
class Metrics:
    def __init__(self, writer: Optional[Writer] = None):
        self.writer = writer
        self.history = {
            'loss': [],
            'accuracy': [],
        }
        self.epoch = 0

    def start(self, mode: str):
        self.mode = mode
        self.epoch += 1
        self.batch = 0
        self.loss = 0
        self.accuracy = 0

    def update(self, batch: int, loss: float, accuracy: float):
        self.batch = batch
        self.loss += loss
        self.accuracy += accuracy
    
    def stop(self):
        self.loss /= self.batch
        self.accuracy /= self.batch
        self.history['loss'].append(self.loss)
        self.history['accuracy'].append(self.accuracy)
        print(f'Processed {self.batch} batches, average loss: {self.loss:.4f}, average accuracy: {self.accuracy:.4f}, in epoch {self.epoch} for {self.mode} mode')

        if self.writer:
            self.writer.add_scalar(f'{self.mode}/loss', self.loss, self.epoch)
            self.writer.add_scalar(f'{self.mode}/accuracy', self.accuracy, self.epoch)

    def reset(self):
        self.history = {
            'loss': [],
            'accuracy': []
        }
        self.batch = 0
        self.loss = 0
        self.accuracy = 0

class Sumary:
    def __init__(self, name: str, id: UUID = None) -> None:
        self.id = id or uuid4()
        self.name = name
        self.metrics = {
            'train': Metrics(),
            'test': Metrics()
        }

    def open(self):
        self.writer = SummaryWriter(log_dir=f'runs/{self.name}-{self.id}')
        self.metrics['train'].writer = self.writer
        self.metrics['test'].writer = self.writer
        print(f"Running experiment {self.name} with id {self.id}")
        print(f"Tensorboard logs are saved in runs/{self.name}-{self.id}")
        print(f"Run tensorboard with: tensorboard --logdir=runs/")
        print(f"Open browser and go to: http://localhost:6006/")
        print(f"----------------------------------------------------------------")

    def close(self):
        self.writer.close()
 
    def add_text(self, tag: str, text: str):
        if self.writer:
            self.writer.add_text(tag, text)
        print(f'{tag}: {text}')
        print(f"----------------------------------------------------------------")