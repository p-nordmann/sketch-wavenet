import os
from dataclasses import dataclass

import numpy as np
from tensorboardX import SummaryWriter

MetricsT = dict[str, float]


@dataclass
class TensorboardLogger:
    log_dir: str
    name: str

    def __post_init__(self):
        self.summary_writer = SummaryWriter(os.path.join(self.log_dir, self.name))
        self.n = 0

    def log(self, metrics: MetricsT) -> None:
        """Logs metrics."""
        for name, value in metrics.items():
            self.summary_writer.add_scalar(name, value, global_step=self.n)
        self.n += 1

    def log_image(self, name: str, image: np.ndarray) -> None:
        self.summary_writer.add_image(name, image, global_step=self.n)

    @property
    def step(self) -> int:
        return self.n

    @step.setter
    def step(self, value: int):
        self.n = value
