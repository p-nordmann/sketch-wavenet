from typing import Any

import toml
from eqx_wavenet import WavenetConfig
from pydantic import BaseModel, field_serializer


class RandomConfig(BaseModel):
    seed_data: int
    seed_model: int
    seed_training: int


class FilesConfig(BaseModel):
    log_dir: str
    examples_dir: str
    out_dir: str


class DataConfig(BaseModel):
    files: list[str]
    training_prop: float
    dev_prop: float
    test_prop: float
    rescale_data: bool
    max_stroke_len: int
    use_data_augmentation: bool


class ModelConfig(BaseModel):
    num_gaussians: int
    wavenet: WavenetConfig

    @field_serializer("wavenet")
    def serialize_wavenet(self, config: WavenetConfig, _info):
        return config._asdict()


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    validate_each: int
    use_dropout: bool
    optimizer: str
    weight_decay: float
    use_gradient_clipping: bool
    schedule: str
    base_learning_rate: float
    peak_learning_rate: float
    pct_start: float
    div_factor: float
    final_div_factor: float


class FullConfig(BaseModel):
    random: RandomConfig
    files: FilesConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def deep_merge(base: dict[Any, Any], patch: dict[Any, Any]):
    """Recursively merges two dictionaries into `base`

    `patch` values will take precedence over `base` values.
    """
    for key in patch:
        if key in base and isinstance(base[key], dict) and isinstance(patch[key], dict):
            deep_merge(base[key], patch[key])
        else:
            base[key] = patch[key]


def read_tomls(paths: list[str]) -> FullConfig:
    config_dict = {}
    for path in paths:
        with open(path, "r") as f:
            deep_merge(config_dict, toml.load(f))
    return FullConfig(**config_dict)


def write_toml(path: str, config: BaseModel) -> None:
    with open(path, "w") as f:
        toml.dump(config.model_dump(), f)
