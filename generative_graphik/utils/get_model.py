from argparse import Namespace
import json
from pathlib import Path
from typing import Dict

import torch
import yaml

from generative_graphik.model import Model

_model = None  # Use get_model to access the model
_model_frozen: bool = False  # Use get_model to access the model
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR.joinpath('config.yaml')


def get_config() -> Dict:
    """Loads the configuration file"""
    with CONFIG_DIR.open('r') as f:
        return yaml.safe_load(f)


def get_model(force_reload: bool = False,
              freeze: bool = False,
              cache: bool = True
              ) -> Model:
    """Loads the model specified in the configuration file or returns the cached model."""
    global _model
    if _model is not None and not force_reload:
        if freeze is not _model_frozen:
            _model_frozen = freeze
            for param in _model.parameters():
                param.requires_grad = not freeze
        if freeze:
            model.eval()
        else:
            model.train()
        return _model

    config = get_config()
    d = Path(config['model'])
    if torch.cuda.is_available():
        state_dict = torch.load(d.joinpath('net.pth'), map_location='cuda')
    else:
        state_dict = torch.load(d.joinpath('net.pth'), map_location='cpu')
    with d.joinpath('hyperparameters.txt').open('r') as f:
        args = Namespace(**json.load(f))
    model = Model(args)
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = not freeze
    if freeze:
        model.eval()
    else:
        model.train()

    if cache:
        _model = model
    return model
