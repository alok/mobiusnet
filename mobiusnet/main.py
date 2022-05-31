#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
)

import torch
import torch.nn.functional as F
from torch import Tensor, as_tensor
from torch import nn as nn


class MobiusLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        # TODO only handle square?
        super().__init__()
        self.weight = nn.utils.parametrizations.orthogonal(
            nn.Linear(in_features, out_features)
        )
        self.power: int = random.choice([0, 2])
        self.bias = nn.Parameter(torch.rand(in_features))
        self.offset = nn.Parameter(torch.rand(in_features))
        self.scaling = nn.Parameter(torch.rand((1,)))

    def forward(self, x: Tensor) -> Tensor:
        y = x - self.offset
        # TODO stop using .weight and use orthogonal init
        return self.bias + self.scaling * self.weight.weight @ y / (
            y.norm() ** self.power
        )

def check_orthogonality(orth_layer:MobiusLayer) -> bool:
    U = orth_layer.weight.weight
    return torch.allclose(U.T @ U, torch.eye(U.shape[0]))

if __name__ == "__main__":
    layer = MobiusLayer(10, 10)
    optim = torch.optim.Adam(layer.parameters(), lr=0.1)
    xs, ys = torch.rand(100, 10), torch.rand(100, 10)
    print(layer.weight.weight @ layer.weight.weight.T)
    for _ in range(1000):
        for x, y in zip(xs, ys):
            y_ = layer(x)
            loss = F.mse_loss(y_, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(loss)
    print(layer.weight.weight @ layer.weight.weight.T)
