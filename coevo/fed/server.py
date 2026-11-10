"""Server side: model aggregation."""
from __future__ import annotations

from collections import OrderedDict

import torch


class FedAvgAggregator:
    """theta = sum_k (n_k / N) * theta_k over every floating-point entry of the state
    dict (BatchNorm running statistics included). Integer buffers such as
    `num_batches_tracked` are copied from the first client."""

    def aggregate(self, states: list[dict], num_samples: list[int]) -> OrderedDict:
        if len(states) != len(num_samples) or not states:
            raise ValueError("need one sample count per client state")
        total = float(sum(num_samples))
        weights = [n / total for n in num_samples]
        out = OrderedDict()
        for key, first in states[0].items():
            if torch.is_floating_point(first):
                acc = torch.zeros_like(first, dtype=torch.float64)
                for state, w in zip(states, weights):
                    acc += state[key].to(torch.float64) * w
                out[key] = acc.to(first.dtype)
            else:
                out[key] = first.clone()
        return out


def build_aggregator(fed_cfg) -> FedAvgAggregator:
    if fed_cfg.aggregator == "fedavg":
        return FedAvgAggregator()
    raise KeyError(f"unknown aggregator '{fed_cfg.aggregator}'")
