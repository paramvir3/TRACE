"""One process of a training run for tests/test_train_data_parallel.py (not a test module).

Run alone (one process) or under ``torchrun`` (one per rank).  Records, for the test to compare:

* every batch this process evaluated, by its energy labels (``AtomisticDataset.collate_batched``
  is called once per evaluated batch, in training and in validation);
* the gradients handed to every optimiser step and the parameters after it;
* the history ``train.main`` returns.

Usage: python data_parallel_worker.py CONFIG OUT_PREFIX  ->  OUT_PREFIX_rank<R>.pt
"""
import math
import os
import sys
import warnings

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(config_path, out_prefix):
    warnings.simplefilter("ignore")
    import train

    record = {"batches": [], "grads": [], "params": []}
    collate = train.AtomisticDataset.collate_batched

    def recording_collate(items):
        record["batches"].append(tuple(float(item["t_E"]) for item in items))
        return collate(items)

    train.AtomisticDataset.collate_batched = staticmethod(recording_collate)

    def parameters(optimizer):
        return [p for group in optimizer.param_groups for p in group["params"]]

    # TRACE_TEST_PERTURB_RANK=r: rank r's optimiser step lands one ulp off (parameters and optimiser
    # state), as a non-deterministic kernel can make it; registered first, so the record shows it.
    perturbed = os.environ.get("TRACE_TEST_PERTURB_RANK")
    if perturbed is not None and int(perturbed) == int(os.environ.get("RANK", 0)):
        def nudge(optimizer, args, kwargs):
            with torch.no_grad():
                for p in parameters(optimizer):
                    for t in [p] + [v for v in optimizer.state.get(p, {}).values() if torch.is_tensor(v)]:
                        t.copy_(torch.nextafter(t, torch.full_like(t, math.inf)))
        register_optimizer_step_post_hook(nudge)

    register_optimizer_step_pre_hook(lambda optimizer, args, kwargs: record["grads"].append(
        [None if p.grad is None else p.grad.detach().clone() for p in parameters(optimizer)]))
    register_optimizer_step_post_hook(lambda optimizer, args, kwargs: record["params"].append(
        [p.detach().clone() for p in parameters(optimizer)]))

    record["history"] = train.main(["--config", config_path])
    torch.save(record, f"{out_prefix}_rank{int(os.environ.get('RANK', 0))}.pt")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
