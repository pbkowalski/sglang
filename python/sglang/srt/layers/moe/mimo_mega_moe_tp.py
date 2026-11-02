from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe.mega_moe_flydsl import (
    _rank_consistent_max_tokens,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_MEGA_TP_INSTANCE: dict[tuple, Any] = {}
_EXPECTED_CHECKPOINT_WEIGHTS = 384 * 6


def _cap() -> int:
    return int(os.getenv("SGLANG_MIMO_MEGA_TP_MAX_TOKENS", "128"))


def should_use_mimo_mega_moe_tp(moe: Any, hidden_states: torch.Tensor) -> bool:
    return (
        len(getattr(moe, "_mega_tp_loaded_weights", ())) == _EXPECTED_CHECKPOINT_WEIGHTS
        and _rank_consistent_max_tokens(hidden_states) <= _cap()
    )


def _build_weights(moe: Any) -> None:
    if hasattr(moe, "_mega_tp_w1"):
        return
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    experts = moe.config.n_routed_experts
    inter = moe.config.moe_intermediate_size // moe.tp_size
    hidden = moe.config.hidden_size
    moe._mega_tp_w1 = shuffle_weight(
        moe._mega_tp_w1_raw.view(dtypes.fp4x2), layout=(16, 16)
    )
    moe._mega_tp_w1_scale = fp4_utils.e8m0_shuffle(
        moe._mega_tp_w1_scale_raw.view(experts * 2 * inter, hidden // 32)
    )
    moe._mega_tp_w2 = shuffle_weight(
        moe._mega_tp_w2_raw.view(dtypes.fp4x2), layout=(16, 16)
    )
    moe._mega_tp_w2_scale = fp4_utils.e8m0_shuffle(
        moe._mega_tp_w2_scale_raw.view(experts * hidden, inter // 32)
    )
    for name in (
        "_mega_tp_w1_raw",
        "_mega_tp_w1_scale_raw",
        "_mega_tp_w2_raw",
        "_mega_tp_w2_scale_raw",
    ):
        moe._buffers.pop(name, None)
        if hasattr(moe, name):
            delattr(moe, name)


def _get_instance(moe: Any):
    from aiter import ActivationType
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    _build_weights(moe)
    torch.cuda.empty_cache()
    parallel = get_parallel()
    cap = _cap()
    key = (
        parallel.tp_rank,
        parallel.tp_size,
        moe.config.hidden_size,
        moe.config.moe_intermediate_size,
        moe.config.n_routed_experts,
        moe.config.num_experts_per_tok,
        cap,
    )
    instance = _MEGA_TP_INSTANCE.get(key)
    if instance is None:
        instance = MegaMoeTP(
            MegaMoeTPConfig(
                rank=parallel.tp_rank,
                world_size=parallel.tp_size,
                model_dim=moe.config.hidden_size,
                inter_dim=moe.config.moe_intermediate_size // parallel.tp_size,
                experts=moe.config.n_routed_experts,
                topk=moe.config.num_experts_per_tok,
                max_local_tokens=math.ceil(cap / parallel.tp_size),
                activation=ActivationType.Silu,
                comm_mode="ar",
                comm_dtype="bf16",
                ar_gather="bf16",
                act_dtype="fp4",
                bigm_min_tokens=0,
                fit_max_local=True,
            ),
            w1=moe._mega_tp_w1,
            w1_scale=moe._mega_tp_w1_scale,
            w2=moe._mega_tp_w2,
            w2_scale=moe._mega_tp_w2_scale,
            group=parallel.tp_group.device_group,
            device=moe._mega_tp_w1.device,
        )
        buckets = [
            math.ceil(tokens / parallel.tp_size)
            for tokens in (32, 64, 128)
            if tokens <= cap
        ]
        instance.prepare(buckets)
        _MEGA_TP_INSTANCE[key] = instance
    else:
        instance.set_weights(
            w1=moe._mega_tp_w1,
            w1_scale=moe._mega_tp_w1_scale,
            w2=moe._mega_tp_w2,
            w2_scale=moe._mega_tp_w2_scale,
        )
    return instance


def prepare_mimo_mega_moe_tp(moe: Any) -> None:
    _get_instance(moe)


def forward_mimo_mega_moe_tp(
    moe: Any,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    router_logits = moe._mega_moe_router_logits(hidden_states, forward_batch)
    topk_output = moe.topk(hidden_states, router_logits)
    output = _get_instance(moe)(
        hidden_states.contiguous(),
        topk_output.topk_weights.to(torch.float32).contiguous(),
        topk_output.topk_ids.to(torch.int32).contiguous(),
    )
    return output.mul(1.0 / get_parallel().moe_ep_size)
