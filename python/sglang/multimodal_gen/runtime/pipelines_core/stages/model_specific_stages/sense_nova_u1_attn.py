# SPDX-License-Identifier: Apache-2.0
"""Attention dispatch for SenseNova MoT: LightX2V FA3, SGLang FA, FA2, then SDPA."""

from __future__ import annotations

from collections.abc import Callable

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _unwrap_attn_out(out):
    return out[0] if isinstance(out, tuple) else out


def _try_lightx2v_fa3() -> Callable | None:
    """Hopper-class FA3 used by LightX2V neopp (`attn_type=flash_attn3`)."""
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    except ImportError:
        return None
    if flash_attn_func_v3 is None:
        return None

    def _fa3(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
        del dropout_p
        return _unwrap_attn_out(
            flash_attn_func_v3(q, k, v, softmax_scale=softmax_scale, causal=causal)
        )

    return _fa3


def _try_sglang_fa(ver: int) -> Callable | None:
    """SGLang FA3 (ver=3) / FA4 (ver=4); dense [B, S, H, D] via cu_seqlens=None."""
    try:
        from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
    except ImportError:
        return None

    def _fa(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
        del dropout_p
        return _unwrap_attn_out(
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                softmax_scale=softmax_scale,
                causal=causal,
                return_softmax_lse=False,
                ver=ver,
            )
        )

    return _fa


def _try_fa2() -> Callable | None:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
    except ImportError:
        return None
    if flash_attn_func_v2 is None:
        return None

    def _fa2(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
        return _unwrap_attn_out(
            flash_attn_func_v2(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        )

    return _fa2


def list_attn_candidates(sdpa: Callable) -> list[tuple[str, Callable]]:
    raw: list[tuple[str, Callable | None]] = [
        ("flash_attn3", _try_lightx2v_fa3()),
        ("sglang_fa3", _try_sglang_fa(3)),
        ("sglang_fa4", _try_sglang_fa(4)),
        ("flash_attn2", _try_fa2()),
    ]
    found = [(name, impl) for name, impl in raw if impl is not None]
    found.append(("sdpa", sdpa))
    return found


class AttnDispatch:
    """Walk remaining FA impls on kernel exception; do not jump to SDPA early."""

    def __init__(self, candidates: list[tuple[str, Callable]]):
        if not candidates:
            raise ValueError("AttnDispatch requires at least one candidate")
        self._queue = list(candidates)
        self.name, self.impl = self._queue.pop(0)

    def __call__(self, q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
        while True:
            try:
                return self.impl(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            except Exception as exc:
                if self.name == "sdpa" or not self._queue:
                    raise
                next_name = self._queue[0][0]
                logger.warning(
                    "SenseNova attention %s failed (%s); trying %s",
                    self.name,
                    exc,
                    next_name,
                )
                self.name, self.impl = self._queue.pop(0)


def install_fast_attn_backend() -> str:
    """Replace official `_flash_or_sdpa` with FA3 / SGLang FA / FA2 / SDPA.

    Do not call `set_attn_backend('flash')`: SGLang's FA4 package occupies
    top-level `flash_attn` without `flash_attn_func`, so that helper raises.
    """
    import sensenova_u1.models.neo_unify.modeling_qwen3 as qwen3

    sdpa = qwen3._sdpa_attn_func
    dispatch = AttnDispatch(list_attn_candidates(sdpa))
    qwen3._flash_or_sdpa = dispatch
    logger.info("SenseNova MoT attention backend: %s", dispatch.name)
    return dispatch.name
