import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa import dsa_topk_backend
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAiterDecodeTopKDispatch(unittest.TestCase):
    def _can_use(
        self,
        *,
        rows=4,
        width=60_000,
        topk=2048,
        method=TopkTransformMethod.PAGED,
        row_starts=None,
        batch_idx_list=None,
        page_table=True,
        speculative_algorithm=None,
    ):
        logits = torch.empty((rows, width), dtype=torch.float32)
        lengths = torch.full((rows,), width, dtype=torch.int32)
        metadata = SimpleNamespace(
            page_table_1=(
                torch.empty((rows, 1), dtype=torch.int32) if page_table else None
            )
        )
        spec = SimpleNamespace(speculative_algorithm=speculative_algorithm)
        with (
            patch.object(dsa_topk_backend, "_is_gfx95", True),
            patch.object(dsa_topk_backend, "get_spec", return_value=spec),
        ):
            return dsa_topk_backend._can_use_aiter_decode_topk(
                logits=logits,
                lengths=lengths,
                topk=topk,
                topk_transform_method=method,
                attn_metadata=metadata,
                row_starts=row_starts,
                batch_idx_list=batch_idx_list,
            )

    def test_aiter_uses_sgl_fallback_without_topk_v2_metadata(self):
        self.assertTrue(DSATopKBackend.AITER.is_aiter())
        self.assertTrue(DSATopKBackend.AITER.is_sgl_kernel())
        self.assertFalse(DSATopKBackend.AITER.should_use_topk_v2())

    def test_supported_gfx950_decode_shapes(self):
        for width in (20_000, 32_768, 60_000, 65_535, 65_536, 128_000):
            with self.subTest(width=width):
                self.assertTrue(self._can_use(width=width))

    def test_unsupported_width_gap_and_row_count(self):
        self.assertFalse(self._can_use(width=20_001))
        self.assertFalse(self._can_use(width=32_767))
        self.assertFalse(self._can_use(rows=5))

    def test_requires_paged_plain_decode_contract(self):
        self.assertFalse(self._can_use(topk=1024))
        self.assertFalse(self._can_use(method=TopkTransformMethod.RAGGED))
        self.assertFalse(self._can_use(row_starts=torch.zeros(4, dtype=torch.int32)))
        self.assertFalse(self._can_use(batch_idx_list=[0, 1, 2, 3]))
        self.assertFalse(self._can_use(page_table=False))
        self.assertFalse(self._can_use(speculative_algorithm="EAGLE"))

    def test_requires_gfx950(self):
        logits = torch.empty((4, 60_000), dtype=torch.float32)
        lengths = torch.full((4,), 60_000, dtype=torch.int32)
        metadata = SimpleNamespace(page_table_1=torch.empty((4, 1), dtype=torch.int32))
        with (
            patch.object(dsa_topk_backend, "_is_gfx95", False),
            patch.object(
                dsa_topk_backend,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
        ):
            self.assertFalse(
                dsa_topk_backend._can_use_aiter_decode_topk(
                    logits=logits,
                    lengths=lengths,
                    topk=2048,
                    topk_transform_method=TopkTransformMethod.PAGED,
                    attn_metadata=metadata,
                    row_starts=None,
                    batch_idx_list=None,
                )
            )


if __name__ == "__main__":
    unittest.main()
