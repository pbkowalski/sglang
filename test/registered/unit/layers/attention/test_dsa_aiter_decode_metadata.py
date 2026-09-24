import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterDSADecodeMetadata(CustomTestCase):
    @patch("sglang.srt.layers.attention.dsa_backend.get_mla_metadata_v1")
    def test_metadata_uses_only_active_kv_indptr_range(self, get_metadata):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend._ensure_aiter_dsa_decode_metadata_buffer = Mock()
        backend.aiter_dsa_kv_last_page_lens = torch.empty(8, dtype=torch.int32)
        backend.num_head_padded = 16
        backend.dsa_index_topk = 2048
        backend.aiter_dsa_max_split_per_batch = 64
        backend.aiter_dsa_work_metadata = object()
        backend.aiter_dsa_work_info_set = object()
        backend.aiter_dsa_work_indptr = object()
        backend.aiter_dsa_reduce_indptr = object()
        backend.aiter_dsa_reduce_final_map = object()
        backend.aiter_dsa_reduce_partial_map = object()

        qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
        kv_indptr = torch.tensor([0, 5, 11, 99, 99], dtype=torch.int32)

        backend._prepare_aiter_dsa_decode_metadata(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            bs=2,
            max_seqlen_q=1,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
        )

        passed_kv_indptr = get_metadata.call_args.args[1]
        torch.testing.assert_close(passed_kv_indptr, kv_indptr[:3])
        self.assertEqual(passed_kv_indptr.shape, (3,))

    def test_metadata_is_reused_only_within_one_decode(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.kv_indptr = torch.zeros(3, dtype=torch.int32)
        backend.aiter_dsa_decode_metadata_owner = None
        backend.aiter_dsa_decode_kv_last_page_lens = None
        backend.aiter_dsa_decode_persistent_kwargs = None

        prepared_results = [
            {
                "kv_last_page_lens": object(),
                "work_meta_data": object(),
            },
            {
                "kv_last_page_lens": object(),
                "work_meta_data": object(),
            },
        ]
        backend._prepare_aiter_dsa_decode_metadata = Mock(side_effect=prepared_results)

        def get_metadata(owner, page_table):
            return backend._get_aiter_dsa_decode_metadata(
                metadata_owner=owner,
                page_table_1=page_table,
                qo_indptr=object(),
                bs=2,
                max_seqlen_q=1,
                q_dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
            )

        first_owner = object()
        first = get_metadata(
            first_owner,
            torch.tensor([[4, 5, -1], [7, -1, -1]], dtype=torch.int32),
        )
        reused = get_metadata(
            first_owner,
            torch.tensor([[8, 9, -1], [10, -1, -1]], dtype=torch.int32),
        )

        self.assertEqual(
            backend._prepare_aiter_dsa_decode_metadata.call_count,
            1,
        )
        self.assertIs(first[0], reused[0])
        self.assertIs(first[1], reused[1])
        torch.testing.assert_close(
            backend.kv_indptr,
            torch.tensor([0, 2, 3], dtype=torch.int32),
        )

        second = get_metadata(
            object(),
            torch.tensor([[11, -1, -1], [12, 13, -1]], dtype=torch.int32),
        )

        self.assertEqual(
            backend._prepare_aiter_dsa_decode_metadata.call_count,
            2,
        )
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        torch.testing.assert_close(
            backend.kv_indptr,
            torch.tensor([0, 1, 3], dtype=torch.int32),
        )


if __name__ == "__main__":
    unittest.main()
