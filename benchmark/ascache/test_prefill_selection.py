"""CPU checks for the batched prefill selection boundary and output padding."""

from unittest import mock

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as backend


def test_selection_sorts_maps_slots_and_preserves_padding():
    """Selection sorts maps slots and preserves padding.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    selected = torch.tensor([[3, -1, 1], [-1, -1, -1]], dtype=torch.int32)
    slots = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    starts = torch.tensor([1, 0], dtype=torch.int32)
    pages = torch.full((3, 5), -99, dtype=torch.int32)
    raw = pages.clone()
    backend._finish_prefill_indexer_selection(
        selected=selected, k_slots=slots, ks=starts, page_indices=pages, raw_indices=raw
    )
    torch.testing.assert_close(
        pages[:2, :3], torch.tensor([[20, 40, -1], [-1, -1, -1]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        raw[:2, :3], torch.tensor([[0, 2, -1], [-1, -1, -1]], dtype=torch.int32)
    )
    assert torch.all(pages[2] == -99) and torch.all(pages[:, 3:] == -99)
    assert torch.all(raw[2] == -99) and torch.all(raw[:, 3:] == -99)


def test_metadata_dataclass_is_preserved():
    """Metadata dataclass is preserved.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    import dataclasses

    assert dataclasses.is_dataclass(backend.DSV4AttnMetadata)


def test_snapshot_keeps_existing_fused_decode_path():
    """Snapshot keeps existing fused decode path.

    Returns:
        None.

    Raises:
        AssertionError: The implementation violates the tested contract.
    """
    from types import SimpleNamespace

    from sglang.srt.environ import envs
    from sglang.srt.speculative.dspark_components import dspark_kv_inject as injection

    pool = SimpleNamespace(
        full_to_swa_index_mapping=torch.arange(20),
        translate_loc_from_full_to_swa=mock.Mock(),
    )
    model = SimpleNamespace(write_target_hidden_kv=mock.Mock())
    injector = injection.TargetHiddenKvInjector(
        draft_model=model,
        draft_model_runner=SimpleNamespace(token_to_kv_pool=pool),
        model_runner=SimpleNamespace(device=torch.device("cpu")),
        device=torch.device("cpu"),
        verify_num_draft_tokens=2,
        block_pos_offsets=torch.arange(2),
    )
    expected = torch.tensor([5, -1], dtype=torch.int32)
    with envs.SGLANG_OPT_DSPARK_INJECT_LOC_FUSE.override(True), mock.patch.object(
        injection, "is_unified_kv_triton", return_value=False
    ), mock.patch.object(
        injection, "can_commit_swa_loc", return_value=True
    ), mock.patch.object(
        injection, "commit_swa_loc", return_value=expected
    ) as fused:
        injector._inject_mla(
            pool=pool,
            target_hidden=torch.ones(2, 3),
            cache_loc=torch.tensor([2, 3]),
            positions=torch.tensor([7, 8]),
            cache_loc_2d=torch.tensor([[2, 3]]),
            commit_lens=torch.tensor([1]),
            row_valid=None,
        )
    fused.assert_called_once()
    pool.translate_loc_from_full_to_swa.assert_not_called()
    assert model.write_target_hidden_kv.call_args.kwargs["swa_loc"] is expected
