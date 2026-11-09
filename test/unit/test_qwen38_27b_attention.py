import torch
import pytest

from sgl_amd_mono.qwen38_27b.errors import ContractError
from sgl_amd_mono.qwen38_27b.gfx936.attention import cache_views, rotary_axes


@pytest.mark.parametrize("layout", ["legacy", "hnd", "nhd"])
def test_cache_views_preserve_native_addresses(layout):
    canonical = torch.arange(3 * 64 * 4 * 256).view(3, 64, 4, 256)
    if layout == "legacy":
        key = canonical.permute(0, 2, 1, 3).contiguous()
        value = canonical.permute(0, 2, 3, 1).contiguous()
    elif layout == "hnd":
        key = canonical.permute(0, 2, 1, 3).contiguous()
        value = key.clone()
    else:
        key = canonical.flatten(0, 1)
        value = key.clone()
    key_view, value_view = cache_views(key, value, 64, layout)
    assert torch.equal(key_view, canonical)
    assert torch.equal(value_view, canonical)
    key_view[2, 17, 3, 255] = -1
    if layout in {"legacy", "hnd"}:
        assert key[2, 3, 17, 255] == -1
    else:
        assert key[2 * 64 + 17, 3, 255] == -1


@pytest.mark.parametrize("layout", ["legacy", "hnd", "nhd", "unknown"])
def test_cache_views_refuse_incompatible_geometry(layout):
    with pytest.raises(ContractError):
        cache_views(torch.empty(3, 63, 4, 256), torch.empty(3, 63, 4, 256), 64, layout)


def test_qwen_interleaved_mrope_axes_match_native_sections():
    axes = rotary_axes(64, (11, 11, 10), True)
    assert axes == [0, 1, 2] * 10 + [0, 1]
    assert rotary_axes(64, (11, 11, 10), False) == [0] * 11 + [1] * 11 + [2] * 10


@pytest.mark.parametrize("dimension,sections", [(0, (0, 0, 0)), (63, (11, 11, 10)), (64, (10, 10, 10)), (512, (86, 85, 85))])
def test_invalid_mrope_geometry_is_rejected(dimension, sections):
    with pytest.raises(ContractError):
        rotary_axes(dimension, sections, True)


@pytest.mark.parametrize("actual,native,accepted", [(0.0077, 0.0076, True), (0.012, 0.0076, False), (float("nan"), 0.0076, False), (0.001, float("inf"), False)])
def test_fp32_comparison_accounts_for_native_intermediate_rounding(actual, native, accepted):
    from sgl_amd_mono.qwen38_27b.gfx936.attention_check import fp32_reference_accepted

    assert fp32_reference_accepted({"finite": True, "normalized_rms_error": actual}, {"finite": True, "normalized_rms_error": native}) is accepted
