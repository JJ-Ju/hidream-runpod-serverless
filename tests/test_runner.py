import pytest

from hidream_o1.runner import HiDreamRunner


def test_runner_rejects_invalid_attention_backend():
    with pytest.raises(ValueError, match="ATTENTION_BACKEND"):
        HiDreamRunner(attention_backend="typo")


def test_runner_accepts_auto_as_sdpa_safe_default():
    runner = HiDreamRunner(attention_backend="auto")

    assert runner.attention_backend == "sdpa"
    assert runner.use_flash_attn is False
