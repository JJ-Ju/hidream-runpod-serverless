import pytest

from hidream_o1.runner import HiDreamRunner


def test_runner_rejects_invalid_attention_backend():
    with pytest.raises(ValueError, match="ATTENTION_BACKEND"):
        HiDreamRunner(attention_backend="typo")
