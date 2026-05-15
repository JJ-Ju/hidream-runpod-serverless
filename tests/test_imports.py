import os
import subprocess
import sys


def test_handler_imports_from_raw_checkout_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", "import handler; print(handler.handler.__name__)"],
        check=False,
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "handler"
