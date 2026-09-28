"""Third-party imports must not choose the application's environment file."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("previous", [None, "0", "1"])
def test_milvus_import_does_not_load_ancestor_env(tmp_path, previous):
    (tmp_path / ".env").write_text("IMPORT_SENTINEL=unexpected\nTOKEN_BUDGET=17\n", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()
    env = os.environ.copy()
    for name in ("IMPORT_SENTINEL", "TOKEN_BUDGET", "PYTHON_DOTENV_DISABLED"):
        env.pop(name, None)
    if previous is not None:
        env["PYTHON_DOTENV_DISABLED"] = previous
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    code = f"""
import os
os.chdir({str(child)!r})
import app.kb.milvus_client
assert 'IMPORT_SENTINEL' not in os.environ, 'dependency loaded an unselected env file'
assert 'TOKEN_BUDGET' not in os.environ, 'dependency overwrote config provenance'
assert os.environ.get('PYTHON_DOTENV_DISABLED') == {previous!r}
from dotenv import dotenv_values
assert dotenv_values('../.env')['TOKEN_BUDGET'] == '17'
"""
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
