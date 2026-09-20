"""v0.4.10 C12: process isolation is loud, bounded, and cannot hang on a large result.

Closes v12 1.28, 2.26, 2.35, 2.49.
"""

import subprocess
import sys
import time
from importlib.util import find_spec

import pytest
from sklearn.linear_model import LogisticRegression

import ictonyx as ix
import ictonyx.memory as memory
from ictonyx.memory import MemoryManager

pytestmark = pytest.mark.skipif(find_spec("cloudpickle") is None, reason="needs ictonyx[isolation]")


def _big(n):
    return {"history": {"loss": [0.0] * n}}


def test_large_result_payload_round_trips():
    """A 200k-float result (~1.7 MB pickled) used to hang until process_timeout."""
    m = MemoryManager(use_process_isolation=True, process_timeout=60)
    t = time.time()
    r = m.run_isolated(_big, args=(200_000,))
    assert r["success"] is True
    assert len(r["result"]["history"]["loss"]) == 200_000
    assert time.time() - t < 30


def test_process_timeout_is_honoured_and_reported():
    import time as _t

    def _sleep(s):
        _t.sleep(s)
        return {"success": True, "result": None}

    m = MemoryManager(use_process_isolation=True, process_timeout=2)
    t = time.time()
    r = m.run_isolated(_sleep, args=(20,))
    assert r["success"] is False
    assert "within 2s" in r["error"]
    assert time.time() - t < 15


def test_unserialisable_builder_raises_by_default(monkeypatch):
    monkeypatch.setattr(memory, "HAS_CLOUDPICKLE", False)
    m = MemoryManager(use_process_isolation=True)
    with pytest.raises(RuntimeError, match="cannot be serialised"):
        m.run_isolated(lambda: 1)


def test_fallback_is_flagged_and_never_logged_as_isolated(monkeypatch, caplog, X, y):
    from ictonyx.api import _get_model_builder
    from ictonyx.config import ModelConfig
    from ictonyx.data import ArraysDataHandler
    from ictonyx.runners import ExperimentRunner

    monkeypatch.setattr(memory, "HAS_CLOUDPICKLE", False)
    runner = ExperimentRunner(
        lambda cfg: ix.ScikitLearnModelWrapper(LogisticRegression(max_iter=200)),  # unpicklable
        ArraysDataHandler(X, y),
        ModelConfig({}),
        use_process_isolation=True,
        allow_isolation_fallback=True,
        seed=0,
        verbose=False,  # no progress bar: warnings go to the logger, not tqdm
    )
    with pytest.warns(UserWarning, match="NOT isolated"):
        r = runner.run_study(num_runs=2)
    assert r.n_runs == 2
    assert runner.fallback_runs == [1, 2]
    assert "(isolated)" not in caplog.text


def test_process_timeout_reaches_memory_manager(X, y, monkeypatch):
    seen = {}
    real_init = MemoryManager.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        real_init(self, *a, **kw)

    monkeypatch.setattr(MemoryManager, "__init__", spy)
    ix.variability_study(
        LogisticRegression,
        data=(X, y),
        runs=2,
        seed=0,
        use_process_isolation=True,
        process_timeout=123,
        verbose=False,
    )
    assert seen.get("process_timeout") == 123


def test_missing_main_guard_is_named(tmp_path):
    """A script without a __main__ guard re-runs itself in every spawned child; the
    child must fail with an Ictonyx message naming the guard, not a Python bootstrap error."""
    script = tmp_path / "noguard.py"
    script.write_text(
        "import numpy as np\n"
        "import ictonyx as ix\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "X = np.random.rand(40, 3); y = (X[:, 0] > 0.5).astype(int)\n"
        "ix.variability_study(LogisticRegression, data=(X, y), runs=2, seed=0,\n"
        "                     use_process_isolation=True, verbose=False)\n"
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=180)
    assert "__name__ == '__main__'" in out.stderr
