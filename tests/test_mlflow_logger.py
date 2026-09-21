"""v0.4.10 C20: MLflowLogger works without TensorFlow and survives a multi-run study."""

from importlib.util import find_spec

import pytest
from sklearn.linear_model import LogisticRegression

import ictonyx as ix

pytestmark = pytest.mark.skipif(find_spec("mlflow") is None, reason="mlflow not installed")


@pytest.fixture
def mlflow_store(tmp_path):
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # the file store is deprecated in mlflow 3
    mlflow.set_tracking_uri(uri)
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()


def test_mlflow_logger_constructs_without_tensorflow(mlflow_store):
    from ictonyx.loggers import HAS_MLFLOW, MLflowLogger

    assert HAS_MLFLOW is True
    assert MLflowLogger(experiment_name="t", verbose=False) is not None


def test_run_not_opened_until_first_log(mlflow_store):
    import mlflow

    from ictonyx.loggers import MLflowLogger

    lg = MLflowLogger(experiment_name="t", verbose=False)
    assert mlflow.active_run() is None
    lg.log_metric("x", 1.0)
    assert mlflow.active_run() is not None
    lg.end_run()
    assert mlflow.active_run() is None


def test_multi_run_study_completes_and_closes(mlflow_store, X, y):
    import mlflow

    from ictonyx.loggers import MLflowLogger

    lg = MLflowLogger(experiment_name="t", verbose=False)
    r = ix.variability_study(
        LogisticRegression, data=(X, y), runs=3, seed=0, verbose=False, tracker=lg
    )
    assert r.n_runs == 3
    assert mlflow.active_run() is None
    run = mlflow.get_run(lg.run_id)
    assert run is not None
    assert run.data.params["num_runs"] == "3"
    assert run.data.tags.get("mode") == "standard"
    hist = mlflow.MlflowClient().get_metric_history(run.info.run_id, "run_id")
    assert sorted(m.value for m in hist) == [1.0, 2.0, 3.0]


def test_end_run_is_idempotent(mlflow_store):
    from ictonyx.loggers import MLflowLogger

    lg = MLflowLogger(experiment_name="t", verbose=False)
    lg.end_run()  # never started
    lg.log_metric("x", 1.0)
    lg.end_run()
    lg.end_run()  # already ended
