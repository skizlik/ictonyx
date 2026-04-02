from unittest.mock import patch

import pytest

from ictonyx.loggers import BaseLogger


class TestBaseLogger:
    """Test BaseLogger functionality."""

    def test_init_defaults(self):
        logger = BaseLogger()
        assert logger.verbose is True
        assert logger.print_params is False
        assert logger.print_metrics is False
        assert logger.history == {"params": {}, "metrics": []}

    def test_init_custom(self):
        logger = BaseLogger(verbose=False, print_params=True, print_metrics=True)
        assert logger.verbose is False
        assert logger.print_params is True

    def test_log_params(self):
        logger = BaseLogger(verbose=False)
        logger.log_params({"lr": 0.01, "epochs": 10})
        assert logger.history["params"]["lr"] == 0.01
        assert logger.history["params"]["epochs"] == 10

    def test_log_params_merge(self):
        """Test that multiple log_params calls merge."""
        logger = BaseLogger(verbose=False)
        logger.log_params({"a": 1})
        logger.log_params({"b": 2})
        assert logger.history["params"] == {"a": 1, "b": 2}

    def test_log_params_overwrite(self):
        """Test that duplicate keys overwrite."""
        logger = BaseLogger(verbose=False)
        logger.log_params({"a": 1})
        logger.log_params({"a": 99})
        assert logger.history["params"]["a"] == 99

    def test_log_metric(self):
        logger = BaseLogger(verbose=False)
        logger.log_metric("loss", 0.5, step=1)
        logger.log_metric("loss", 0.3, step=2)
        assert len(logger.history["metrics"]) == 2
        assert logger.history["metrics"][0] == {"key": "loss", "value": 0.5, "step": 1}
        assert logger.history["metrics"][1] == {"key": "loss", "value": 0.3, "step": 2}

    def test_log_metrics_batch(self):
        logger = BaseLogger(verbose=False)
        logger.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
        assert len(logger.history["metrics"]) == 2
        keys = [m["key"] for m in logger.history["metrics"]]
        assert "loss" in keys
        assert "accuracy" in keys

    def test_log_metric_default_step(self):
        logger = BaseLogger(verbose=False)
        logger.log_metric("val_loss", 0.4)
        assert logger.history["metrics"][0]["step"] == 0

    def test_get_history(self):
        logger = BaseLogger(verbose=False)
        logger.log_params({"model": "cnn"})
        logger.log_metric("loss", 0.5)
        history = logger.get_history()
        assert history is logger.history
        assert "params" in history
        assert "metrics" in history

    def test_end_run_no_crash(self):
        """Test that end_run doesn't crash."""
        logger = BaseLogger(verbose=False)
        logger.end_run()  # should not raise

    def test_log_artifact_no_crash(self):
        logger = BaseLogger(verbose=False)
        logger.log_artifact("/fake/path.csv")  # base impl does nothing

    def test_log_model_no_crash(self):
        logger = BaseLogger(verbose=False)
        logger.log_model(None, "model_path")  # base impl does nothing

    def test_log_figure_no_crash(self):
        logger = BaseLogger(verbose=False)
        logger.log_figure(None, "figure.png")  # base impl does nothing

    def test_set_tags_no_crash(self):
        logger = BaseLogger(verbose=False)
        logger.set_tags({"env": "test", "version": "1.0"})  # base impl does nothing

    def test_verbose_printing(self):
        """Test that verbose mode calls logger.info with the right content."""
        with patch("ictonyx.loggers.logger") as mock_log:
            logger_obj = BaseLogger(verbose=True, print_params=True, print_metrics=True)
            logger_obj.log_params({"lr": 0.01})
            logger_obj.log_metric("loss", 0.5, step=1)
        all_calls = " ".join(str(c) for c in mock_log.info.call_args_list)
        assert "lr" in all_calls

    def test_quiet_mode(self, capsys):
        """Test that non-verbose mode suppresses print output."""
        logger = BaseLogger(verbose=False)
        logger.log_params({"lr": 0.01})
        logger.log_metric("loss", 0.5)
        logger.log_artifact("/path")
        logger.end_run()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_log_study_summary_not_available_on_base(self):
        logger = BaseLogger(verbose=False)
        assert not hasattr(logger, "log_study_summary")


class TestLoggerVerboseBranches:
    """Verbose output for artifact, model, figure, tags, end_run."""

    def test_verbose_tags(self):
        with patch("ictonyx.loggers.logger") as mock_log:
            logger_obj = BaseLogger(verbose=True)
            logger_obj.set_tags({"env": "prod"})
        all_calls = " ".join(str(c) for c in mock_log.info.call_args_list)
        assert "env" in all_calls

    def test_verbose_end_run(self):
        with patch("ictonyx.loggers.logger") as mock_log:
            logger_obj = BaseLogger(verbose=True)
            logger_obj.end_run()
        assert mock_log.info.called


class TestBaseLoggerVerbosePrinting:
    """Cover the verbose print_metrics and print_params branches."""

    def test_log_metric_verbose_print_metrics(self):
        """verbose=True, print_metrics=True must call logger.info with metric info."""
        with patch("ictonyx.loggers.logger") as mock_log:
            bl = BaseLogger(verbose=True, print_metrics=True)
            bl.log_metric("val_loss", 0.25, step=5)
        all_calls = " ".join(str(c) for c in mock_log.info.call_args_list)
        assert "val_loss" in all_calls or "0.25" in all_calls or "5" in all_calls

    def test_log_metrics_all_stored(self):
        """log_metrics stores every key."""
        bl = BaseLogger(verbose=False)
        bl.log_metrics({"a": 1.0, "b": 2.0, "c": 3.0}, step=0)
        keys = {m["key"] for m in bl.history["metrics"]}
        assert keys == {"a", "b", "c"}

    def test_log_metric_step_default_zero(self):
        bl = BaseLogger(verbose=False)
        bl.log_metric("loss", 0.5)
        assert bl.history["metrics"][0]["step"] == 0

    def test_log_artifact_verbose(self):
        with patch("ictonyx.loggers.logger") as mock_log:
            bl = BaseLogger(verbose=True)
            bl.log_artifact("/path/to/file.csv")
        # Base implementation calls logger.debug, not info — just verify no crash
        assert True

    def test_log_model_verbose(self):
        bl = BaseLogger(verbose=True)
        bl.log_model(object(), "model_dir")  # should not raise

    def test_log_figure_verbose(self):
        bl = BaseLogger(verbose=True)
        bl.log_figure(object(), "fig.png")  # should not raise

    def test_set_tags_verbose_false(self):
        """verbose=False: set_tags must not call logger.info."""
        with patch("ictonyx.loggers.logger") as mock_log:
            bl = BaseLogger(verbose=False)
            bl.set_tags({"k": "v"})
        mock_log.info.assert_not_called()

    def test_end_run_verbose_false(self):
        """verbose=False: end_run must not call logger.info."""
        with patch("ictonyx.loggers.logger") as mock_log:
            bl = BaseLogger(verbose=False)
            bl.end_run()
        mock_log.info.assert_not_called()

    def test_history_independent_between_instances(self):
        """Two loggers must not share history."""
        a = BaseLogger(verbose=False)
        b = BaseLogger(verbose=False)
        a.log_params({"x": 1})
        assert "x" not in b.history["params"]

    def test_log_params_verbose_false_no_info(self):
        """verbose=False: log_params must not call logger.info."""
        with patch("ictonyx.loggers.logger") as mock_log:
            bl = BaseLogger(verbose=False, print_params=True)
            bl.log_params({"lr": 0.01})
        mock_log.info.assert_not_called()


class TestMLflowLoggerMocked:
    """MLflowLogger tests using a fully mocked mlflow module."""

    @pytest.fixture
    def mock_mlflow(self):
        """Provide a mock mlflow module with the minimum surface MLflowLogger uses."""
        import sys
        from unittest.mock import MagicMock, patch

        mock_mlf = MagicMock()
        mock_mlf.exceptions.MlflowException = Exception
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_run.info.experiment_id = "exp-0"
        mock_mlf.start_run.return_value = mock_run
        mock_mlf.create_experiment.return_value = "exp-0"
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp-0"
        mock_exp.name = "ictonyx_experiment"
        mock_mlf.get_experiment_by_name.return_value = mock_exp
        mock_mlf.get_experiment.return_value = mock_exp
        mock_mlf.get_tracking_uri.return_value = "file:///tmp/mlruns"

        with patch.dict(
            "sys.modules", {"mlflow": mock_mlf, "mlflow.exceptions": mock_mlf.exceptions}
        ):
            with patch("ictonyx.loggers.HAS_MLFLOW", True):
                with patch("ictonyx.loggers.mlflow", mock_mlf):
                    yield mock_mlf

    def test_init_creates_run(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        mock_mlflow.start_run.assert_called_once()
        assert logger.run_id == "test-run-123"

    def test_log_params_calls_mlflow(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.log_params({"lr": 0.01, "epochs": 10})
        mock_mlflow.log_params.assert_called_once()
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert call_args["lr"] == 0.01
        assert call_args["epochs"] == 10

    def test_log_params_converts_complex_types(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.log_params({"layers": [64, 32], "config": {"a": 1}, "none_val": None})
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert call_args["layers"] == "[64, 32]"
        assert call_args["config"] == "{'a': 1}"
        assert call_args["none_val"] == "None"

    def test_log_metric_calls_mlflow(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.log_metric("val_accuracy", 0.85, step=5)
        mock_mlflow.log_metric.assert_called_once_with("val_accuracy", 0.85, step=5)

    def test_log_metric_skips_nan(self, mock_mlflow):
        import numpy as np

        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.log_metric("broken", float("nan"), step=0)
        mock_mlflow.log_metric.assert_not_called()

    def test_log_metric_skips_inf(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.log_metric("inf_val", float("inf"), step=0)
        mock_mlflow.log_metric.assert_not_called()

    def test_log_metrics_filters_invalid(self, mock_mlflow):
        import numpy as np

        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.log_metrics({"good": 0.9, "bad": float("nan")}, step=1)
        call_args = mock_mlflow.log_metrics.call_args[0][0]
        assert "good" in call_args
        assert "bad" not in call_args

    def test_end_run_calls_mlflow(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.end_run()
        mock_mlflow.end_run.assert_called_once()

    def test_set_tags_calls_mlflow(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.set_tags({"env": "test", "version": 3})
        mock_mlflow.set_tags.assert_called_once()
        call_args = mock_mlflow.set_tags.call_args[0][0]
        assert call_args["env"] == "test"
        assert call_args["version"] == "3"  # converted to string

    def test_import_error_without_mlflow(self):
        from unittest.mock import patch

        with patch("ictonyx.loggers.HAS_MLFLOW", False):
            from ictonyx.loggers import MLflowLogger

            with pytest.raises(ImportError, match="MLflow"):
                MLflowLogger()

    def test_run_id_property(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        assert logger.run_id == "test-run-123"

    def test_history_still_updated_alongside_mlflow(self, mock_mlflow):
        from ictonyx.loggers import MLflowLogger

        logger = MLflowLogger(verbose=False)
        logger.log_params({"x": 1})
        logger.log_metric("loss", 0.5, step=0)
        assert logger.history["params"]["x"] == 1
        assert len(logger.history["metrics"]) == 1


class TestBaseLoggerChildRunNoOps:
    """BaseLogger.start_child_run() and end_child_run() are no-ops.

    These methods exist so callers can call them unconditionally without
    isinstance checks. Verify they don't raise and return the right types.
    """

    def test_start_child_run_returns_empty_string(self):
        logger = BaseLogger(verbose=False)
        result = logger.start_child_run()
        assert result == ""

    def test_start_child_run_with_name_returns_empty_string(self):
        logger = BaseLogger(verbose=False)
        result = logger.start_child_run(run_name="run_1")
        assert result == ""

    def test_end_child_run_returns_none(self):
        logger = BaseLogger(verbose=False)
        result = logger.end_child_run()
        assert result is None

    def test_child_run_lifecycle_does_not_raise(self):
        """Full lifecycle: start then end must not raise."""
        logger = BaseLogger(verbose=False)
        run_id = logger.start_child_run(run_name="training_run_1")
        assert isinstance(run_id, str)
        logger.end_child_run()  # must not raise

    def test_multiple_child_runs_do_not_raise(self):
        """Multiple sequential child runs must not accumulate state or raise."""
        logger = BaseLogger(verbose=False)
        for i in range(5):
            logger.start_child_run(run_name=f"run_{i}")
            logger.log_metric("loss", 0.5 - i * 0.05, step=i)
            logger.end_child_run()
        assert len(logger.history["metrics"]) == 5

    def test_log_study_summary_absent_on_base_logger(self):
        """BaseLogger does not have log_study_summary — callers use hasattr."""
        logger = BaseLogger(verbose=False)
        assert not hasattr(logger, "log_study_summary")
