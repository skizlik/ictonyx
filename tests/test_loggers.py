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

    def test_verbose_printing(self, capsys):
        """Test that verbose mode prints output."""
        logger = BaseLogger(verbose=True, print_params=True, print_metrics=True)
        logger.log_params({"lr": 0.01})
        logger.log_metric("loss", 0.5, step=1)
        captured = capsys.readouterr()
        assert "lr" in captured.out
        assert "loss" in captured.out

    def test_quiet_mode(self, capsys):
        """Test that non-verbose mode suppresses print output."""
        logger = BaseLogger(verbose=False)
        logger.log_params({"lr": 0.01})
        logger.log_metric("loss", 0.5)
        logger.log_artifact("/path")
        logger.end_run()
        captured = capsys.readouterr()
        assert captured.out == ""
