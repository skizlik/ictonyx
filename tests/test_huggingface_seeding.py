"""Regression test for X-40: HuggingFaceModelWrapper must vary seed per run.

Background: in v0.4.4-v0.4.6, HuggingFaceModelWrapper.fit() read run_seed
from self.model_config, which is never populated by the runner. The
defensive fallback run_seed=42 fired every run, producing bit-for-bit
identical metrics across all runs in a variability study.

v0.4.7 threads run_seed through fit_kwargs. This test asserts that three
runs with distinct spawned seeds produce non-identical val_accuracy
values, catching any regression of the X-40 pattern.
"""

import pytest

# Skip the whole module if transformers isn't available
transformers = pytest.importorskip("transformers")


@pytest.mark.slow
def test_huggingface_variability_study_varies_seed_per_run():
    """X-40: Three runs must produce non-identical val_accuracy."""
    import ictonyx as ix
    from ictonyx import HuggingFaceModelWrapper

    # Tiny text-classification dataset. Small enough to run in a few
    # minutes on CPU or ~30s on GPU; large enough for the model to
    # actually learn something (so that seed variation produces measurable
    # metric variation).
    train_texts = [f"This is training text number {i} about topic {i % 4}" for i in range(60)]
    train_labels = [i % 4 for i in range(60)]
    val_texts = [f"Validation text {i} about topic {i % 4}" for i in range(40)]
    val_labels = [i % 4 for i in range(40)]

    def build_hf_tiny(config):
        return HuggingFaceModelWrapper(
            model_name_or_path="google/bert_uncased_L-2_H-128_A-2",
            num_labels=4,
        )

    results = ix.variability_study(
        model=build_hf_tiny,
        data=(train_texts, train_labels),
        validation_data=(val_texts, val_labels),
        runs=3,
        epochs=2,
        batch_size=16,
        learning_rate=5e-5,
        seed=2026,
        verbose=False,
    )

    # Collect val_accuracy across runs. If X-40 has regressed, all three
    # values are identical.
    val_accs = results.get_metric_values("val_accuracy")
    assert len(val_accs) == 3, f"Expected 3 runs, got {len(val_accs)} metric values: {val_accs}"

    # Round to 4 decimal places before uniqueness check — this is looser
    # than the bit-exact equality that X-40 produced but strict enough
    # that genuine per-run variance will pass. With the bug, runs produced
    # identical metrics to 6+ decimals; with varying seeds, runs typically
    # differ at the 2nd or 3rd decimal place.
    unique_accs = {round(v, 4) for v in val_accs}
    assert len(unique_accs) > 1, (
        f"X-40 regression: all 3 runs produced val_accuracy={val_accs[0]:.6f}. "
        f"Seeds are not varying per run. Values: {val_accs}"
    )


class TestHuggingFaceVerboseHandling:
    """Regression test for X-39: HuggingFaceModelWrapper must respect
    verbose=False.

    Pre-v0.4.7: variability_study(verbose=False, model=HuggingFaceModelWrapper)
    still produced 50+ lines of stdout per run as the Trainer logged
    mid-epoch loss/grad_norm/learning_rate every logging_steps. In a
    20-run variability study this was 1000+ lines of noise.

    Fix: HF wrapper's fit() reads verbose from kwargs (threaded from
    runner alongside run_seed) and translates to:
      - TrainingArguments(disable_tqdm=not verbose, log_level=...)
      - transformers.logging.set_verbosity_error() when verbose=False
    """

    @pytest.mark.slow
    def test_verbose_false_produces_quiet_output(self, capfd):
        """A 2-run verbose=False HF study must produce fewer than 30 lines
        of stdout+stderr. Previously 50+ lines per run (100+ for 2 runs)."""
        pytest.importorskip("transformers", reason="transformers not installed")
        import ictonyx as ix
        from ictonyx import HuggingFaceModelWrapper

        train_texts = [f"training text {i}" for i in range(20)]
        train_labels = [i % 2 for i in range(20)]
        val_texts = [f"val text {i}" for i in range(10)]
        val_labels = [i % 2 for i in range(10)]

        _ = ix.variability_study(
            model=HuggingFaceModelWrapper,
            model_kwargs={
                "model_name_or_path": "google/bert_uncased_L-2_H-128_A-2",
                "num_labels": 2,
            },
            data=(train_texts, train_labels),
            validation_data=(val_texts, val_labels),
            runs=2,
            epochs=1,
            batch_size=8,
            learning_rate=5e-5,
            seed=2026,
            verbose=False,
        )

        out, err = capfd.readouterr()
        combined = out + err
        # Filter out HF Hub chatter, warnings, deprecation notices — count
        # only genuinely informational lines. The regression target is
        # eliminating the per-step Trainer loss-log stream.
        nontrivial_lines = [
            line
            for line in combined.split("\n")
            if line.strip() and not line.startswith(("Warning:", "W ", "I ", "E "))
        ]
        assert len(nontrivial_lines) < 30, (
            f"Expected fewer than 30 nontrivial stdout lines with verbose=False, "
            f"got {len(nontrivial_lines)}. Output:\n{combined}"
        )
