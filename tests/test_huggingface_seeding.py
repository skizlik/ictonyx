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
