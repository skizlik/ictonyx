"""v0.4.10 C14: model instances are seeded per run like classes are (v12 0.9)."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import ictonyx as ix

KW = dict(runs=5, seed=3, verbose=False, val_split=0.3)


def test_instance_with_fixed_random_state_varies_per_run(X, y):
    inst = RandomForestClassifier(n_estimators=10, random_state=0, max_features=1)
    with pytest.warns(UserWarning, match="overridden with the per-run child seed"):
        r = ix.variability_study(inst, data=(X, y), **KW)
    assert len(set(r.get_metric_values("val_accuracy"))) >= 2


def test_instance_and_class_paths_identical_under_one_seed(X, y):
    r_cls = ix.variability_study(
        RandomForestClassifier, n_estimators=10, max_features=1, data=(X, y), **KW
    )
    r_inst = ix.variability_study(
        RandomForestClassifier(n_estimators=10, max_features=1), data=(X, y), **KW
    )
    assert r_cls.run_seeds == r_inst.run_seeds
    assert r_cls.get_metric_values("val_accuracy") == r_inst.get_metric_values("val_accuracy")


def test_instance_without_random_state_warns_once(X, y, recwarn):
    ix.variability_study(
        LinearRegression(), data=(X, y.astype(float)), runs=3, seed=0, verbose=False
    )
    msgs = [str(w.message) for w in recwarn if "no random_state parameter" in str(w.message)]
    assert len(msgs) == 1


def test_readme_example_instances_are_seeded(X, y):
    """The README's headline example passes instances; on 0.4.9 it was unseeded."""
    from sklearn.datasets import make_classification
    from sklearn.neural_network import MLPClassifier

    # Noisy labels keep accuracy off its ceiling, so run-to-run variation from
    # seeding is visible whatever the split (it saturated at 1.0 on the
    # stratified split, promise ledger #13).
    Xn, yn = make_classification(
        n_samples=300, n_features=8, n_informative=3, flip_y=0.2, random_state=0
    )
    r = ix.compare_models(
        [
            MLPClassifier(hidden_layer_sizes=(8,), max_iter=50),
            RandomForestClassifier(n_estimators=10, max_features=1),
        ],
        data=(Xn, yn),
        runs=6,
        seed=1,
        verbose=False,
    )
    for name, series in r.raw_data.items():
        assert len(set(series.round(6))) >= 2, f"{name} produced identical runs"
