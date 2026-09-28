"""pytest plugin: run the test suite on shuffled copies of sklearn's toy datasets.

Many tests use small real datasets (iris, wine, breast cancer, digits). A test
whose assertion holds only for the particular train/validation/test split it
happens to get is testing luck, and breaks when splitting or seeding changes
for reasons unrelated to what it tests. This plugin shuffles
the rows that ``load_iris``, ``load_wine``, ``load_breast_cancer`` and
``load_digits`` return. The data are the same as a set, so every study sees a
different split under its own seed. Run it after any change to splitting or
seeding, with two or three permutations:

    PYTHONPATH=scripts SPLIT_AUDIT=1 pytest -p split_audit -m "not slow"
    PYTHONPATH=scripts SPLIT_AUDIT=2 pytest -p split_audit -m "not slow"

(On Windows: ``set PYTHONPATH=scripts`` and ``set SPLIT_AUDIT=1``.) A test that
passes normally but fails here depends on its split: build a fixture for the
property it tests instead, for example noisy synthetic data where variation is
asserted, or a model with no randomness where determinism is asserted.

``SPLIT_AUDIT`` unset or 0 leaves the loaders unchanged.
"""

import os

import numpy as np

LOADERS = ("load_iris", "load_wine", "load_breast_cancer", "load_digits")


def _rows(a, p):
    return a.iloc[p] if hasattr(a, "iloc") else a[p]


def _shuffled(loader, k):
    def load(*args, **kwargs):
        out = loader(*args, **kwargs)
        if isinstance(out, tuple):  # return_X_y=True
            p = np.random.default_rng(k).permutation(len(out[1]))
            return (_rows(out[0], p), _rows(out[1], p)) + tuple(out[2:])
        p = np.random.default_rng(k).permutation(len(out.target))
        out.data, out.target = _rows(out.data, p), _rows(out.target, p)
        if getattr(out, "frame", None) is not None:
            out.frame = out.frame.iloc[p]
        return out

    return load


def pytest_configure(config):
    # Runs before collection, so module-level "from sklearn.datasets import
    # load_wine" in a test file already gets the shuffled loader.
    k = int(os.environ.get("SPLIT_AUDIT", "0") or 0)
    if not k:
        return
    import sklearn.datasets as datasets

    for name in LOADERS:
        setattr(datasets, name, _shuffled(getattr(datasets, name), k))


def pytest_report_header(config):
    k = int(os.environ.get("SPLIT_AUDIT", "0") or 0)
    return f"split audit: toy datasets shuffled with permutation {k}" if k else None
