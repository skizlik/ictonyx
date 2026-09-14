"""Module-level builders and estimators for tests.

Everything here must be importable by name so process-isolation (spawn) and
joblib workers can pickle it. Test files import from this module; nothing here
imports from test files.
"""

import numpy as np  # noqa: F401  (used by builders added in later commits)
