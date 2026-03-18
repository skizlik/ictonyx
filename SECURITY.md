# Security

## Pickle-based serialization

ictonyx uses Python's `pickle` module in:

- `ScikitLearnModelWrapper.save_model()` / `load_model()`
- `save_object()` / `load_object()` in `ictonyx.utils`

**Only load files from trusted sources.** Pickle files can execute
arbitrary code on deserialization. Do not load `.pkl` files downloaded
from the internet or received from unknown parties.

## PyTorch checkpoints

`PyTorchModelWrapper.load_model()` uses `torch.load(weights_only=True)`
by default. This is safe for checkpoints written by ictonyx's own
`save_model()`. Pass `weights_only=False` only for legacy checkpoints
from trusted sources.

## Reporting a vulnerability

Open a GitHub issue with the label `security`. For sensitive reports,
contact the maintainer directly rather than disclosing publicly.
