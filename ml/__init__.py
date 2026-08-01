"""Machine learning: features, labels, model, training, distillation.

This package consumes book states from `data_engine` and produces a trained,
compressed model plus honest evaluation numbers. It never talks to a socket
and never writes to the tape — if something here needs new data, the fix
belongs in `data_engine`, not in a side channel.
"""
