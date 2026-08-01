"""Data engine: live capture, book reconstruction, binary storage, replay.

This package owns everything between "bytes off the exchange socket" and
"a deterministic stream of book states an ML pipeline can consume". Nothing
in here imports from `ml/` — the dependency arrow points one way only, so
the storage format can be tested and benchmarked without torch installed.
"""
