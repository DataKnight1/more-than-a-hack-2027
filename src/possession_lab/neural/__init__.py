"""Tracking-first neural possession models trained only on eligible external matches.

Sub-modules:

- ``canonical``: provider-neutral per-frame match representation (metres, 25 Hz).
- ``idsse``: DFL / IDSSE open-data converter (CC BY 4.0, other games).
- ``atd``: local-only reader for the challenge ATD (inference input; never a training source).
- ``augment``: domain randomisation that degrades clean tracking towards automatic tracking.
- ``baselines``: nearest-player + majority-vote reference on canonical data.
- ``external_eval``: wrapper around the *unchanged* official metric for external holdouts.
"""
