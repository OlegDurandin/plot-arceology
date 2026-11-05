"""Locations of the input data.

The Ororo export (subtitles + metadata) may be unpacked either directly into datasets/
(datasets/Serials, datasets/ororo.sqlite) or as datasets/data_ororo/ (layout of the final
archive). Override with the ORORO_DIR environment variable.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "datasets")


def _ororo_dir():
    env = os.environ.get("ORORO_DIR")
    if env:
        return env
    for cand in (os.path.join(DATASETS, "data_ororo"), DATASETS):
        if os.path.isdir(os.path.join(cand, "Serials")):
            return cand
    return DATASETS


ORORO_DIR = _ororo_dir()
SERIALS = os.path.join(ORORO_DIR, "Serials")
ORORO_DB = os.path.join(ORORO_DIR, "ororo.sqlite")
