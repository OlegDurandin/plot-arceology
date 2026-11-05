#!/usr/bin/env bash
# Full pipeline. Requires datasets/ (see README.md). Run from the repository root.
# Approximate wall time on an 18-core machine: ~3 h (topic inference dominates).
set -euo pipefail
PY=${PY:-python}
cd "$(dirname "$0")/.."

# 1. TV subtitles: preprocessing, 127-topic LDA, figures of the paper
for L in EN RU; do
  $PY src/prep.py "$L"
  $PY src/topics.py "$L"
  $PY src/analyze.py "$L"
done

# 2. EN <-> RU comparison on parallel chunks
(cd src && $PY compare.py)

# 3. Genres and decades (metadata from datasets/ororo.sqlite)
for L in EN RU; do
  (cd src && $PY meta.py "$L" && $PY genres.py "$L")
done

# 4. Prose: fairy tales and proza.ru stories
$PY src/prep_text.py TALES
$PY src/prep_text.py PROZA
#   4a. projection onto the Russian TV model
$PY src/project.py TALES
$PY src/project.py PROZA
$PY src/analyze.py PROJ_TALES
$PY src/analyze.py PROJ_PROZA
(cd src && $PY compare_prose.py PROJ)
#   4b. joint model TV + tales + prose
(cd src && $PY joint.py)
for S in TV TALES PROZA; do $PY src/analyze.py "JOINT_$S"; done
(cd src && $PY compare_prose.py JOINT)
