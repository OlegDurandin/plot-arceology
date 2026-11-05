"""Join analysed episodes with show / episode metadata from datasets/ororo.sqlite.

shows.directory_name == folder name in datasets/Serials; episodes are keyed by
(show_id, season_number, episode_number).
Output: work/<lang>/episodes_meta.csv
Usage:  python src/meta.py EN|RU
"""
import json
import os
import sqlite3
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import ORORO_DB as DB

NONFICTION = {"Documentary", "Reality-TV", "Talk-Show", "Game-Show", "News", "Education"}


def load_meta():
    con = sqlite3.connect(f"file:{DB}?immutable=1", uri=True)
    shows = pd.read_sql(
        "select show_id, name, directory_name as series, imdb_id, rating, release_year, "
        "genres_json, countries_json, duration_minutes as show_duration, status from shows", con)
    eps = pd.read_sql("select show_id, season_number as season, episode_number as episode, "
                      "airdate from episodes", con)
    con.close()
    shows["genres"] = shows.genres_json.map(json.loads)
    shows["countries"] = shows.countries_json.map(json.loads)
    shows["fiction"] = shows.genres.map(lambda g: len(g) > 0 and not (set(g) & NONFICTION))
    return shows.drop(columns=["genres_json", "countries_json"]), eps


def main(lang):
    lang = lang.upper()
    shows, eps = load_meta()
    ep = pd.read_csv(os.path.join(ROOT, "work", lang, "episodes.csv"))
    df = ep.merge(shows, on="series", how="left")
    df = df.merge(eps, on=["show_id", "season", "episode"], how="left")
    df["air_year"] = pd.to_datetime(df.airdate, errors="coerce").dt.year
    df["genres"] = df.genres.map(lambda g: "|".join(g) if isinstance(g, list) else "")
    df["countries"] = df.countries.map(lambda g: "|".join(g) if isinstance(g, list) else "")
    df.to_csv(os.path.join(ROOT, "work", lang, "episodes_meta.csv"), index=False)
    print(f"{lang}: {len(df)} episodes, show matched {df.show_id.notna().mean():.2%}, "
          f"airdate {df.air_year.notna().mean():.2%}, fiction {df.fiction.mean():.2%}")


if __name__ == "__main__":
    main(sys.argv[1])
