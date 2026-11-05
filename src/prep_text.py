"""Prose pipeline, stage 1: load tales / proza.ru stories, clean, chunk by words, lemmatise.

Generalisation of the paper's chunking to written text (REPORT.md, section 7):
  narrative time = share of words; a text is split into 3 parts, each recursively halved
  until a chunk has 150-300 words (=> 3 * 2^k chunks); boundaries snap to sentence ends.

Usage:  python src/prep_text.py TALES|PROZA
Output: work/<SRC>/raw/*.jsonl.gz   {"eid","i","n","t": lemmas (names marked \\x00), "w": words, "d": dialogue share}
        work/<SRC>/episodes.csv     text metadata (eid, series=group key, ...)
        work/<SRC>/vocab_stats.pkl  cap / chunk_df / text_df / tok_total for vocabulary building
"""
import collections
import gzip
import hashlib
import json
import os
import pickle
import re
import sqlite3
import sys
import urllib.parse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from prep import tokenize_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import DATASETS as DATA
MIN_W, MAX_W = 150, 300           # chunk size in words
PER_AUTHOR = 20                   # proza.ru: max stories per author
WORD_RE = re.compile(r"\w+(?:[-']\w+)*")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"—–A-ZА-ЯЁ0-9])")
DIALOGUE = ("—", "–", "-", "«")
BOILER = re.compile(r"^\s*(фото|иллюстрац|рисунок|картинк|изображени|источник|©|copyright|"
                    r"продолжение след|конец|окончание след|начало (?:тут|здесь)|"
                    r"\(?продолжение\)?$|часть \d+$|глава \d+$|\*+$)", re.I)


def connect(name):
    p = os.path.join(DATA, name)
    return sqlite3.connect("file:" + urllib.parse.quote(p) + "?immutable=1", uri=True)


FIC_DIR = os.path.join(DATA, "data_ficbook")
FIC_DB = "ficbook_originals_2026.sqlite"
FIC_DROP = re.compile(r"https?://|www\.|^(Примечани[ея]|Примечание автора|Публичная бета|Посвящение|"
                      r"Описание|Начать читать|Содержание|Отзывы: ?\d*|\|)\s*:?\s*$|"
                      r"^\d{1,2} [а-я]+ \d{4} г\.(, \d{1,2}:\d{2})?$", re.I)


def ficbook_db():
    """Copy the live DB (+ its WAL, which immutable mode would ignore) to work/ and open the copy."""
    import shutil
    dst = os.path.join(ROOT, "work", "FICBOOK_db")
    os.makedirs(dst, exist_ok=True)
    for suffix in ("", "-wal"):
        src_f = os.path.join(FIC_DIR, FIC_DB + suffix)
        if os.path.exists(src_f) and os.path.getsize(src_f) > 0:
            shutil.copy(src_f, os.path.join(dst, FIC_DB + suffix))
    return sqlite3.connect(os.path.join(dst, FIC_DB))


def ficbook_text(paths, titles):
    """Concatenate chapter files; drop table-of-contents pages, author notes, links, dates."""
    out = []
    for p in paths:
        lines = open(os.path.join(DATA, p), encoding="utf-8").read().split("\n")
        head = [l.strip() for l in lines[:6]]
        if "Содержание" in head or "Начать читать" in head:
            continue                                        # table-of-contents page
        keep = [l for l in lines if l.strip() and not FIC_DROP.search(l.strip())
                and l.strip() not in titles]
        out.extend(keep)
    return "\n".join(out)


def load(src):
    if src == "FICBOOK":
        con = ficbook_db()
        w = pd.read_sql("select work_id, title, author_url, directions_json, tags_json, fandoms_json, "
                        "rating, status, published_year, published_month from works", con)
        ct = pd.read_sql("select work_id, chapter_number, title, path, status from chapter_texts", con)
        bad = set(ct[ct.status.isin(["error", "empty"])].work_id)
        gaps = ct.groupby("work_id").chapter_number.agg(lambda x: x.max() - len(x) > 0)
        ok = ct[ct.status == "complete"].sort_values(["work_id", "chapter_number"])
        chaps = ok.groupby("work_id").agg(paths=("path", list), ctitles=("title", list))
        w = w.merge(chaps, left_on="work_id", right_index=True)
        n0 = len(w)
        w = w[(w.status == "finished") & ~w.work_id.isin(bad) & ~w.work_id.map(gaps).fillna(True)]
        n1 = len(w)
        tags = w.tags_json.map(lambda j: set(json.loads(j)))
        dirs = w.directions_json.map(lambda j: set(json.loads(j)))
        w = w[~tags.map(lambda t: "Стихи" in t) & ~dirs.map(lambda d: "Статья" in d)]
        n2 = len(w)
        w["text"] = [ficbook_text(ps, set(ts) | {t}) for ps, ts, t in zip(w.paths, w.ctitles, w.title)]
        w["n_words"] = w.text.map(lambda x: len(WORD_RE.findall(x)))
        med_line = w.text.map(lambda x: np.median([len(l.split()) for l in x.split("\n") if l.strip()] or [0]))
        w = w[(w.n_words >= 1000) & (w.n_words <= 10000) & (med_line >= 8)]
        n3 = len(w)
        w["bucket"] = w.work_id.map(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 10000)
        w = w.sort_values("bucket").groupby("author_url").head(PER_AUTHOR)
        print(f"FICBOOK: {n0} works with text -> {n1} finished & complete -> {n2} without verse/articles "
              f"-> {n3} with 1000-10000 words -> {len(w)} after cap {PER_AUTHOR}/author", flush=True)
        w["eid"] = w.work_id
        w["series"] = w.author_url
        w["direction"] = w.directions_json.map(lambda j: "|".join(json.loads(j)))
        w["tags"] = w.tags_json.map(lambda j: "|".join(json.loads(j)))
        w["original"] = w.fandoms_json.map(lambda j: json.loads(j) == ["Ориджиналы"])
        w["para_sep"] = "\n"
        return w[["eid", "series", "title", "direction", "tags", "original", "rating",
                  "published_year", "published_month", "n_words", "bucket", "text", "para_sep"]]
    if src == "TALES":
        con = connect("fairytalez[Rus].sqlite")
        df = pd.read_sql("select tale_id, title, category, author, readability, n_words, text, "
                         "moral is not null as has_moral from tales where n_words >= 600", con)
        reg = pd.read_sql("select tale_id, group_concat(region_name, '|') regions "
                          "from tale_regions group by tale_id", con)
        df = df.merge(reg, on="tale_id", how="left")
        df["eid"] = "tale" + df.tale_id.astype(str)
        df["series"] = df.category.fillna("?")
        df["para_sep"] = "\n\n"
        return df.drop(columns=["tale_id"])
    frames = []
    for name in ["proza[2022-2023].sqlite", "proza[2024-2025].sqlite"]:
        frames.append(pd.read_sql(
            "select story_id, title, author_login, day, n_words, bucket, text from stories "
            "where status = 200 and n_words between 1000 and 10000", connect(name)))
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df["h"] = df.text.map(lambda x: hashlib.md5(re.sub(r"\W+", "", x.lower()).encode()).hexdigest())
    df = df.drop_duplicates("h")
    n1 = len(df)
    med_line = df.text.map(lambda x: np.median([len(l.split()) for l in x.split("\n") if l.strip()] or [0]))
    df = df[med_line >= 8]                                  # drop verse
    n2 = len(df)
    df = df.sort_values("bucket").groupby("author_login").head(PER_AUTHOR)
    print(f"PROZA: {n0} stories 1000-10000 words -> {n1} after dedup -> {n2} after verse filter "
          f"-> {len(df)} after cap {PER_AUTHOR}/author", flush=True)
    df["eid"] = df.story_id
    df["series"] = df.author_login
    df["year"] = df.day.str[:4].astype(int)
    df["para_sep"] = "\n"
    return df.drop(columns=["h"])


def chunk_count(nw):
    n, length = 3, nw / 3
    while length > MAX_W:
        length /= 2
        n *= 2
    return n


def sentences(text, sep):
    """-> list of (sentence, is_dialogue, n_words)"""
    out = []
    for para in text.split(sep):
        para = para.strip()
        if not para or BOILER.search(para) and len(para) < 80:
            continue
        dlg = para.startswith(DIALOGUE)
        for s in SENT_SPLIT.split(para.replace("\n", " ")):
            nw = len(WORD_RE.findall(s))
            if nw:
                out.append((s, dlg, nw))
    return out


def process(args):
    shard, rows, out_path = args
    cap = collections.defaultdict(lambda: [0, 0])
    meta = []
    with gzip.open(out_path, "wt", encoding="utf-8") as fo:
        for r in rows:
            sents = sentences(r["text"], r["para_sep"])
            nw = sum(s[2] for s in sents)
            if nw < 3 * MIN_W:
                continue
            n = chunk_count(nw)
            cum = np.cumsum([s[2] for s in sents])
            # chunk boundaries at k/n of the words, snapped to the nearest sentence end
            cuts = [0]
            for k in range(1, n):
                j = int(np.argmin(np.abs(cum - nw * k / n))) + 1
                cuts.append(max(j, cuts[-1] + 1))
            cuts.append(len(sents))
            if any(b <= a for a, b in zip(cuts[:-1], cuts[1:])):
                continue                                    # too few sentences
            dl_total = 0
            for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                part = sents[a:b]
                toks = tokenize_lines([s[0] for s in part], "RU", cap)
                w = sum(s[2] for s in part)
                d = sum(s[2] for s in part if s[1]) / max(w, 1)
                dl_total += d * w
                fo.write(json.dumps({"eid": r["eid"], "i": i, "n": n, "t": toks, "w": w,
                                     "d": round(d, 3)}, ensure_ascii=False) + "\n")
            m = {k: v for k, v in r.items() if k not in ("text", "para_sep")}
            m.update(n_chunks=n, n_words_clean=nw, dialogue=round(dl_total / nw, 3))
            meta.append(m)
    return meta, dict(cap)


def main(src):
    src = src.upper()
    work = os.path.join(ROOT, "work", src)
    raw = os.path.join(work, "raw")
    os.makedirs(raw, exist_ok=True)
    df = load(src)
    recs = df.to_dict("records")
    nshard = 64
    tasks = [(k, recs[k::nshard], os.path.join(raw, f"{k:05d}.jsonl.gz")) for k in range(nshard)]
    cap = collections.defaultdict(lambda: [0, 0])
    meta = []
    with Pool(os.cpu_count()) as pool:
        for m, cs in pool.imap_unordered(process, tasks):
            meta.extend(m)
            for w, (a, b) in cs.items():
                cap[w][0] += a
                cap[w][1] += b
    ep = pd.DataFrame(meta)
    ep.to_csv(os.path.join(work, "episodes.csv"), index=False)
    print(f"{src}: {len(ep)} texts, {ep.n_chunks.sum()} chunks, "
          f"median {ep.n_chunks.median():.0f} chunks/text, dialogue share median {ep.dialogue.median():.2f}",
          flush=True)

    chunk_df, text_df, tok = collections.Counter(), collections.Counter(), collections.Counter()
    n_chunks = 0
    for t in tasks:
        cur, seen = None, set()
        with gzip.open(t[2], "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                n_chunks += 1
                s = set(d["t"])
                chunk_df.update(s)
                tok.update(d["t"])
                if d["eid"] != cur:
                    text_df.update(seen)
                    cur, seen = d["eid"], set()
                seen |= s
        text_df.update(seen)
    with open(os.path.join(work, "vocab_stats.pkl"), "wb") as f:
        pickle.dump(dict(cap=dict(cap), chunk_df=chunk_df, text_df=text_df, tok_total=tok,
                         n_chunks=n_chunks, n_texts=len(ep)), f)
    print("done", n_chunks, "chunks", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
