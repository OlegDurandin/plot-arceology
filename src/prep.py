"""Stage 1: parse .srt subtitles, chunk each episode, tokenize.

Reproduces Schmidt, "Plot Arceology" (2015), section II:
  * each script is split into 3 parts, then every part is recursively halved
    until the chunk length is between 2 and 4 minutes (=> 3 * 2^k chunks);
  * every chunk is a "document" for the topic model.

Usage:  python src/prep.py EN|RU
Output: work/<lang>/chunks.jsonl.gz   (one line per chunk, tokens already filtered)
        work/<lang>/episodes.csv       (episode metadata)
"""
import collections
import gzip
import json
import os
import re
import sys
from multiprocessing import Pool

import pandas as pd

from stopwords import STOP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "Serials")

MIN_DUR, MAX_DUR = 15.0, 95.0   # minutes: TV episodes only
MIN_TOKENS = 400                # raw word tokens per episode

TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
TAG_RE = re.compile(r"<[^>]*>|\{[^}]*\}")
HI_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪[^♪]*♪?|#[^#]*#?")  # sound descriptions, songs
CREDIT_RE = re.compile(r"www\.|\.com|\.org|\.net|addic7ed|opensubtitles|subtitle|sync|"
                       r"corrected by|ripped by|transcript|субтитр|перевод|синхрон|"
                       r"переведено|релиз|notabenoid|tvsubs|кураж|lostfilm|newstudio", re.I)
EP_RE = re.compile(r"_s(\d+)_e(\d+)_(EN|RU)\.srt$", re.I)

WORD = {
    "EN": re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*"),
    "RU": re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*"),
}
SENT_END = re.compile(r"[.!?…:\"«»—–-]\s*$")

_morph = None
_lemma_cache = {}


def lemma_ru(w):
    """pymorphy3 normal form; returns (lemma, is_person_name)."""
    global _morph
    r = _lemma_cache.get(w)
    if r is None:
        if _morph is None:
            import pymorphy3
            _morph = pymorphy3.MorphAnalyzer()
        p = _morph.parse(w)[0]
        tag = p.tag
        r = (p.normal_form.replace("ё", "е"),
             ("Name" in tag) or ("Surn" in tag) or ("Patr" in tag))
        _lemma_cache[w] = r
    return r


def to_min(h, m, s, ms):
    return int(h) * 60 + int(m) + (int(s) + int(ms[:3].ljust(3, "0")) / 1000) / 60


def parse_srt(path):
    """-> list of (mid_time_minutes, [lines])"""
    raw = open(path, "rb").read()
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = raw.decode("cp1251" if path.endswith("_RU.srt") else "cp1252", "replace")
    txt = txt.replace("\r", "")
    blocks = []
    for blk in re.split(r"\n\s*\n", txt):
        m = TIME_RE.search(blk)
        if not m:
            continue
        g = m.groups()
        t0, t1 = to_min(*g[:4]), to_min(*g[4:])
        body = blk[m.end():]
        body = TAG_RE.sub(" ", body)
        body = HI_RE.sub(" ", body)
        lines = [l.strip() for l in body.split("\n") if l.strip() and not CREDIT_RE.search(l)]
        if lines:
            blocks.append(((t0 + t1) / 2, t1, lines))
    return blocks


def tokenize_lines(lines, lang, cap_stats):
    """Lower-cased (EN) / lemmatised (RU) tokens. Updates cap_stats[token] = [cap_mid, mid]
    so that proper nouns (character names) can be detected corpus-wide."""
    out = []
    rx = WORD[lang]
    for line in lines:
        initial = True
        for m in rx.finditer(line):
            w = m.group(0)
            if not initial:
                # look at the text just before the word: sentence boundary?
                pre = line[max(0, m.start() - 3):m.start()]
                if SENT_END.search(pre):
                    initial = True
            lw = w.lower().replace("ё", "е")
            name = False
            if lang == "RU":
                lw, name = lemma_ru(lw)
                if name:
                    lw = "\x00" + lw  # mark as person name, dropped later
            else:
                lw = lw.replace("'s", "") if lw.endswith("'s") else lw
            if not initial and len(w) > 1:
                st = cap_stats[lw]
                st[1] += 1
                if w[0].isupper():
                    st[0] += 1
            out.append(lw)
            initial = False
    return out


def chunk_count(dur):
    n, length = 3, dur / 3
    while length > 4.0:
        length /= 2
        n *= 2
    return n


def process_series(args):
    series, files, lang, out_path = args
    cap_stats = collections.defaultdict(lambda: [0, 0])
    episodes = []
    with gzip.open(out_path, "wt", encoding="utf-8") as fo:
        for path in files:
            m = EP_RE.search(path)
            if not m:
                continue
            season, ep = int(m.group(1)), int(m.group(2))
            try:
                blocks = parse_srt(path)
            except Exception:
                continue
            if len(blocks) < 50:
                continue
            dur = max(b[1] for b in blocks)
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue
            n = chunk_count(dur)
            chunks = [[] for _ in range(n)]
            for mid, _, lines in blocks:
                k = min(n - 1, max(0, int(mid / dur * n)))
                chunks[k].extend(tokenize_lines(lines, lang, cap_stats))
            ntok = sum(map(len, chunks))
            if ntok < MIN_TOKENS:
                continue
            eid = f"{series}|{season}|{ep}"
            episodes.append(dict(eid=eid, series=series, season=season, episode=ep,
                                 duration=round(dur, 2), n_chunks=n, n_raw_tokens=ntok))
            for i, toks in enumerate(chunks):
                fo.write(json.dumps({"eid": eid, "i": i, "n": n, "t": toks},
                                    ensure_ascii=False) + "\n")
    return series, episodes, dict(cap_stats)


def load_names(lang):
    """First-name list (NLTK names corpus) for English; Russian uses pymorphy3 tags."""
    if lang != "EN":
        return set()
    d = os.path.join(ROOT, "resources", "dl", "names")
    if not os.path.exists(os.path.join(d, "male.txt")):
        import io
        import urllib.request
        import zipfile
        url = ("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
               "packages/corpora/names.zip")
        print("downloading NLTK names corpus:", url, flush=True)
        data = urllib.request.urlopen(url, timeout=60).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.dirname(d))
    out = set()
    for fn in ("male.txt", "female.txt"):
        with open(os.path.join(d, fn)) as f:
            out |= {l.strip().lower() for l in f if l.strip() and not l.startswith("#")}
    return out


def main(lang, mode="all"):
    lang = lang.upper()
    work = os.path.join(ROOT, "work", lang)
    raw_dir = os.path.join(work, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    by_series = collections.defaultdict(list)
    for dp, _, fns in os.walk(DATA):
        for fn in fns:
            if fn.endswith(f"_{lang}.srt"):
                series = os.path.relpath(dp, DATA).split(os.sep)[0]
                by_series[series].append(os.path.join(dp, fn))
    tasks = [(s, sorted(f), lang, os.path.join(raw_dir, f"{i:05d}.jsonl.gz"))
             for i, (s, f) in enumerate(sorted(by_series.items()))]
    print(f"{lang}: {sum(len(t[1]) for t in tasks)} files in {len(tasks)} series", flush=True)

    stats_path = os.path.join(work, "vocab_stats.pkl")
    if mode == "vocab":
        import pickle
        with open(stats_path, "rb") as f:
            cap, chunk_df, series_df, tok_total, n_chunks, n_series = pickle.load(f)
        shard_files = [t[3] for t in tasks]
    else:
        cap, chunk_df, series_df, tok_total, n_chunks, n_series = parse_all(tasks, work, stats_path)
        shard_files = [t[3] for t in tasks]
    build_vocab_and_corpus(lang, work, shard_files, cap, chunk_df, series_df, tok_total,
                           n_chunks, n_series)


def parse_all(tasks, work, stats_path):
    import pickle
    cap = collections.defaultdict(lambda: [0, 0])
    series_df = collections.Counter()
    tok_total = collections.Counter()
    episodes = []
    with Pool(16) as pool:
        for k, (series, eps, cs) in enumerate(pool.imap_unordered(process_series, tasks, chunksize=1)):
            episodes.extend(eps)
            for w, (a, b) in cs.items():
                c = cap[w]
                c[0] += a
                c[1] += b
            if k % 200 == 0:
                print(f"  {k}/{len(tasks)} series, {len(episodes)} episodes", flush=True)
    ep_df = pd.DataFrame(episodes)
    ep_df.to_csv(os.path.join(work, "episodes.csv"), index=False)
    print(f"kept {len(ep_df)} episodes, {ep_df.n_chunks.sum()} chunks", flush=True)

    # ---- vocabulary statistics: frequency, chunk-df, series-df -------------------
    chunk_df = collections.Counter()
    n_chunks = 0
    shard_files = [t[3] for t in tasks]
    for sf in shard_files:
        seen_series = set()
        with gzip.open(sf, "rt", encoding="utf-8") as fi:
            for line in fi:
                d = json.loads(line)
                n_chunks += 1
                s = set(d["t"])
                chunk_df.update(s)
                seen_series |= s
                tok_total.update(d["t"])
        series_df.update(seen_series)
    n_series = len(shard_files)
    cap = dict(cap)
    with open(stats_path, "wb") as f:
        pickle.dump((cap, chunk_df, series_df, tok_total, n_chunks, n_series), f)
    return cap, chunk_df, series_df, tok_total, n_chunks, n_series


def build_vocab_and_corpus(lang, work, shard_files, cap, chunk_df, series_df, tok_total,
                           n_chunks, n_series):
    # ---- vocabulary rules ---------------------------------------------------------
    stop = STOP[lang]
    names = load_names(lang)
    drop_reason = collections.Counter()
    keep = set()
    for w, f in tok_total.items():
        if w.startswith("\x00"):
            drop_reason["person-name (pymorphy)"] += 1; continue
        if w in stop or len(w) < 2:
            drop_reason["stopword"] += 1; continue
        if chunk_df[w] < 50:
            drop_reason["rare"] += 1; continue
        a, b = cap.get(w, (0, 0))
        capr = a / b if b >= 5 else 0.0
        # first names (mostly written capitalised: keeps "will", "hope", "grace" as words)
        if w in names and capr >= 0.5:
            drop_reason["first-name"] += 1; continue
        # other proper nouns not shared by most shows => characters, local places
        if capr >= 0.8 and series_df[w] < 0.3 * n_series:
            drop_reason["proper-noun"] += 1; continue
        if chunk_df[w] > 0.5 * n_chunks:
            drop_reason["too-common"] += 1; continue
        keep.add(w)
    print("vocab kept:", len(keep), "dropped:", dict(drop_reason), flush=True)
    with open(os.path.join(work, "vocab_kept_propernouns.txt"), "w") as f:
        for w, _ in tok_total.most_common():
            a, b = cap.get(w, (0, 0))
            if w in keep and b >= 5 and a / b >= 0.5:
                f.write(f"{w}\t{tok_total[w]}\t{a / b:.2f}\t{series_df[w]}\n")
    with open(os.path.join(work, "vocab_dropped_common.txt"), "w") as f:
        for w, _ in tok_total.most_common(400):
            if w not in keep:
                f.write(f"{w}\t{tok_total[w]}\n")

    # ---- final corpus -------------------------------------------------------------
    out = os.path.join(work, "chunks.jsonl.gz")
    kept_tokens = 0
    with gzip.open(out, "wt", encoding="utf-8") as fo:
        for sf in shard_files:
            with gzip.open(sf, "rt", encoding="utf-8") as fi:
                for line in fi:
                    d = json.loads(line)
                    d["t"] = [w for w in d["t"] if w in keep]
                    kept_tokens += len(d["t"])
                    fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"done: {n_chunks} chunks, {kept_tokens} content tokens "
          f"({kept_tokens / n_chunks:.0f}/chunk)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
