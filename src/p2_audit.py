"""Tokenizer audit limited to the primary matched P2 cohorts."""
import gzip
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from compare_registers import ROOT, OUT


def main():
    tokenizer=AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small',local_files_only=True)
    cohort=pd.read_csv(OUT/'cohorts.csv').query("case=='matched'")
    result={}
    for source in ['TALES','PROZA']:
        keep=set(cohort[cohort.source==source].eid)
        lengths=[];affected=set();chunks=0;bad_chunks=0
        with gzip.open(ROOT/f'work/EMB/{source}_texts.jsonl.gz','rt') as f:
            for line in f:
                row=json.loads(line)
                if row['eid'] not in keep:continue
                words=row['text'].split();n=max(1,-(-len(words)//220))
                text=['passage: '+' '.join(words[q*len(words)//n:(q+1)*len(words)//n]) for q in range(n)]
                sizes=[len(t) for t in tokenizer(text,truncation=False,add_special_tokens=True)['input_ids']]
                lengths.extend(sizes);chunks+=1
                if any(s>512 for s in sizes):affected.add(row['eid']);bad_chunks+=1
        lengths=np.asarray(lengths)
        result[source]=dict(texts=len(keep),chunks=chunks,windows=len(lengths),
                            over_512=int((lengths>512).sum()),fraction_over_512=float((lengths>512).mean()),
                            affected_chunks=bad_chunks,affected_texts=len(affected),affected_eids=sorted(affected),
                            median_tokens=float(np.median(lengths)),max_tokens=int(lengths.max()))
        print(source,{k:v for k,v in result[source].items() if k!='affected_eids'},flush=True)
    json.dump(result,open(OUT/'window_audit.json','w'),indent=2)


if __name__=='__main__':main()
