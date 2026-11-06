"""Post-hoc bridge to earlier top-100 LDA analysis; preserve P2 primary results."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compare_registers import OUT, load_source, bootstrap_weights, profiles, eigstats


def main():
    cohorts=pd.read_csv(OUT/'cohorts.csv').query("case=='matched'")
    variants={}
    for source in ['TALES','PROZA']:
        ep,reps=load_source(source)
        sub=cohorts[cohorts.source==source]
        idx=pd.Index(ep.eid).get_indexer(sub.eid)
        assert (idx>=0).all()
        rep=reps['LDA']
        sh,le=rep['shares'][idx],rep['length'][idx]
        w,pt,_=bootstrap_weights(sub.unit_author.to_numpy(),sub.stratum.to_numpy(),
                                np.random.default_rng(42 if source=='TALES' else 43))
        variants[source]=dict(sh=sh,le=le,w=w,pt=pt)
    common=np.argsort(-(variants['TALES']['sh'].mean((0,1))+variants['PROZA']['sh'].mean((0,1))))[:100]
    rows=[]
    for variant in ['all127','shared100','source_specific100']:
        for weighting in ['text','word']:
            stats={}
            for source,v in variants.items():
                top=np.arange(127) if variant=='all127' else common if variant=='shared100' else np.argsort(-v['sh'].mean((0,1)))[:100]
                a=profiles(v['sh'][:,:,top],v['le'],np.concatenate([v['pt'][None],v['w']]),weighting)
                stats[source]=eigstats(a,standardized=True)['pc2_remaining']
            delta=stats['TALES']-stats['PROZA']
            lo,hi=np.quantile(delta[1:],[.025,.975])
            rows.append(dict(variant=variant,weighting=weighting,tales=stats['TALES'][0],
                             proza=stats['PROZA'][0],difference=delta[0],ci95_low=lo,ci95_high=hi,
                             status='post-hoc coordinate sensitivity; primary unchanged'))
    pd.DataFrame(rows).to_csv(OUT/'coordinate_sensitivity.csv',index=False)
    json.dump(dict(status='post-hoc bridge; no new primary test',shared100=common.tolist(),
                   note='source-specific top100 preserves earlier coordinate selection style, not identical old cohorts'),
              open(OUT/'coordinate_protocol.json','w'),indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
