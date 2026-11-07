import csv,gzip,json,hashlib,collections,itertools,math,time,sys,platform
from pathlib import Path
import numpy as np

def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p,encoding='utf8')) as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,r):
    with open(p,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in r for k in x)),delimiter='\t');w.writeheader();w.writerows(r)
def tau(x,y):
    x=np.array(x);y=np.array(y);i,j=np.triu_indices(len(x),1);dx=np.sign(x[i]-x[j]);dy=np.sign(y[i]-y[j]);den=np.sqrt(np.sum(dx!=0)*np.sum(dy!=0))
    return float(np.sum(dx*dy)/den) if den else math.nan
def choice_loss(source,target,tol):
    methods=sorted(set(source)&set(target));assert methods
    best=max(source[m] for m in methods);chosen=[m for m in methods if abs(source[m]-best)<=tol]
    optimum=max(target[m] for m in methods);span=optimum-min(target[m] for m in methods)
    losses=[max(0.,optimum-target[m]) for m in chosen]
    return dict(chosen='|'.join(chosen),n_tied=len(chosen),loss_mean=float(np.mean(losses)),loss_min=min(losses),loss_max=max(losses),target_range=span,relative_loss=float(np.mean(losses))/span if span>tol else '',oracle=optimum)
def tests():
    assert tau([1,2,3],[1,2,3])==1
    assert tau([1,2,3],[3,2,1])==-1
    assert math.isnan(tau([1,1,1],[1,2,3]))
    r=choice_loss({'A':1,'B':1},{'A':0,'B':.2},1e-12);assert r['loss_mean']==.1 and r['loss_min']==0 and r['loss_max']==.2 and r['n_tied']==2
    assert choice_loss({'A':1,'B':0},{'A':.3,'B':.1},1e-12)['loss_mean']==0
    assert choice_loss({'A':1,'B':0},{'A':.1,'B':.1},1e-12)['relative_loss']==''

def run(cfg):
    tests();start=time.time();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    scores=read(cfg['source_scores']);markers=read(cfg['matched_markers']);groups=collections.defaultdict(list)
    for r in markers:groups[(r['dataset_key'],float(r['depth_fraction']),r['representation'],r['cluster'])].append(r)
    percluster=[]
    for (ds,dep,rep,cluster),rr in groups.items():
        n=len(rr);hits=sum(float(r['rank'])<=cfg['top_n'] for r in rr)
        percluster.append(dict(dataset=ds,depth=dep,representation=rep,cluster=cluster,n_markers=n,hits=hits,precision=hits/cfg['top_n'],recall=hits/n,mean_percentile=float(np.mean([float(r['marker_percentile']) for r in rr]))))
    write(out/'matched_marker_clusters.tsv',percluster)
    summary={}
    for key in set((r['dataset'],r['depth'],r['representation']) for r in percluster):
        rr=[r for r in percluster if (r['dataset'],r['depth'],r['representation'])==key]
        summary[key]={k:float(np.mean([r[k] for r in rr])) for k in ['precision','recall','mean_percentile']}
    corrections=[]
    for r in scores:
        r['value']=float(r['value'])
        if r['stratum'].startswith('real') and r['endpoint'].startswith('marker_'):
            _,ds,dep=r['unit_id'].split('__');rep='F_signac_geneactivity' if r['method_id']=='F' else r['method_id'];key=(ds,float(dep[1:]),rep)
            endpoint='precision' if r['endpoint']=='marker_mean_precision' else 'recall';new=summary[key][endpoint]
            corrections.append(dict(unit=r['unit_id'],method=r['method_id'],endpoint=r['endpoint'],old=r['value'],corrected=new,delta=new-r['value']))
            r['value']=new
        for k in ['rank','top1_set','top1_unique']:r.pop(k,None)
    write(out/'marker_correction_audit.tsv',corrections)
    write(out/'corrected_unit_scores.tsv',scores)
    unit=collections.defaultdict(lambda:collections.defaultdict(dict))
    for r in scores:unit[(r['stratum'],r['unit_id'])][r['endpoint']][r['method_id']]=r['value']
    losses=[];taus=[]
    for (stratum,uid),eps in unit.items():
        for a,b in itertools.product(sorted(eps),repeat=2):
            ms=sorted(set(eps[a])&set(eps[b]));result=choice_loss(eps[a],eps[b],cfg['tie_tolerance'])
            losses.append(dict(stratum=stratum,unit=uid,source=a,target=b,n_candidates=len(ms),**result))
            if a!=b:taus.append(dict(stratum=stratum,unit=uid,source=a,target=b,tau=tau([eps[a][m] for m in ms],[eps[b][m] for m in ms])))
    write(out/'cross_task_loss.tsv',losses);write(out/'endpoint_tau.tsv',taus)
    agg=[]
    for stratum,a,b in sorted(set((r['stratum'],r['source'],r['target']) for r in losses)):
        rr=[r for r in losses if (r['stratum'],r['source'],r['target'])==(stratum,a,b)];rel=[r['relative_loss'] for r in rr if r['relative_loss']!='']
        agg.append(dict(stratum=stratum,source=a,target=b,n_units=len(rr),n_nonzero=sum(r['loss_mean']>cfg['tie_tolerance'] for r in rr),n_tied_source=sum(r['n_tied']>1 for r in rr),mean_loss=float(np.mean([r['loss_mean'] for r in rr])),median_loss=float(np.median([r['loss_mean'] for r in rr])),mean_relative_loss=float(np.mean(rel)) if rel else '',n_relative_estimable=len(rel)))
    write(out/'cross_task_summary.tsv',agg)
    # Hold out whole study families, keeping mouse and human Zhang samples together.
    domain=read(cfg['domain_ds1'])+read(cfg['domain_human']);dgroups=collections.defaultdict(list)
    for r in domain:
        rep=r.get('representation',r.get('representation_key'));ds=r['dataset_key']
        if float(r['depth_fraction'])!=cfg['cv_depth'] or rep not in cfg['cv_candidates']:continue
        for endpoint in ['ARI','NMI']:dgroups[(ds,endpoint,rep)].append(float(r[endpoint]))
    dmed={k:float(np.median(v)) for k,v in dgroups.items()};families=sorted(set(cfg['family_map'][ds] for ds,_,_ in dmed))
    family_scores={}
    for family,ep,method in itertools.product(families,['ARI','NMI'],cfg['cv_candidates']):
        vals=[v for (ds,e,m),v in dmed.items() if cfg['family_map'][ds]==family and e==ep and m==method]
        if vals:family_scores[(family,ep,method)]=float(np.mean(vals))
    cv=[]
    for heldout,ep in itertools.product(families,['ARI','NMI']):
        train=[f for f in families if f!=heldout];assert heldout not in train
        assert len(train)>=cfg['cv_min_training_families']
        tr={m:float(np.mean([family_scores[f,ep,m] for f in train])) for m in cfg['cv_candidates']};target={m:family_scores[heldout,ep,m] for m in cfg['cv_candidates']}
        res=choice_loss(tr,target,cfg['tie_tolerance']);cv.append(dict(heldout_family=heldout,endpoint=ep,training_families='|'.join(train),n_training_families=len(train),train_B_minus_P=tr['B']-tr['P'],test_B_minus_P=target['B']-target['P'],fixed_B_loss=max(target.values())-target['B'],fixed_P_loss=max(target.values())-target['P'],uniform_random_loss=max(target.values())-np.mean(list(target.values())),**res))
    write(out/'leave_family_out.tsv',cv)
    # Explicitly account for endpoint scope instead of labelling two-family fits validated.
    write(out/'leave_family_out_applicability.tsv',[{'endpoint':'domain_ARI','families':3,'status':'retrospective_complete'},{'endpoint':'domain_NMI','families':3,'status':'retrospective_complete'},{'endpoint':'marker','families':2,'status':'abstain_fewer_than_two_training_families'},{'endpoint':'svg','families':2,'status':'abstain_fewer_than_two_training_families'}])
    sources={cfg[k]:hashlib.sha256(Path(cfg[k]).read_bytes()).hexdigest() for k in ['source_scores','matched_markers','domain_ds1','domain_human']}
    record=dict(status='pass',runtime_seconds=time.time()-start,seed=None,software={'python':platform.python_version(),'numpy':np.__version__},config=cfg,inputs=sources,n_corrections=len(corrections),n_changed=sum(abs(r['delta'])>1e-12 for r in corrections),n_loss_rows=len(losses),n_cv_rows=len(cv),tests='tau order/constant, tie averaging, zero range, grouped study exclusion passed')
    record['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.tsv')};(out/'run_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps({k:record[k] for k in ['status','n_corrections','n_changed','n_loss_rows','n_cv_rows']}));print('CV',cv)
    print('REAL task losses',[r for r in agg if r['stratum']=='real_primary' and r['source']=='domain_ari'])

if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
