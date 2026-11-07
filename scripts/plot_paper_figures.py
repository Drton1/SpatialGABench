from pathlib import Path
import csv,gzip,json,math,hashlib,collections,os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize,TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

O=Path('results/paper/figures');O.mkdir(parents=True,exist_ok=True)
E=Path('data/paper/evidence');X=Path('data/paper/depth');S=Path('results/paper/selection');R=Path('data/paper/frozen_peak')
BLUE='#0072B2';GREEN='#009E73';RED='#D55E00';PURPLE='#8863AB';INK='#25282B';PALETTE=[BLUE,GREEN,RED,PURPLE]
SEQ=LinearSegmentedColormap.from_list('ocean',['#F4F7FA','#C0D8EA','#6AA9CD','#236E9A','#083D5E']);SEQ.set_bad('#D9DCDD')
DIV=LinearSegmentedColormap.from_list('contrast',['#2166AC','#A5C7E0','#FAF8F3','#E9A28C','#AD1735']);DIV.set_bad('#D9DCDD')
WARM=LinearSegmentedColormap.from_list('warm',['#FFF8D6','#F6CC64','#E7864C','#B8455D','#623677']);WARM.set_bad('#D9DCDD')
def paper_font():
    if os.environ.get('SPATIALGABENCH_FONT'):return os.environ['SPATIALGABENCH_FONT']
    for name in ['Arial','Liberation Sans']:
        try:mpl.font_manager.findfont(name,fallback_to_default=False);return name
        except ValueError:pass
    return 'DejaVu Sans'

mpl.rcParams.update({'font.family':paper_font(),'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
METHODS=['promoter_only_2kb_pc','gene_body_pc','promoter_plus_gene_body_2kb_pc','distance_weighted_distal_250kb_decay50kb_pc','promoter_plus_distal_2kb_250kb_decay50kb_pc'];ML=['Promoter','Body','Pmt+body','Distal','Pmt+distal'];DS=['GSE263333','P2S3','P2S1','P2S2'];DEPS=[.01,.05,.25,1.];DL=['1%','5%','25%','100%'];REPS=['macs3_summit_501bp','fixed_bin_25kb'];INPUTS=['Peak','B25','Signac'];IC=[BLUE,GREEN,RED];audit=[];sources={}
def tab(p):
    p=Path(p);sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    with (gzip.open(p,'rt') if p.suffix=='.gz' else open(p,encoding='utf8')) as f:return list(csv.DictReader(f,delimiter='\t'))
def num(x):
    try:return float(x)
    except:return math.nan
def med(v):
    v=[num(x) for x in v if math.isfinite(num(x))];return float(np.median(v)) if v else math.nan
def title(ax,letter,text):
    ax._panel_letter=letter.upper();ax._panel_name=text
    ax.set_title(text,loc='left',pad=9);ax.tick_params(length=2.5,width=.6)

def frame(ax,nr,nc):
    ax.set_xticks(np.arange(-.5,nc,1),minor=True);ax.set_yticks(np.arange(-.5,nr,1),minor=True);ax.grid(which='minor',color='#62666A',lw=.35);ax.tick_params(which='both',length=0)
    for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.6)
def heat(ax,a,yl,xl,vmin=None,vmax=None,cmap=SEQ,annotate=False):
    a=np.array(a);im=ax.imshow(a,aspect='auto',interpolation='none',cmap=cmap,vmin=vmin,vmax=vmax);ax.set_yticks(range(a.shape[0]),yl);ax.set_xticks(range(a.shape[1]),xl);frame(ax,*a.shape)
    if annotate:
        for i,j in np.ndindex(a.shape):
            v=a[i,j];rgba=im.cmap(im.norm(v));lum=sum(x*w for x,w in zip(rgba[:3],[.2126,.7152,.0722]));txt='NA' if not np.isfinite(v) else (f'{v:.2f}' if abs(v)>=.01 else f'{v:.3f}')
            if np.isfinite(v) and 0<abs(v)<.0005:txt=f'{v:.1e}'
            ax.text(j,i,txt,ha='center',va='center',fontsize=7.5,color='white' if lum<.5 else INK)
    audit.append({'title':ax.get_title(loc='left'),'shape':list(a.shape),'values':[[None if not np.isfinite(v) else float(v) for v in row] for row in a]});return im
def cb(fig,im,ax,label):
    c=fig.colorbar(im,ax=ax,fraction=.045,pad=.035,aspect=28)
    c.set_label(label,fontsize=8,labelpad=3);c.ax.tick_params(labelsize=8,length=2,pad=2);c.outline.set_linewidth(.5)
    c.locator=mpl.ticker.MaxNLocator(nbins=4);c.update_ticks()

def grouped(ax,names,block=5):
    for i,s in enumerate(names):
        ax.text(-.46,i*block+(block-1)/2,s,transform=ax.get_yaxis_transform(),ha='center',va='center',rotation=90,fontsize=8,fontweight='normal')
        if i:ax.axhline(i*block-.5,color=INK,lw=1)
def legend(ax,labels=INPUTS,colors=IC,**kwargs):ax.legend([Line2D([0],[0],marker=(['o','s','^'][i] if labels==INPUTS else 'o'),lw=1,color=c,markersize=4) for i,c in enumerate(colors)],labels,frameon=False,fontsize=8,**kwargs)
def save(fig,n):
    # Author at publication width. Text is never shrunk by the manuscript include.
    panels=[a for a in fig.axes if hasattr(a,'_panel_letter')]
    for a in panels:a.set_title('',loc='left')
    for text in fig.findobj(mpl.text.Text):
        if text.get_text() and text.get_visible():text.set_fontsize(min(10,max(8,text.get_fontsize())))
    for a in fig.axes:
        a.xaxis.labelpad=3;a.yaxis.labelpad=3
        a.tick_params(axis='both',which='major',pad=2,labelsize=8)
    fig.canvas.draw()
    for a in panels:
        p=a.get_position();x=p.x0-0.10
        if n==3:x=p.x0-.11
        fig.text(max(.012,x),p.y1+.017,a._panel_letter,fontsize=12,fontweight='bold',va='bottom')
    fig.text(.012,.994,'Chen et al.  |  Figure '+str(n),fontsize=8,va='top')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();W,H=fig.canvas.get_width_height()
    unused=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo-1e-9<=tick.get_loc()<=hi+1e-9:unused.update([tick.label1,tick.label2])
    outside=[];textlog=[]
    for a in fig.findobj(mpl.text.Text):
        if not a.get_visible() or not a.get_text() or a in unused:continue
        # Ignore unused default axis labels for panels composed of inset axes.
        if a.axes is not None and not a.axes.axison and a in (list(a.axes.get_xticklabels())+list(a.axes.get_yticklabels())):continue
        b=a.get_window_extent(renderer)
        if b.width==0 or b.height==0:continue
        textlog.append({'text':a.get_text(),'font_pt':a.get_fontsize(),'bbox_px':[b.x0,b.y0,b.x1,b.y1]})
        if b.x0 < -1 or b.y0 < -1 or b.x1>W+1 or b.y1>H+1:outside.append(a.get_text())
    (O/f'Fig{n}_layout_audit.json').write_text(json.dumps({'size_inches':list(fig.get_size_inches()),'outside_canvas_text':outside,'text':textlog},indent=2))
    for fmt in ['pdf','svg','png']:fig.savefig(O/f'Chen_Fig{n}.{fmt}',dpi=600 if fmt=='pdf' else 180)
    plt.close(fig);print('Fig',n,'saved; outside:',outside,flush=True)

def fig2():
    domain=tab('data/paper/spatial/domain_seed_metrics.tsv');marker=tab('data/paper/spatial/marker_percentiles_cluster_matched.tsv.gz');svg=tab('data/paper/spatial/svg_moran_i.tsv.gz');program=tab(E/'spatial_program_controls.tsv')
    fig=plt.figure(figsize=(7,8.6));gs=fig.add_gridspec(4,2,left=.12,right=.91,top=.94,bottom=.065,wspace=.65,hspace=.33,height_ratios=[1.7,1.05,1.0,2.1])
    base=fig.add_subplot(gs[0,:]);title(base,'a','Spatial domains');base.axis('off');sub=gs[0,:].subgridspec(2,4,wspace=.03,hspace=.38)
    p=json.load(gzip.open('data/paper/spatial/fig2_real_spatial_sample.json.gz','rt'));arr=np.array(p['rows']);xy=arr[:,:2].astype(float);cols=['#6EA8C8','#DF8A7D','#82B196','#D6B15B','#AA91BE','#46769C','#C7A18E','#86BDB1']
    for i,dep in enumerate([1.,.05]):
        for j,(rep,name) in enumerate(zip([None,'P','B','F_signac_geneactivity'],['RNA','Peak','B25','Signac'])):
            a=fig.add_subplot(sub[i,j]);labels=arr[:,2 if rep is None else p['keys'].index(f'{dep}_{rep}')+3].astype(int)
            a.scatter(xy[:,0],-xy[:,1],s=.65,c=[cols[k] for k in labels],lw=0,rasterized=True);a.set_aspect('equal');a.set_axis_off()
            if i==0:a.set_title(name,fontsize=9,pad=2)
            if j==0:a.text(-.15,.5,['100%','5%'][i],transform=a.transAxes,rotation=90,ha='right',va='center',fontsize=9)
            if rep:
                rr=[r for r in domain if r['dataset_key']=='d1_gse263333' and num(r['depth_fraction'])==dep and r['representation']==rep and r['clustering_seed']=='20261018'];assert len(rr)==1
                a.text(.5,-.03,f"{float(rr[0]['ARI']):.3f}",transform=a.transAxes,ha='center',va='top',fontsize=8)
    ax=fig.add_subplot(gs[1,0]);title(ax,'b','Domain agreement')
    for j,(rep,col) in enumerate(zip(['P','B','F_signac_geneactivity'],IC)):
        vals=[]
        for k,d in enumerate(DEPS):
            rr=[r for r in domain if r['dataset_key']=='d1_gse263333' and num(r['depth_fraction'])==d and r['representation']==rep and r['depth_fallback_used']=='False'];v=[num(r['ARI']) for r in rr];vals.append(med(v));ax.scatter(np.full(len(v),k)+(j-1)*.06,v,s=14,c=col,edgecolors='white',lw=.3,zorder=3)
        ax.plot(range(4),vals,c=col,lw=1,marker=['o','s','^'][j],markersize=3)
    ax.set_xticks(range(4),DL);ax.set_ylabel('ARI');ax.set_xlabel('Retained depth');legend(ax,loc='upper left');ax.set_ylim(-.004,.14)
    ax=fig.add_subplot(gs[1,1]);title(ax,'c','Marker enrichment')
    for dep,ls in [(1.,'-'),(.05,'--')]:
        for rep,c in zip(['P','B','F_signac_geneactivity'],IC):
            v=sorted(num(r['marker_percentile']) for r in marker if r['dataset_key']=='d1_gse263333' and num(r['depth_fraction'])==dep and r['representation']==rep)
            ax.plot(v,np.arange(1,len(v)+1)/len(v),color=c,ls=ls,lw=.9,marker=['o','s','^'][IC.index(c)],markevery=max(1,len(v)//10),markersize=2)
    ax.plot([0,1],[0,1],color='#BBB',lw=.7,zorder=0);ax.set(xlabel='Marker percentile',ylabel='Cumulative fraction',xlim=(0,1),ylim=(0,1));ax.legend([Line2D([0],[0],color=INK,ls=x) for x in ['-','--']],['100%','5%'],frameon=False,fontsize=8,loc='upper left')
    base=fig.add_subplot(gs[2,:]);title(base,'d','Spatially variable genes');base.set_title('Spatially variable genes',loc='left',pad=30);base.axis('off');sub=gs[2,:].subgridspec(1,3,wspace=.35)
    for j,(rep,col) in enumerate(zip(['P','B','F_signac_geneactivity'],IC)):
        a=fig.add_subplot(sub[0,j]);rr=[r for r in svg if r['dataset_key']=='d1_gse263333' and num(r['depth_fraction'])==1 and r['representation']==rep]
        a.scatter([num(r['rna_moran_i']) for r in rr],[num(r['atac_moran_i']) for r in rr],s=3,c=col,alpha=.35,lw=0,rasterized=True);a.set_title(INPUTS[j],fontsize=9);a.set_xlim(-.035,.38);a.set_ylim(-.025,.10);a.set_xticks([0,.15,.3]);a.set_yticks([0,.04,.08]);a.tick_params(labelsize=7)
        if j==0:a.set_ylabel('Activity Moran I')
        else:a.set_yticklabels([])
        if j==0:a.set_xlabel('RNA Moran I')
    ax=fig.add_subplot(gs[3,0]);title(ax,'e','Human-brain input effects');ax.axvline(0,c='#777',lw=.7)
    dom=tab('data/paper/human/e1_domain_pairwise.tsv');mar=tab('data/paper/human/e1_marker_pairwise.tsv');sv=tab('data/paper/human/e1_svg_pairwise.tsv')
    records=[]
    for rr,metric in [(dom,'ARI'),(dom,'NMI'),(mar,None),(sv,None)]:
        r=next(r for r in rr if r['contrast']=='B-P' and (metric is None or r['metric']==metric));records.append(r)
    for i,r in enumerate(records):
        x=num(r['point_difference_b_minus_a']);lo=num(r['ci95_low']);hi=num(r['ci95_high']);ax.plot([lo,hi],[3-i]*2,color=PALETTE[i],lw=1.5);ax.scatter(x,3-i,s=40,c=PALETTE[i],edgecolor=INK,lw=.4,zorder=3)
    ax.set_yticks(range(4),['SVG','Markers','NMI','ARI'][::-1][::-1]);ax.set_yticks([3,2,1,0],['ARI','NMI','Markers','SVG']);ax.set(xlabel='B25 − peak',ylim=(-.5,3.5));ax.grid(axis='x',lw=.4,c='#E5E5E5')
    ax=fig.add_subplot(gs[3,1]);title(ax,'f','Input resolution')
    datasets=['gse263333_5m_20um_atac','gse308623_p2s3','gse205052_mousebrain_20um','gse205055_humanbrain_50um'];reps=[REPS[0],'fixed_bin_10kb',REPS[1],'fixed_bin_50kb'];a=np.full((20,4),np.nan)
    for i,ds in enumerate(datasets):
        for j,m in enumerate(METHODS):
            for k,rep in enumerate(reps):a[i*5+j,k]=med([r['median_heldout_component_spearman'] for r in program if r['dataset_id']==ds and r['method_id']==m and r['representation_id']==rep])
    im=heat(ax,a,ML*4,['Peak','10 kb','25 kb','50 kb'],0,.5);ax.tick_params(axis='y',labelsize=8);grouped(ax,['263333','P2S3','Mouse','Human']);cb(fig,im,ax,'Program ρ');save(fig,2)

def fig3():
    coverage=tab(X/'coverage.tsv');quant=tab(X/'gene_quantiles.tsv');common=tab(E/'common_variable.tsv')
    fig=plt.figure(figsize=(7,8.65));gs=fig.add_gridspec(3,2,left=.16,right=.90,top=.945,bottom=.09,wspace=.75,hspace=.26,height_ratios=[1.1,1.15,1])
    for index,(col,letter,name,limits) in enumerate([('coverage_adjusted_median_gene_wise_spearman','a','RNA agreement',(-.005,.085)),('evaluable_gene_fraction','b','Variable-gene coverage',(0,1))]):
        ax=fig.add_subplot(gs[0,index]);title(ax,letter,name);a=np.full((20,8),np.nan)
        for i,s in enumerate(DS):
            for j,m in enumerate(METHODS):
                for k,(d,rep) in enumerate((d,r) for d in DEPS for r in REPS):a[i*5+j,k]=med([r[col] for r in coverage if r['sample']==s and num(r['depth'])==d and r['method']==m and r['representation']==rep])
        im=heat(ax,a,ML*4,['P','B']*4,*limits);ax.set_title(name,loc='left',pad=28);grouped(ax,DS);ax.tick_params(axis='y',labelsize=7)
        for k,l in enumerate(DL):ax.text(k*2+.5,-1.2,l,ha='center',fontsize=8);ax.axvline(k*2-.5,c=INK,lw=.9)
        cb(fig,im,ax,'RNA score' if index==0 else 'Fraction')
    ax=fig.add_subplot(gs[1,0]);title(ax,'c','Gene-wise distributions · 5%');ax.axvline(0,c='#BBB',lw=.6)
    for i,s in enumerate(DS):
        for j,rep in enumerate(REPS):
            for m in METHODS:
                rr=[r for r in quant if r['sample']==s and num(r['depth'])==.05 and r['method']==m and r['representation']==rep and r['subset']=='present'];qs=[med([r[k] for r in rr]) for k in ['q05','q25','q50','q75','q95']]
                y=i*5+METHODS.index(m)+(j-.5)*.3;ax.plot([qs[0],qs[4]],[y,y],color=IC[j],lw=.6);ax.plot([qs[1],qs[3]],[y,y],color=IC[j],lw=2.5);ax.scatter(qs[2],y,c=IC[j],s=8,zorder=3)
    ax.set_xlim(-.025,.085);ax.set_ylim(19.6,-.6);ax.set_yticks(range(20),ML*4);ax.set_xticks([0,.04,.08]);ax.set_xlabel('Gene-wise ρ');grouped(ax,DS);legend(ax,['Peak','B25'],IC[:2],loc='lower right',bbox_to_anchor=(1,1.01),ncol=2,handlelength=.7,columnspacing=.7,borderaxespad=0)
    ax=fig.add_subplot(gs[1,1]);title(ax,'d','Zero-output observations');a=np.full((20,8),np.nan)
    for i,s in enumerate(DS):
        for j,m in enumerate(METHODS):
            for k,(d,rep) in enumerate((d,r) for d in DEPS for r in REPS):a[i*5+j,k]=med([r['zero_observation_fraction'] for r in coverage if r['sample']==s and num(r['depth'])==d and r['method']==m and r['representation']==rep])
    im=heat(ax,a,ML*4,['P','B']*4,0,1,WARM);ax.set_title('Zero-output observations',loc='left',pad=28);grouped(ax,DS);ax.tick_params(axis='y',labelsize=7);cb(fig,im,ax,'Zero-spot fraction')
    for k,l in enumerate(DL):ax.text(k*2+.5,-1.2,l,ha='center',fontsize=8);ax.axvline(k*2-.5,c=INK,lw=.9)
    ax=fig.add_subplot(gs[2,0]);title(ax,'e','Peak-output states');counts=np.zeros((16,3))
    for i,s in enumerate(DS):
        for j,d in enumerate(DEPS):
            for r in coverage:
                if r['sample']==s and num(r['depth'])==d and r['representation']==REPS[0]:counts[i*4+j,0 if r['evaluation_status']!='pass' else (1 if num(r['coverage_adjusted_median_gene_wise_spearman'])==0 else 2)]+=1
    left=np.zeros(16)
    for k,c in enumerate(['#BEC3C6','#EDD29B',BLUE]):ax.barh(np.arange(16),counts[:,k],left=left,color=c,edgecolor='white',lw=.4,height=.8);left+=counts[:,k]
    ax.invert_yaxis();ax.set_yticks(range(16),DL*4);ax.set_xlim(0,15);ax.set_xticks([0,5,10,15]);ax.set_xlabel('Conditions');grouped(ax,['263333','P2S3','P2S1','P2S2'],4)
    ax.legend([Patch(fc=c) for c in ['#BEC3C6','#EDD29B',BLUE]],['Unavailable','Zero','Nonzero'],frameon=False,fontsize=7.5,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,handlelength=.7,columnspacing=.6,handletextpad=.3)
    ax=fig.add_subplot(gs[2,1]);title(ax,'f','Shared-variable gene sensitivity');a=np.full((10,4),np.nan)
    for i,s in enumerate(['P2S1','P2S2']):
        for j,m in enumerate(METHODS):
            for k,d in enumerate(DEPS):
                prefix={.01:'d001',.05:'d005',.25:'d025',1.:'full'}[d];rr=[r for r in common if r['sample']==s and r['method']==m and r['condition'].startswith(prefix) and r['selection']=='all_ten_variable' and num(r['n_genes'])>0];a[i*5+j,k]=med([num(r['B25_median'])-num(r['P_median']) for r in rr])
    im=heat(ax,a,ML*2,DL,-.04,.04,DIV,True);grouped(ax,['P2S1','P2S2']);cb(fig,im,ax,'Δ median ρ');save(fig,3)

def fig4():
    program=tab(E/'spatial_program_depth.tsv');coverage=tab(X/'coverage.tsv');frozen=tab(R/'frozen_peak_rna_metrics.tsv');occupancy=tab(X/'feature_occupancy.tsv');control=tab(E/'random_controls.tsv');pc=tab(E/'spatial_program_controls.tsv')
    fig=plt.figure(figsize=(7,7.2));gs=fig.add_gridspec(3,2,left=.16,right=.90,top=.94,bottom=.09,wspace=.74,hspace=.52)
    datasets=['gse263333_5m_20um_atac','gse308623_p2s3'];roles=['de_novo_recalled_peak','frozen_full_depth_peak','fixed_bin_25kb'];labels=['P','Fr','B'];cols=[BLUE,PURPLE,GREEN]
    ax=fig.add_subplot(gs[0,0]);title(ax,'a','Spatial-program recovery');a=np.full((10,9),np.nan)
    for i,ds in enumerate(datasets):
        for j,m in enumerate(METHODS):
            for k,(d,role) in enumerate((d,r) for d in DEPS[:3] for r in roles):a[i*5+j,k]=med([r['median_heldout_component_spearman'] for r in program if r['dataset_id']==ds and num(r['depth_fraction'])==d and r['method_id']==m and r['representation_role']==role])
    im=heat(ax,a,ML*2,labels*3,0,.2);ax.set_title('Spatial-program recovery',loc='left',pad=28);grouped(ax,['263333','P2S3']);cb(fig,im,ax,'Program ρ')
    for k,l in enumerate(DL[:3]):ax.text(k*3+1,-.9,l,ha='center',fontsize=8);ax.axvline(k*3-.5,color=INK,lw=1)
    ax=fig.add_subplot(gs[0,1]);title(ax,'b','Matched peak-set rescue · 5%')
    for i,ds in enumerate(datasets):
        for j,m in enumerate(METHODS):
            for seed in [42001,42002,42003]:
                vals=[med([r['median_heldout_component_spearman'] for r in program if r['dataset_id']==ds and num(r['depth_fraction'])==.05 and r['method_id']==m and int(r['seed'])==seed and r['representation_role']==role]) for role in roles]
                ax.plot(np.arange(3)+i*4,vals,color='#ABB3B9',lw=.6,zorder=1)
                for k,c in enumerate(cols):ax.scatter(k+i*4,vals[k],c=c,s=15,lw=.25,edgecolor='white',zorder=2)
        ax.text(i*4+1,.151,['263333','P2S3'][i],ha='center',fontsize=9)
    ax.set_xticks([0,1,2,4,5,6],['Peak','Frozen','B25']*2,rotation=40,ha='right');ax.set_ylabel('Held-out program ρ');ax.set_ylim(-.005,.155)
    ax=fig.add_subplot(gs[1,0]);title(ax,'c','Gene-support recovery');a=np.full((10,9),np.nan)
    for i,s in enumerate(DS[:2]):
        for j,m in enumerate(METHODS):
            for k,d in enumerate(DEPS[:3]):
                for offset,rep in [(0,REPS[0]),(2,REPS[1])]:a[i*5+j,k*3+offset]=med([r['evaluable_gene_fraction'] for r in coverage if r['sample']==s and num(r['depth'])==d and r['method']==m and r['representation']==rep])
                a[i*5+j,k*3+1]=med([r['evaluable_gene_fraction'] for r in frozen if r['dataset_id']==datasets[i] and num(r['depth_fraction'])==d and r['method_id']==m])
    im=heat(ax,a,ML*2,labels*3,0,1);ax.set_title('Gene-support recovery',loc='left',pad=28);grouped(ax,['263333','P2S3']);cb(fig,im,ax,'Variable-gene fraction')
    for k,l in enumerate(DL[:3]):ax.text(k*3+1,-.9,l,ha='center',fontsize=8);ax.axvline(k*3-.5,c=INK,lw=1)
    ax=fig.add_subplot(gs[1,1]);title(ax,'d','Feature availability and occupancy')
    for s,m in zip(DS[:2],['o','s']):
        for label,col in zip(['Peak','Frozen','B25'],cols):
            rr=[r for r in occupancy if r['sample']==s and r['input']==label and r['status']=='pass']
            ax.scatter([num(r['n_features']) for r in rr],[num(r['median_nonzero_features_per_spot'])+.1 for r in rr],s=[12+num(r['depth'])*100 for r in rr],c=col,marker=m,edgecolors=INK,lw=.35,alpha=.82)
    ax.set(xscale='log',yscale='log',xlabel='Available features',ylabel='Occupied features / spot\n(median + 0.1)');legend(ax,['Peak','Frozen','B25'],cols,loc='upper left',ncol=1)
    ax=fig.add_subplot(gs[2,0]);title(ax,'e','Promoter mapping above background')
    keys=[(s,cond,rep) for s in ['P2S1','P2S2'] for cond in ['full','d005_s42001'] for rep in REPS]
    for i,(s,cond,rep) in enumerate(keys):
        rr=[r for r in control if r['sample']==s and r['condition']==cond and r['representation']==rep];assert len(rr)==3
        p=num(rr[0]['promoter_primary']);v=[num(r['control_primary']) for r in rr];ax.plot([min(v),p],[i]*2,color='#8C969E',lw=1);ax.scatter(v,np.arange(3)*.10+i-.1,s=20,facecolor='white',edgecolor=PURPLE,lw=.8);ax.scatter(p,i,s=35,c=GREEN,marker='D',edgecolor=INK,lw=.4,zorder=3)
    ax.set_yticks(range(8),['100% P','100% B','5% P','5% B']*2);ax.invert_yaxis();grouped(ax,['P2S1','P2S2'],4);ax.set_xlabel('RNA score');ax.legend([Line2D([0],[0],marker='D',lw=0,color=GREEN),Line2D([0],[0],marker='o',lw=0,mfc='white',mec=PURPLE)],['Promoter','Random'],frameon=False,fontsize=7,loc='lower right')
    ax=fig.add_subplot(gs[2,1]);title(ax,'f','Program-control contrasts');a=np.full((5,8),np.nan);dsets=['gse263333_5m_20um_atac','gse308623_p2s3','gse205052_mousebrain_20um','gse205055_humanbrain_50um']
    for i,m in enumerate(METHODS):
        for j,(ds,rep) in enumerate((d,r) for d in dsets for r in REPS):a[i,j]=med([r['delta_heldout_component_spearman_vs_control'] for r in pc if r['dataset_id']==ds and r['method_id']==m and r['representation_id']==rep])
    im=heat(ax,a,ML,['P','B']*4,0,.5);ax.set_title('Program-control contrasts',loc='left',pad=28);ax.axhline(.5,color=INK,lw=1.8);cb(fig,im,ax,'Δ program ρ')
    for k,l in enumerate(['263333','P2S3','Mouse','Human']):ax.text(k*2+.5,-.8,l,ha='center',fontsize=7.5);ax.axvline(k*2-.5,c=INK,lw=1)
    save(fig,4)

def matrix_tau(rows,stratum,endpoints):
    a=np.full((len(endpoints),len(endpoints)),np.nan)
    for i,e in enumerate(endpoints):
        for j,f in enumerate(endpoints):
            if i!=j:a[i,j]=med([r['tau'] for r in rows if r['stratum']==stratum and r['source']==e and r['target']==f])
    return a
def bubbles(ax,a,labels):
    n=len(labels);ax.set_xlim(-.5,n-.5);ax.set_ylim(n-.5,-.5);ax.set_aspect('auto');ax.set_xticks(range(n),labels,rotation=45,ha='right');ax.set_yticks(range(n),labels);frame(ax,n,n)
    for i,j in np.ndindex(a.shape):
        v=a[i,j]
        if not np.isfinite(v):ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fc='#E7E9EA',ec='none'));continue
        ax.scatter(j,i,s=420*abs(v),c=[v],cmap=DIV,vmin=-1,vmax=1,lw=.3,edgecolor='#555',alpha=.65)
        if abs(v)>.70:ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,color=INK)
    audit.append({'title':ax.get_title(loc='left'),'shape':list(a.shape),'values':[[None if not np.isfinite(v) else float(v) for v in row] for row in a]})
def fig5():
    rank=tab('data/paper/context/rank2_context_rankings.tsv');effects=tab('data/paper/context/fam1_endpoint_family_effects.tsv');taus=tab(S/'endpoint_tau.tsv');loss=tab(S/'cross_task_summary.tsv');cv=tab(S/'leave_family_out.tsv')
    fig=plt.figure(figsize=(7,8.15));gs=fig.add_gridspec(3,2,left=.16,right=.90,top=.94,bottom=.09,wspace=.74,hspace=.52)
    ax=fig.add_subplot(gs[0,0]);title(ax,'a','Workflow performance by context');contexts=['d1_gse263333__d0.05','d1_gse263333__d0.25','d1_gse263333__d1.00','d2_gse205052__d1.00','d2_human_gse205055__d1.00']
    for i,context in enumerate(contexts):
        for j,(rep,c) in enumerate(zip(['P','B','F'],IC)):
            r=next(r for r in rank if r['context_id']==context and r['metric']=='ARI' and r['method_id']==rep);x=num(r['seed_median']);y=i+(j-1)*.19;ax.plot([0,x],[y,y],c=c,lw=1);ax.scatter(x,y,s=26,c=c,marker=['o','s','^'][j],edgecolor=INK,lw=.4);ax.text(x+.003,y,str(int(float(r['overall_rank']))),va='center',fontsize=7.5)
    ax.set_yticks(range(5),['263333 5%','263333 25%','263333 100%','205052 100%','205055 100%']);ax.invert_yaxis();ax.set_xlim(0,max(num(r['seed_median']) for r in rank if r['metric']=='ARI')*1.16);ax.set_xlabel('ARI');ax.set_ylim(5.6,-.5);legend(ax,ncol=1,loc='lower right',handlelength=1,labelspacing=.15)
    ax=fig.add_subplot(gs[0,1]);title(ax,'b','Study-family input effects');families=['guo_2025_spatial_mux','zhang_2023','llorens_bobadilla_2023'];ends=['domain_ARI','domain_NMI','marker','svg'];a=np.full((4,3),np.nan)
    for i,e in enumerate(ends):
        for j,f in enumerate(families):a[i,j]=med([r['family_median_effect'] for r in effects if r['endpoint']==e and r['study_family']==f])
    im=heat(ax,a,['ARI','NMI','Markers','SVG'],['Guo','Zhang','Llorens'],-.06,.06,DIV,True);cb(fig,im,ax,'B25 − peak');ax.set_aspect('auto')
    real=['domain_ari','domain_nmi','marker_mean_precision','marker_mean_recall','svg_spearman','svg_jaccard'];rl=['ARI','NMI','M-pre','M-rec','SVG-ρ','SVG-J']
    ax=fig.add_subplot(gs[1,0]);title(ax,'c','Endpoint agreement · real');bubbles(ax,matrix_tau(taus,'real_primary',real),rl)
    ax=fig.add_subplot(gs[1,1]);title(ax,'d','Endpoint agreement · simulations');ends=['domain_ari','domain_nmi','marker_domain_auroc','peak_gene_link_auprc','svg_moran_rank_spearman','svg_jaccard_at_10'];sl=['ARI','NMI','M-AUC','Link','SVG-ρ','SVG-J']
    pairs=[(i,j) for i in range(6) for j in range(i+1,6)];a=np.full((15,2),np.nan)
    for k,st in enumerate(['mechanistic','empirical_semisynthetic']):
        mat=matrix_tau(taus,st,ends)
        for n,(i,j) in enumerate(pairs):a[n,k]=mat[i,j]
    im=heat(ax,a,[sl[i]+' / '+sl[j] for i,j in pairs],['Mech.','Semi-syn.'],-1,1,DIV,True);ax.tick_params(axis='y',labelsize=7.5);cb(fig,im,ax,'Median Kendall τ')
    ax=fig.add_subplot(gs[2,0]);title(ax,'e','Cost of transferring a task choice');a=np.full((6,6),np.nan);raw=np.zeros((6,6))
    for i,e in enumerate(real):
        for j,f in enumerate(real):
            r=next(r for r in loss if r['stratum']=='real_primary' and r['source']==e and r['target']==f);a[i,j]=num(r['mean_relative_loss']);raw[i,j]=num(r['mean_loss'])
    im=heat(ax,a,rl,rl,0,1,WARM);ax.set_xticklabels(rl,rotation=45,ha='right');ax.set_xlabel('Evaluation task');ax.set_ylabel('Selection task');cb(fig,im,ax,'Fraction of task range')
    for i,j in np.ndindex(a.shape):
        v=raw[i,j];text='0' if v==0 else f'{v*1000:.2g}';ax.text(j,i,text,ha='center',va='center',fontsize=6.5,color='white' if a[i,j]>.7 else INK)
    ax.text(.5,1.025,'Loss (×0.001)',transform=ax.transAxes,ha='center',fontsize=8)
    ax=fig.add_subplot(gs[2,1]);title(ax,'f','Leave-study-family-out selection');keys=[(f,e) for f in ['Guo 2025','Zhang 2023','Llorens 2023'] for e in ['ARI','NMI']]
    for i,(family,ep) in enumerate(keys):
        r=next(r for r in cv if r['heldout_family']==family and r['endpoint']==ep)
        for j,(key,c,marker) in enumerate([('loss_mean',PURPLE,'D'),('fixed_B_loss',GREEN,'o'),('uniform_random_loss','#9DA6AE','s')]):
            v=num(r[key]);ax.scatter(v,i+(j-1)*.16,s=30,c=c,marker=marker,edgecolor=INK,lw=.35,zorder=3)
    ax.set_yticks(range(6),[f'{f.split()[0]} · {e}' for f,e in keys]);ax.invert_yaxis();ax.set_xlim(-.002,.058);ax.set_ylim(6.9,-.5);ax.set_xlabel('Held-out loss');ax.grid(axis='x',lw=.4,c='#E2E2E2');ax.legend([Line2D([0],[0],marker=m,c=c,lw=0,markersize=4) for c,m in [(PURPLE,'D'),(GREEN,'o'),('#9DA6AE','s')]],['Train-selected','Fixed B25','Uniform'],fontsize=7,frameon=False,loc='lower right');save(fig,5)

if __name__=='__main__':
    import sys
    for n in [int(x) for x in sys.argv[1:]] or [2,3,4,5]:globals()[f'fig{n}']()
    (O/'plotted_matrices.json').write_text(json.dumps(audit,indent=2));(O/'source_checksums.json').write_text(json.dumps(sources,indent=2))

