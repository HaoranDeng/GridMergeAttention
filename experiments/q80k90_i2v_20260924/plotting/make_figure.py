"""Combine the existing T2V and I2V figures without changing experimental data."""
import copy
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, LogLocator, NullLocator

HERE = Path(__file__).resolve().parent
SCAN = HERE.parent
PAPER = Path('C:/GMA/GMA-ICLR2027/figure_sources/tradeoff')
I2V = SCAN/'Q80K90_I2V_C4_C6_C8_C10_20260924/tradeoff'
COLORS = {'SVG-EAR':'#2a78d6','SVG2':'#eb6834','SVG':'#1baf7a','GCA (ours)':'#d62728'}
MARKERS = {'SVG-EAR':'o','SVG2':'s','SVG':'^','GCA (ours)':'*'}
PANELS = [('psnr','flops'),('psnr','runtime'),('ssim','flops'),('ssim','runtime')]
GUIDE_RATIOS = {'flops': (6,5,4,3), 'runtime': (6,5,4,3,2)}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def draw(data, stem, sensitivity):
    fig, axes = plt.subplots(2,4,figsize=(14,8.0))
    fig.subplots_adjust(left=.07,right=.99,bottom=.13,top=.89,wspace=.33,hspace=.90)
    row_data = [data['T2V'], data['I2V']]
    for row, (task, datum) in enumerate(zip(('T2V','I2V'),row_data)):
        for ax, (metric,cost) in zip(axes[row],PANELS):
            if cost=='runtime':ax.set_xscale('log',base=10)
            ax.set_axisbelow(True);ax.grid(color='#e5e5e5',linewidth=.65)
            for side in ('top','right'):ax.spines[side].set_visible(False)
            for side in ('left','bottom'):ax.spines[side].set_color('#c8c8c8')
            xs,ys=[],[]
            for method in COLORS:
                points=[p for p in datum['series'][method] if p.get(cost) is not None]
                # Preserve each figure's established connection rule: T2V is
                # cost-sorted; I2V is parameter-ordered in both cost panels.
                if task=='T2V':points=sorted(points,key=lambda p:p[cost])
                xs.extend(p[cost] for p in points);ys.extend(p[metric] for p in points)
                ax.plot([p[cost] for p in points],[p[metric] for p in points],
                        color=COLORS[method],marker=MARKERS[method],linewidth=1.5,
                        markersize=8.5 if method=='GCA (ours)' else 5,
                        markerfacecolor=COLORS[method],markeredgecolor='white',markeredgewidth=.65,
                        zorder=5 if method=='GCA (ours)' else 3)
            ratios=GUIDE_RATIOS[cost]
            dense=datum['dense'][cost]
            positions=[dense/r for r in ratios]
            for pos in positions:ax.axvline(pos,color='#9a9a96',linewidth=.75,linestyle=(0,(3,3)),zorder=1)
            top=ax.secondary_xaxis('top');top.set_xticks(positions,labels=[f'{r}×' for r in ratios])
            top.tick_params(labelsize=8,length=0,pad=3)
            top.set_xlabel('Arithmetic reduction vs Dense' if cost=='flops' else 'Speedup vs Dense',fontsize=8,labelpad=5)
            top.spines['top'].set_color('#c8c8c8');top.xaxis.set_minor_locator(NullLocator())
            if cost=='runtime':
                ax.set_xlim(data['shared_runtime_limits_ms'])
                ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,1.5,2,3,4,6,8)))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.xaxis.set_minor_locator(NullLocator())
            else:
                ax.set_xlim(data['shared_arithmetic_limits_tflops_per_call'])
                ax.xaxis.set_major_locator(MaxNLocator(4))
            span=max(ys)-min(ys);ax.set_ylim(min(ys)-span*.12,max(ys)+span*.12)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if metric=='ssim':ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(labelsize=9,length=0,pad=3,colors='#454545')
            ax.set_ylabel('PSNR (dB)' if metric=='psnr' else 'SSIM',fontsize=10,labelpad=5)
            xlabel='Attention arithmetic\n(TFLOPs / call)' if cost=='flops' else 'Attention latency\n(ms per attention layer)'
            ax.set_xlabel(xlabel,fontsize=9.2,labelpad=5)
    for row,label in enumerate(['T2V','I2V (excluding 019)' if sensitivity else 'I2V']):
        bounds=axes[row,0].get_position()
        fig.text(.07,bounds.y1+.077,label,ha='left',va='bottom',fontsize=11,fontweight='bold',color='#292929')
    handles=[Line2D([0],[0],color=COLORS[m],marker=MARKERS[m],linewidth=1.6,
                    markersize=8 if m=='GCA (ours)' else 5,label=m) for m in COLORS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.014),ncol=4,frameon=False,fontsize=10,columnspacing=2.8)
    for ext in ('pdf','png','svg'):fig.savefig(HERE/f'{stem}.{ext}',dpi=220,facecolor='white')
    assert [ax.get_xscale() for ax in axes.ravel()]==['linear','log','linear','log']*2
    plt.close(fig)

def main():
    paths={'t2v_data':PAPER/'data.json','t2v_notes':PAPER/'README.md','i2v_data':I2V/'data.json',
           'i2v_sensitivity':I2V/'sensitivity_drop019_vs_ear_p08/data.json',
           'runtime_overrides':HERE/'runtime_overrides.json'}
    hashes={name:digest(path) for name,path in paths.items()}
    sources=HERE/'sources';sources.mkdir(exist_ok=True)
    for name,path in paths.items():
        (sources/(name+path.suffix)).write_bytes(path.read_bytes())
    t2v=read(paths['t2v_data']);full=read(paths['i2v_data']);subset=read(paths['i2v_sensitivity'])
    overrides=read(paths['runtime_overrides'])
    assert overrides['task']=='I2V' and overrides['unit']=='ms per attention layer'
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    manifest={'layout':'2 rows x 4 columns; T2V above I2V','sources':{k:{'path':str(paths[k]),'sha256':v} for k,v in hashes.items()},'outputs':[]}
    for suffix,raw in [('full20',full),('i2v_exclude019',subset)]:
        t=copy.deepcopy(t2v);i=copy.deepcopy(raw)
        applied=[]
        for point in i['series']['GCA (ours)']:
            setting=point['setting']
            if setting in overrides['GCA (ours)']:
                point['measured_runtime_ms_before_user_override']=point['runtime']
                point['runtime']=overrides['GCA (ours)'][setting]
                point['runtime_override_source']=overrides['source']
                applied.append(setting)
        i['runtime_plot_overrides']={'applied':applied,
            'pending_settings':sorted(set(overrides['GCA (ours)'])-set(applied)),
            'values_ms':overrides['GCA (ours)'],'source':overrides['source']}
        calls=i['protocol']['attention_calls_per_video']
        assert calls==4000
        pf_per_video_to_tf_per_call=1000/calls
        for rows in i['series'].values():
            for point in rows:
                point['source_full_50_step_pflops']=point['flops']
                point['flops']*=pf_per_video_to_tf_per_call
        i['dense']={'flops':i['dense_full_video_pflops']*pf_per_video_to_tf_per_call,'runtime':i['dense_runtime_reference_ms']}
        i['plotted_arithmetic_unit']='TFLOPs per call, averaged over the complete50-step schedule'
        values=[p['runtime'] for d in (t,i) for rows in d['series'].values() for p in rows if p.get('runtime') is not None]
        values += [d['dense']['runtime']/r for d in (t,i) for r in GUIDE_RATIOS['runtime']]
        limits=[min(values)/1.06,max(values)*1.045]
        costs=[p['flops'] for d in (t,i) for rows in d['series'].values() for p in rows if p.get('flops') is not None]
        costs += [d['dense']['flops']/r for d in (t,i) for r in GUIDE_RATIOS['flops']]
        lo,hi=min(costs),max(costs);pad=(hi-lo)*.07
        data={'T2V':t,'I2V':i,'axis_scales':{'runtime':'log10','flops':'linear'},'shared_runtime_limits_ms':limits,
              'runtime_axis_label':'Attention latency (ms per attention layer)',
              'display_notes':{'footer_notes':False,'log_scale_text':False,'suptitle':False,'sample_counts':False},
              'shared_arithmetic_limits_tflops_per_call':[lo-pad,hi+pad],
              'displayed_dense_guides':GUIDE_RATIOS,
              'arithmetic_units':{'T2V':'TFLOPs per call, preserved historical figure scope','I2V':'TFLOPs per call, full50-step average'},
              'i2v_arithmetic_conversion':{'input_unit':'PFLOPs per full50-step video','output_unit':'TFLOPs per call','calls_per_video':calls,'factor':pf_per_video_to_tf_per_call},
              'scope_note':'Historical T2V reference costs are not extrapolated into unverified full50 totals. No times or quality values pooled across tasks or GPUs.',
              'source_hashes':hashes,'paper_modified':False}
        stem='t2v_i2v_tradeoff_'+suffix
        (HERE/(stem+'_data.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        draw(data,stem,suffix!='full20')
        assert data['T2V']['series']==t2v['series']
        for method,rows in data['I2V']['series'].items():
            for converted,source in zip(rows,raw['series'][method]):
                restored=copy.deepcopy(converted)
                restored['flops']=restored.pop('source_full_50_step_pflops')
                if 'measured_runtime_ms_before_user_override' in restored:
                    restored['runtime']=restored.pop('measured_runtime_ms_before_user_override')
                    restored.pop('runtime_override_source')
                assert restored==source and math.isclose(converted['flops']*calls/1000,source['flops'],rel_tol=1e-12)
        manifest['outputs'].append({'stem':stem,'panels':8,'i2v_n':raw['baseline_n'],
            'series_points':{'T2V':{m:len(v) for m,v in t['series'].items()},'I2V':{m:len(v) for m,v in i['series'].items()}},
            'all_source_quantities_preserved':True,'runtime_plot_overrides':i['runtime_plot_overrides'],
            'i2v_flops_conversion_factor':pf_per_video_to_tf_per_call,'pdf_sha256':digest(HERE/(stem+'.pdf'))})
    assert all(digest(paths[k])==v for k,v in hashes.items())
    manifest['all_sources_unchanged']=True
    (HERE/'verification.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (HERE/'README.md').write_text('''# T2V and I2V combined trade-off figures

Two rows by four columns: T2V above I2V; PSNR/arithmetic, PSNR/runtime, SSIM/arithmetic, SSIM/runtime. The global title and sample-count labels are omitted. Row labels are T2V and I2V; the sensitivity version says I2V (excluding019). Runtime axes retain log10 with shared millisecond limits; the displayed unit is ms per attention layer, with no log-scale annotation. The two footer-note lines are omitted from the graphic; their provenance remains here and in data.json. GCA is red; one shared legend. Runtime Dense2x/3x/4x/5x/6x and arithmetic3x/4x/5x/6x guides retain their own task references.

`t2v_i2v_tradeoff_full20` uses complete20 results in both rows. `t2v_i2v_tradeoff_i2v_exclude019` keeps complete20 T2V and uses the explicitly labeled, common19-sample I2V sensitivity analysis. These are independent outputs; paper files and original individual figures remain unchanged.

All quality coordinates, point membership and existing line connection rules are preserved from the current sources. T2V GCA runtime coordinates remain the user's53.9/79.5/100.0/131.7ms values, with original measured times retained in source provenance. I2V uses per-GPU measured times except user-specified plotting coordinates in runtime_overrides.json: C6=83.6, C8=104.1 and C10=133.8 ms per attention layer. These apply to both complete20 and exclude019 figures; original measured runtime is retained in each affected point. A setting without a quality point remains pending; its runtime override is applied when the point is added. C12 stays excluded; LPIPS is not plotted.

Both rows show TFLOPs/call on a shared arithmetic axis range. I2V is converted from its measured/derived full50-step PFLOPs/video by multiplying1000/4000=0.25, and each original total is preserved as source_full_50_step_pflops. Dense arithmetic and guide positions receive the same conversion. The per-call I2V values are full50-step averages, not short-schedule extrapolations. Historical T2V per-call costs are retained at their original scope. SVG-EAR clustering upper bounds and mixed T2V campaign limitations remain documented in the copied source audits. Comparison panels do not pool datasets or establish matched cross-task/cross-GPU speedups.

Source snapshots and hashes are saved under `sources/` and `verification.json`. Rebuild locally with `python make_figure.py`.
''',encoding='utf-8')
    print(json.dumps(manifest['outputs']))

if __name__=='__main__':main()
