#!/usr/bin/env python3


from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

# reusing the tracker and loaders from the track script 
from track import (
    CONFIG,
    track_single_member,           
    load_ecmwf_ensemble,
    load_gencast_ensemble,
    load_lagged_ensemble,
    MODEL_ORDER,
)

# 
RI_START = datetime(2008, 5, 1, 0, 0)   
RI_END   = datetime(2008, 5,  2, 0, 0)
OUTPUT = "/scratch/apoorva/work1/codes/tracks/FIG2_INTENSITY.png"

YLIM = (920, 1015)   #hPa

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.linewidth':    1.2,
    'axes.labelsize':    13,
    'axes.titlesize':    15,
    'xtick.labelsize':   10,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'legend.framealpha': 0.95,
    'grid.linewidth':    0.8,
})



def load_jtwc_intensity():
    print("Loading JTWC best-track MSLP")
    try:
        with open(CONFIG['dirs']['jtwc'], 'r') as fh:
            lines = fh.readlines()
    except Exception as e:
        print(f"     ERROR: {e}")
        return None

    rows = {}
    for line in lines:
        parts = line.split(',')
        if len(parts) < 10:
            continue
        try:
            dt = datetime.strptime(parts[2].strip()[:10], '%Y%m%d%H')
            if not (CONFIG['time']['start'] <= dt <= CONFIG['time']['end']):
                continue
            mslp_v = float(parts[9].strip())
            if 800 < mslp_v < 1050:
                rows[dt] = mslp_v
        except Exception:
            continue

    if not rows:
        return None
    times = sorted(rows.keys())
    return {'times': times,
            'mslp':  np.array([rows[t] for t in times])}


def compute_mslp_stats(tracks, min_members=None, synoptic_12h=True):
    if min_members is None:
        min_members = CONFIG['tracking']['min_members_for_mean']

    all_times = set()
    for trk in tracks:
        all_times.update(trk['times'])

    times, mean, p25, p75, lo, hi, n = [], [], [], [], [], [], []
    for vt in sorted(all_times):
        if synoptic_12h and vt.hour not in (0, 12):
            continue
        vals = []
        for trk in tracks:
            if vt in trk['times']:
                idx = trk['times'].index(vt)
                vals.append(float(trk['mslp'][idx]))
        if len(vals) >= min_members:
            v = np.array(vals)
            times.append(vt)
            mean.append(float(np.mean(v)))
            p25.append(float(np.percentile(v, 25)))
            p75.append(float(np.percentile(v, 75)))
            lo.append(float(np.min(v)))
            hi.append(float(np.max(v)))
            n.append(len(v))

    return {'times': times,
            'mean':  np.array(mean),
            'p25':   np.array(p25),
            'p75':   np.array(p75),
            'min':   np.array(lo),
            'max':   np.array(hi),
            'n':     np.array(n)}


def _jtwc_12h_markers(jtwc):
    m_t, m_p = [], []
    for t, p in zip(jtwc['times'], jtwc['mslp']):
        if t.hour in (0, 12):
            m_t.append(t)
            m_p.append(p)
    return m_t, np.array(m_p)


def plot_figure(model_data, jtwc, output):
    print(f"\n{'='*72}\nCREATING FIGURE 1b: MSLP TIME SERIES\n{'='*72}")

    anchor = CONFIG['time']['anchor']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                             sharex=True, sharey=True)

    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        conf  = CONFIG['models'][model_name]
        stats = model_data[model_name]['stats']
        print(f"  Panel: {model_name}  "
              f"({len(stats['times'])} valid times)")

        ax.set_facecolor('#fafafa')
        ax.grid(True, ls='--', alpha=0.4, color='gray')

        
        ax.axvspan(RI_START, RI_END, color='grey', alpha=0.18,
                   zorder=0, label='RI window')

        if len(stats['times']) >= 2:
            t = stats['times']
            # min-max band, then IQR on top, then the mean line
            ax.fill_between(t, stats['min'], stats['max'],
                            color=conf['color'], alpha=0.15,
                            zorder=2, label='Full ensemble range')
            
            ax.fill_between(t, stats['p25'], stats['p75'],
                            color=conf['color'], alpha=0.35,
                            zorder=3, label='Interquartile range')
            
            ax.plot(t, stats['mean'], color='white', lw=4, zorder=4)
            ax.plot(t, stats['mean'], color=conf['color'], lw=2.5,
                    zorder=5, label='Ensemble mean')

        
        if jtwc is not None:
            ax.plot(jtwc['times'], jtwc['mslp'],
                    color='k', lw=2.5, zorder=6, label='JTWC best-track')
            m_t, m_p = _jtwc_12h_markers(jtwc)
            if len(m_p) > 0:
                ax.plot(m_t, m_p, ls='none', marker='s',
                        color='gold', ms=8, mec='k', mew=1.3, zorder=7)

        ax.set_title(conf['label'], fontsize=15, fontweight='bold', pad=10)
        ax.set_ylim(*YLIM)
        ax.set_xlim(CONFIG['time']['start'], CONFIG['time']['end'])
        # date label at 00Z, "12Z" at midday
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: mdates.num2date(x).strftime('%d %b')
            if mdates.num2date(x).hour == 0 else '12Z'))
        ax.tick_params(axis='x', rotation=45, labelsize=9)

         # force tick labels back on for the inner panels
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_ylabel('MSLP (hPa)', fontweight='bold', fontsize=11)
        ax.set_xlabel('Valid time (2008)', fontweight='bold', fontsize=11)

    
    handles = [
        mlines.Line2D([], [], color='k', lw=2.5, marker='s',
                      mfc='gold', mec='k', ms=8, label='JTWC best-track'),
        mlines.Line2D([], [], color='#555555', lw=2.5,
                      label='Ensemble mean'),
        mpatches.Patch(color='#555555', alpha=0.35,
                       label='Interquartile range'),
        mpatches.Patch(color='#555555', alpha=0.15,
                       label='Full ensemble range'),
        mpatches.Patch(color='grey', alpha=0.18, label='RI window'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=11, frameon=True, edgecolor='0.7',
               bbox_to_anchor=(0.5, -0.01))

    init_str = anchor.strftime('%Y-%m-%d %HZ')
    fig.suptitle(
        f"Multi-Model Ensemble Intensity (MSLP): TC Nargis (2008)\n"
        f"Primary Initialization: {init_str}",
        fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output}")
    plt.close()


# 
def print_diagnostics(model_data, jtwc):
    print(f"\n{'='*72}\nINTENSITY DIAGNOSTICS\n{'='*72}")

    # 3 cyclone lifecycle timestamps used 
    snaps = {'Pre-RI  28Apr12Z': datetime(2008, 4, 28, 12),
             'RI      01May00Z': datetime(2008, 5, 1, 0),
             'Near - Landfall 02May00Z': datetime(2008, 5, 2, 0)}
    jt = dict(zip(jtwc['times'], jtwc['mslp'])) if jtwc else {}

    for label, t in snaps.items():
        obs = jt.get(t, np.nan)
        print(f"\n  {label}  (JTWC = {obs:.0f} hPa)")
        for mn in MODEL_ORDER:
            stats = model_data[mn]['stats']
            if t in stats['times']:
                i = stats['times'].index(t)
                bias = stats['mean'][i] - obs
                print(f"    {mn:14s}: mean={stats['mean'][i]:6.1f} hPa  "
                      f"bias={bias:+5.1f}  "
                      f"IQR=[{stats['p25'][i]:.0f},{stats['p75'][i]:.0f}]  "
                      f"n={stats['n'][i]}")
            else:
                print(f"    {mn:14s}: no data at this time")



def main():
    print("=" * 72)
    print("FIGURE 2: TC NARGIS (2008) ENSEMBLE MSLP TIME SERIES")
    print("Tracker imported from track.py by valid time")
    print("=" * 72)

    model_data = {}

    tracks = load_ecmwf_ensemble()
    model_data['ECMWF'] = {'tracks': tracks,
                           'stats':  compute_mslp_stats(tracks)}

    tracks = load_gencast_ensemble()
    model_data['GenCast'] = {'tracks': tracks,
                             'stats':  compute_mslp_stats(tracks)}

    tracks = load_lagged_ensemble('PanguWeather', 'pangu')
    model_data['PanguWeather'] = {'tracks': tracks,
                                  'stats':  compute_mslp_stats(tracks)}

    tracks = load_lagged_ensemble('FourCastNet', 'fourcast')
    model_data['FourCastNet'] = {'tracks': tracks,
                                 'stats':  compute_mslp_stats(tracks)}

    jtwc = load_jtwc_intensity()
    print_diagnostics(model_data, jtwc)
    plot_figure(model_data, jtwc, OUTPUT)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
