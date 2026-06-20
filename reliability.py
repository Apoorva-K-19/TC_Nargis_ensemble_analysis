#!/usr/bin/env python3

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from track import (
    CONFIG,
    haversine_km,
    load_ecmwf_ensemble,
    load_gencast_ensemble,
    load_lagged_ensemble,
    MODEL_ORDER,
)

OUT_4 = "/scratch/apoorva/work1/codes/tracks/FIG4_RELIABILITY.png"

VERIF_CADENCE_H = 12
START_LEAD_H    = 12          
MAX_LEAD_D      = 5.5

RI_START = datetime(2008, 5, 1, 0, 0)
RI_END   = datetime(2008, 5, 2, 0, 0)

MODEL_STYLE = {
    'ECMWF':        {'c': '#d62728', 'm': 'o', 'lab': 'ECMWF IFS'},
    'GenCast':      {'c': '#1f77b4', 'm': 's', 'lab': 'GenCast'},
    'PanguWeather': {'c': '#ff7f0e', 'm': '^', 'lab': 'Pangu-Weather'},
    'FourCastNet':  {'c': '#2ca02c', 'm': 'D', 'lab': 'FourCastNetv2'},
}

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       9,
    'axes.linewidth':  0.8,
    'axes.labelsize':  9,
    'axes.titlesize':  9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})


def load_all_tracks():
    print(" Loading tracks")
    return {
        'ECMWF':        load_ecmwf_ensemble(),
        'GenCast':      load_gencast_ensemble(),
        'PanguWeather': load_lagged_ensemble('PanguWeather', 'pangu'),
        'FourCastNet':  load_lagged_ensemble('FourCastNet',  'fourcast'),
    }


def load_jtwc():
    print(" Loading JTWC")
    out = {}
    with open(CONFIG['dirs']['jtwc']) as fh:
        for line in fh:
            p = line.split(',')
            if len(p) < 10:
                continue
            try:
                dt = datetime.strptime(p[2].strip()[:10], '%Y%m%d%H')
                if not (CONFIG['time']['start'] <= dt <= CONFIG['time']['end']):
                    continue
                las, los = p[6].strip(), p[7].strip()
                lat = float(las[:-1]) / 10.0 * (-1 if las.endswith('S') else 1)
                lon = float(los[:-1]) / 10.0 * (-1 if los.endswith('W') else 1)
                mslp = float(p[9].strip())
                if 800 < mslp < 1050:
                    out[dt] = {'lat': lat, 'lon': lon, 'mslp': mslp}
            except Exception:
                continue
    print(f"     {len(out)} JTWC records")
    return out


def verif_times():
    ts = []
    t = CONFIG['time']['anchor'] + timedelta(hours=START_LEAD_H)
    end = CONFIG['time']['anchor'] + timedelta(days=MAX_LEAD_D)
    while t <= min(end, CONFIG['time']['end']):
        ts.append(t)
        t += timedelta(hours=VERIF_CADENCE_H)
    return ts


def _nearest_jtwc(jtwc, vt, tol_h=6):
    best, best_dh = None, None
    for t, v in jtwc.items():
        dh = abs((t - vt).total_seconds()) / 3600.0
        if dh <= tol_h and (best_dh is None or dh < best_dh):
            best_dh, best = dh, v
    return best


def _member_mslp_at(trk, vt, tol_h=6):
    for i, t in enumerate(trk['times']):
        if abs((t - vt).total_seconds()) / 3600.0 <= tol_h:
            return float(trk['mslp'][i])
    return None


def _member_pos_at(trk, vt, tol_h=6):
    for i, t in enumerate(trk['times']):
        if abs((t - vt).total_seconds()) / 3600.0 <= tol_h:
            return (float(trk['lats'][i]), float(trk['lons'][i]))
    return None


def compute_spread_skill(tracks, jtwc, variable):
    leads, spreads, errors = [], [], []
    for vt in verif_times():
        lead = (vt - CONFIG['time']['anchor']).total_seconds() / 86400.0
        obs = _nearest_jtwc(jtwc, vt)
        if obs is None:
            continue
        if variable == 'mslp':
            vals = [v for v in (_member_mslp_at(t, vt) for t in tracks)
                    if v is not None]
            if len(vals) < 2 or np.isnan(obs['mslp']):
                continue
            vals = np.array(vals)
            spr = float(np.std(vals, ddof=1))
            err = float(np.sqrt(np.mean((vals - obs['mslp']) ** 2)))
        else:
            pos = [p for p in (_member_pos_at(t, vt) for t in tracks)
                   if p is not None]
            if len(pos) < 2:
                continue
            mla = np.mean([p[0] for p in pos])
            mlo = np.mean([p[1] for p in pos])
            spr = float(np.sqrt(np.mean(
                [haversine_km(mla, mlo, la, lo) ** 2 for la, lo in pos])))
            err = float(np.sqrt(np.mean(
                [haversine_km(obs['lat'], obs['lon'], la, lo) ** 2
                 for la, lo in pos])))
        if err > 0:
            leads.append(lead); spreads.append(spr); errors.append(err)

    leads, spreads, errors = (np.array(x) for x in (leads, spreads, errors))
    order = np.argsort(leads)
    leads, spreads, errors = leads[order], spreads[order], errors[order]
    ratio = np.where(errors > 0, spreads / errors, np.nan)
    return leads, spreads, errors, ratio


def figure_4(ss, output):
    print("  Building Figure 4 (2 panels)")
    ri_s = (RI_START - CONFIG['time']['anchor']).total_seconds() / 86400.0
    ri_e = (RI_END   - CONFIG['time']['anchor']).total_seconds() / 86400.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.18, wspace=0.22)

    def ratio_panel(ax, var, title):
        ax.axhline(1.0, color='#555', lw=1.0, ls='--', alpha=0.85)
        for mn in MODEL_ORDER:
            st = MODEL_STYLE[mn]
            ld, _, _, rat = ss[mn][var]
            if not len(ld):
                continue
            ax.plot(ld, rat, color=st['c'], marker=st['m'],
                    ms=4, lw=1.7, alpha=0.9)
        ax.axvspan(ri_s, ri_e, color='#cccccc', alpha=0.30, zorder=0)
        ax.set_xlim(0.5, MAX_LEAD_D)
        ax.set_ylim(0, 1.4)
        ax.set_xlabel('Forecast lead time (days)')
        ax.set_ylabel('Spread / Error')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.13, lw=0.4)

    ratio_panel(axes[0], 'mslp',  '(a) MSLP spread–error ratio')
    ratio_panel(axes[1], 'track', '(b) Track spread–error ratio')

    handles = [
        mlines.Line2D([], [], color=MODEL_STYLE[m]['c'],
                      marker=MODEL_STYLE[m]['m'], lw=1.7,
                      label=MODEL_STYLE[m]['lab'])
        for m in MODEL_ORDER
    ] + [
        mlines.Line2D([], [], color='#555', ls='--', lw=1.0,
                      label='Reliable (ratio = 1)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 1.00))
    fig.suptitle('Ensemble reliability — TC Nargis (2008)',
                 fontsize=11.5, fontweight='bold', y=1.08)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output}")
    plt.close()


def print_diagnostics(ss):
    ri_s = (RI_START - CONFIG['time']['anchor']).total_seconds() / 86400.0
    ri_e = (RI_END   - CONFIG['time']['anchor']).total_seconds() / 86400.0
    sep = '=' * 72
    print(f"\n{sep}\nSPREAD-SKILL RATIOS (from T+{START_LEAD_H}h)\n{sep}")
    for var in ('mslp', 'track'):
        print(f"\n  {var.upper()}")
        for mn in MODEL_ORDER:
            ld, sp, er, rat = ss[mn][var]
            if not len(ld):
                print(f"    {mn:14s}: no data"); continue
            m = (ld >= ri_s) & (ld <= ri_e)
            ri_r = float(np.nanmedian(rat[m])) if np.any(m) else np.nan
            print(f"    {mn:14s}: median={np.nanmedian(rat):.3f}  "
                  f"RI-median={ri_r:.3f}  n={len(ld)}")


def main():
    print("=" * 72)
    print("FIGURE 4 — ENSEMBLE RELIABILITY (2 panels)")
    print("=" * 72)
    Path(OUT_4).parent.mkdir(parents=True, exist_ok=True)

    tracks = load_all_tracks()
    jtwc   = load_jtwc()

    print("\nComputing spread-skill")
    ss = {m: {v: compute_spread_skill(tracks[m], jtwc, v)
              for v in ('mslp', 'track')}
          for m in MODEL_ORDER}

    print_diagnostics(ss)
    print("\nRendering")
    figure_4(ss, OUT_4)
    print("\nCompleted.\n")


if __name__ == "__main__":
    main()
