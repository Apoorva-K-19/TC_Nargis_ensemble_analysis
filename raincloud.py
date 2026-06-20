#!/usr/bin/env python3

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

from track import (
    CONFIG,
    haversine_km,
    load_ecmwf_ensemble,
    load_gencast_ensemble,
    load_lagged_ensemble,
    MODEL_ORDER,
)

OUTPUT = "/scratch/apoorva/work1/codes/tracks/FIG3_RAINCLOUD.png"


SNAPSHOTS = {
    'Pre-RI\n(28 Apr 12Z)':               datetime(2008, 4, 28, 12, 0),
    'Rapid-Intensification\n(01 May 12Z)': datetime(2008, 5,  1, 12, 0),
    'Near-landfall\n(02 May 00Z)':         datetime(2008, 5,  2,  0, 0),
}
TOL_H = 6   

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.linewidth':    0.9,
    'axes.labelsize':    9,
    'axes.titlesize':    10,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
})


def load_jtwc_full():
    print("Loading JTWC best track (position, MSLP)")
    try:
        with open(CONFIG['dirs']['jtwc'], 'r') as fh:
            lines = fh.readlines()
    except Exception as e:
        print(f"     ERROR: {e}")
        return {}

    rows = {}
    for line in lines:
        parts = line.split(',')
        if len(parts) < 10:
            continue
        try:
            dt = datetime.strptime(parts[2].strip()[:10], '%Y%m%d%H')
            if not (CONFIG['time']['start'] <= dt <= CONFIG['time']['end']):
                continue
            lat_s = parts[6].strip()
            lon_s = parts[7].strip()
            lat = float(lat_s[:-1]) / 10.0 * (-1 if lat_s.endswith('S') else 1)
            lon = float(lon_s[:-1]) / 10.0 * (-1 if lon_s.endswith('W') else 1)
            mslp = float(parts[9].strip())
            if 800 < mslp < 1050:
                rows[dt] = (lat, lon, mslp)
        except Exception:
            continue
    return rows


def _jtwc_at(jtwc, target_time, tol_h=TOL_H):
    if not jtwc:
        return None
    best, best_dt = None, None
    for t, v in jtwc.items():
        dh = abs((t - target_time).total_seconds()) / 3600.0
        if dh <= tol_h and (best_dt is None or dh < best_dt):
            best_dt, best = dh, v
    return best


def _member_value_at(trk, target_time, key, tol_h=TOL_H):
    times = trk['times']
    best_i, best_dh = None, None
    for i, t in enumerate(times):
        dh = abs((t - target_time).total_seconds()) / 3600.0
        if dh <= tol_h and (best_dh is None or dh < best_dh):
            best_dh, best_i = dh, i
    if best_i is None:
        return None
    if key == 'pos':
        return (trk['lats'][best_i], trk['lons'][best_i])
    return float(trk[key][best_i])


def mslp_bias_at(tracks, jtwc_mslp, target_time):
    out = []
    for trk in tracks:
        v = _member_value_at(trk, target_time, 'mslp')
        if v is not None and 850 < v < 1050:
            out.append(v - jtwc_mslp)
    return np.array(out)


def track_error_at(tracks, jtwc_lat, jtwc_lon, target_time):
    out = []
    for trk in tracks:
        pos = _member_value_at(trk, target_time, 'pos')
        if pos is not None and not (np.isnan(pos[0]) or np.isnan(pos[1])):
            out.append(haversine_km(jtwc_lat, jtwc_lon, pos[0], pos[1]))
    return np.array(out)

def draw_half_violin(ax, data, pos, color, width=0.30):
    if len(data) < 5:
        return
    try:
        kde = gaussian_kde(data)
    except Exception:
        return
    pad = max(2.0, 0.05 * (np.ptp(data) + 1e-9))
    yg = np.linspace(data.min() - pad, data.max() + pad, 200)
    d = kde(yg)
    d = d / d.max() * width
    vx = pos + 0.06
    ax.fill_betweenx(yg, vx, vx + d, color=color, alpha=0.25,
                     edgecolor='none', zorder=2)
    ax.plot(vx + d, yg, color=color, lw=0.8, alpha=0.7, zorder=2)


def draw_box(ax, data, pos, color, width=0.14):
    if len(data) < 5:
        return
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    iqr = q75 - q25
    wlo = max(data.min(), q25 - 1.5 * iqr)
    whi = min(data.max(), q75 + 1.5 * iqr)
    box = FancyBboxPatch((pos - width / 2, q25), width, q75 - q25,
                         boxstyle="round,pad=0.005", facecolor=color,
                         edgecolor='black', alpha=0.55, linewidth=0.7,
                         zorder=5)
    ax.add_patch(box)
    ax.plot([pos - width / 2, pos + width / 2], [q50, q50],
            color='white', lw=2.0, zorder=6, solid_capstyle='round')
    ax.plot([pos - width / 2, pos + width / 2], [q50, q50],
            color='black', lw=1.0, zorder=7, solid_capstyle='round')
    ax.plot([pos, pos], [wlo, q25], color='black', lw=0.7, zorder=4)
    ax.plot([pos, pos], [q75, whi], color='black', lw=0.7, zorder=4)
    cw = width * 0.4
    for yy in (wlo, whi):
        ax.plot([pos - cw / 2, pos + cw / 2], [yy, yy],
                color='black', lw=0.7, zorder=4)


def draw_points(ax, data, pos, color, offset=-0.22, jitter=0.10,
                size=7, alpha=0.45):
    if len(data) == 0:
        return
    j = np.random.uniform(-jitter / 2, jitter / 2, size=len(data))
    ax.scatter(pos + offset + j, data, s=size, c=color,
               alpha=alpha, edgecolors='none', zorder=3)


def draw_raincloud(ax, data, pos, color):
    draw_points(ax, data, pos, color)
    draw_box(ax, data, pos, color)
    draw_half_violin(ax, data, pos, color)


def annotate(ax, data, pos, color, fmt, idx=0):
    if len(data) < 2:
        return
    mean, sd = np.mean(data), np.std(data)
    yl = ax.get_ylim()
    span = yl[1] - yl[0]
    #
    base = np.percentile(data, 95) + 0.04 * (np.ptp(data) + 5)
    stagger = (0.075 if (idx % 2) else 0.0) * span
    ann_y = base + stagger
    ann_y = min(ann_y, yl[0] + 0.97 * span)
    ann_y = max(ann_y, mean + sd + 0.02 * span)
    ax.text(pos, ann_y, fmt.format(mean=mean, sd=sd),
            fontsize=7, ha='center', va='bottom', color=color,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.4))


def create_figure(bias_data, track_data, obs, output):
    print(f"\n{'='*72}\nCREATING FIGURE 3 RAINCLOUD\n{'='*72}")

    labels = list(SNAPSHOTS.keys())
    n_models = len(MODEL_ORDER)
    panel = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5),
                             gridspec_kw={'hspace': 0.42, 'wspace': 0.08})
    for col in range(1, 3):
        axes[0, col].sharey(axes[0, 0])
        axes[1, col].sharey(axes[1, 0])

    #ylimits
    all_bias = np.concatenate([bias_data[c][m] for c in range(3)
                               for m in MODEL_ORDER if len(bias_data[c][m])])
    all_trk = np.concatenate([track_data[c][m] for c in range(3)
                              for m in MODEL_ORDER if len(track_data[c][m])])
    bias_ylim = (min(all_bias.min() - 5, -15), all_bias.max() + 14)
    trk_ylim = (-20, all_trk.max() + 90)

    for col in range(3):
        ax = axes[0, col]
        ax.set_facecolor('#fafafa')
        ax.grid(True, axis='y', ls='-', alpha=0.2, lw=0.3, color='grey')
        ax.axhline(0, color='black', lw=1.2, alpha=0.8, zorder=8)
        ax.axhspan(30, bias_ylim[1] + 10, alpha=0.05, color='red', zorder=0)
        ax.set_ylim(*bias_ylim)

        for i, mn in enumerate(MODEL_ORDER):
            d = bias_data[col][mn]
            if len(d):
                draw_raincloud(ax, d, i, CONFIG['models'][mn]['color'])
        for i, mn in enumerate(MODEL_ORDER):
            d = bias_data[col][mn]
            if len(d):
                annotate(ax, d, i, CONFIG['models'][mn]['color'],
                         '+{mean:.0f}\u00b1{sd:.0f}', idx=i)

        o = obs[col][2]
        ax.text(0.03, 0.04, f'JTWC: {o:.0f} hPa', transform=ax.transAxes,
                fontsize=7, fontweight='bold', ha='left', va='bottom',
                bbox=dict(facecolor='gold', alpha=0.75, edgecolor='black',
                          lw=0.5, boxstyle='round,pad=0.2'))
        if col == 2:
            ax.text(0.97, 0.40, 'Severe\nunderestimation',
                    transform=ax.transAxes, fontsize=6, color='darkred',
                    alpha=0.6, style='italic', ha='right', va='center')

        ax.set_xlim(-0.6, n_models - 0.2)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([CONFIG['models'][m]['label'] for m in MODEL_ORDER],
                           rotation=35, ha='right', fontsize=7)
        ax.set_title(labels[col], fontsize=9, fontweight='bold', pad=6)
        ax.text(0.04, 0.97, panel[col], transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top', ha='left')
        if col > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
    axes[0, 0].set_ylabel('MSLP bias (hPa)\n(model \u2212 JTWC; + = too weak)',
                          fontsize=8.5)

    
    for col in range(3):
        ax = axes[1, col]
        ax.set_facecolor('#fafafa')
        ax.grid(True, axis='y', ls='-', alpha=0.2, lw=0.3, color='grey')
        ax.axhline(0, color='black', lw=1.2, alpha=0.8, zorder=8)
        ax.set_ylim(*trk_ylim)

        for i, mn in enumerate(MODEL_ORDER):
            d = track_data[col][mn]
            if len(d):
                draw_raincloud(ax, d, i, CONFIG['models'][mn]['color'])
        for i, mn in enumerate(MODEL_ORDER):
            d = track_data[col][mn]
            if len(d):
                annotate(ax, d, i, CONFIG['models'][mn]['color'],
                         '{mean:.0f}\u00b1{sd:.0f}', idx=i)

        ax.set_xlim(-0.6, n_models - 0.2)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([CONFIG['models'][m]['label'] for m in MODEL_ORDER],
                           rotation=35, ha='right', fontsize=7)
        ax.set_title(labels[col], fontsize=9, fontweight='bold', pad=6)
        ax.text(0.04, 0.97, panel[col + 3], transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top', ha='left')
        if col > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
    axes[1, 0].set_ylabel('Track error (km)\n(distance from JTWC position)',
                          fontsize=8.5)

    
    handles = [
        mlines.Line2D([], [], color='black', lw=1.2,
                      label='JTWC observed (zero line)'),
        mlines.Line2D([], [], marker='o', color='grey', lw=0, ms=4, alpha=0.5,
                      label='Individual members'),
        mpatches.Patch(facecolor='grey', alpha=0.55, edgecolor='black',
                       lw=0.6, label='IQR (box)'),
        mpatches.Patch(facecolor='grey', alpha=0.25, label='KDE (half-violin)'),
        mpatches.Patch(facecolor='red', alpha=0.10, label='Severe (>30 hPa)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
               frameon=True, edgecolor='0.7', bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Ensemble MSLP bias and track-position error at TC lifecycle phases',
                 fontsize=11, fontweight='bold', y=1.0)

    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white',
                pad_inches=0.15)
    print(f"  Saved: {output}")
    plt.close()

def print_diagnostics(bias_data, track_data, obs):
    print(f"\n{'='*72}\nRAINCLOUD DIAGNOSTICS\n{'='*72}")
    labels = [l.replace('\n', ' ') for l in SNAPSHOTS.keys()]
    for col in range(3):
        lat, lon, mslp = obs[col]
        print(f"\n  {labels[col]}  —  JTWC {mslp:.0f} hPa, ({lat:.1f}N, {lon:.1f}E)")
        for mn in MODEL_ORDER:
            b = bias_data[col][mn]
            t = track_data[col][mn]
            bstr = (f"bias +{np.mean(b):.1f}\u00b1{np.std(b):.1f} (n={len(b)})"
                    if len(b) else "no MSLP")
            tstr = (f"track {np.mean(t):.0f}\u00b1{np.std(t):.0f} km (n={len(t)})"
                    if len(t) else "no track")
            print(f"    {mn:14s}: {bstr:32s} | {tstr}")


def main():
    print("=" * 72)
    print("FIGURE 3: TC NARGIS (2008) RAINCLOUD (BIAS, TRACK ERROR)")
    print("Tracker imported from track.py | No Savgol | Snapshots from Table")
    print("=" * 72)

    tracks = {
        'ECMWF':        load_ecmwf_ensemble(),
        'GenCast':      load_gencast_ensemble(),
        'PanguWeather': load_lagged_ensemble('PanguWeather', 'pangu'),
        'FourCastNet':  load_lagged_ensemble('FourCastNet', 'fourcast'),
    }
    jtwc = load_jtwc_full()

    snap_times = list(SNAPSHOTS.values())
    bias_data = {c: {} for c in range(3)}
    track_data = {c: {} for c in range(3)}
    obs = {}

    for c, t in enumerate(snap_times):
        jv = _jtwc_at(jtwc, t)
        if jv is None:
            raise RuntimeError(f"No JTWC record near {t}")
        jlat, jlon, jmslp = jv
        obs[c] = (jlat, jlon, jmslp)
        for mn in MODEL_ORDER:
            bias_data[c][mn] = mslp_bias_at(tracks[mn], jmslp, t)
            track_data[c][mn] = track_error_at(tracks[mn], jlat, jlon, t)

    print_diagnostics(bias_data, track_data, obs)
    create_figure(bias_data, track_data, obs, OUTPUT)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
