#!/usr/bin/env python3

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pygrib

from track import (
    CONFIG, _step_hours, MODEL_ORDER,
    load_ecmwf_ensemble, load_gencast_ensemble, load_lagged_ensemble,
)

# Upper levels
ECMWF_UPPER = "/scratch/apoorva/ECMWF/ecmwf_tigge_nargis_upper_perturbed.grib"
OUT_FIG     = "/scratch/apoorva/work1/tracks/FIG5_DTE.png"

# constants 
CP, TR   = 1004.5, 270.0
KAPPA    = CP / TR                 
LEVELS   = [850, 500, 200]         
BOX_DEG  = 10.0                    
FIT_LO_D = 0.5                     
FIT_HI_D = 3.0                     
MAX_LEAD_D = 5.5
RI_START = datetime(2008, 5, 1, 0, 0)
RI_END   = datetime(2008, 5,  2,  0, 0)

MODEL_STYLE = {
    'ECMWF':        {'c': '#d62728', 'lab': 'ECMWF IFS'},
    'GenCast':      {'c': '#1f77b4', 'lab': 'GenCast'},
    'PanguWeather': {'c': '#ff7f0e', 'lab': 'Pangu-Weather'},
    'FourCastNet':  {'c': '#2ca02c', 'lab': 'FourCastNetv2'},
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.8, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
})


def ensemble_mean_centers(tracks):
    times = set()
    for trk in tracks:
        times.update(trk['times'])
    centers = {}
    for vt in sorted(times):
        las, los = [], []
        for trk in tracks:
            for i, t in enumerate(trk['times']):
                if t == vt:
                    las.append(trk['lats'][i]); los.append(trk['lons'][i])
                    break
        if len(las) >= CONFIG['tracking']['min_members_for_mean']:
            centers[vt] = (float(np.mean(las)), float(np.mean(los)))
    return centers



def _norm_var(shortname):
    s = shortname.lower()
    if s in ('u', 'u component of wind', '10u') and s != '10u':
        return 'u'
    if s == 'u':
        return 'u'
    if s == 'v':
        return 'v'
    if s == 't':
        return 't'
    return None


def _norm_level(msg):
    tol = getattr(msg, 'typeOfLevel', '')
    if tol not in ('isobaricInhPa', 'isobaricInPa'):
        return None
    lvl = msg.level
    if tol == 'isobaricInPa':
        lvl = lvl / 100.0
    lvl = int(round(lvl))
    return lvl if lvl in LEVELS else None


#storm-centered box
def _box_indices(lat1d, lon1d, center):   
    lat_c, lon_c = center
    rows = np.where((lat1d >= lat_c - BOX_DEG) & (lat1d <= lat_c + BOX_DEG))[0]
    cols = np.where((lon1d >= lon_c - BOX_DEG) & (lon1d <= lon_c + BOX_DEG))[0]
    return rows, cols



class BoxStore:
    def __init__(self):
        self.data = {}          
        self.lat1d = None
        self.lon1d = None

    def add(self, vt, level, var, mid, box_rows):
        self.data.setdefault(vt, {}).setdefault(level, {}) \
                 .setdefault(var, {})[mid] = box_rows


def _extract_file(filepath, centers, store, member_id=None, fixed_mid=None,
                  anchor=None):
    if anchor is None:
        anchor = CONFIG['time']['anchor']
    try:
        grbs = pygrib.open(str(filepath))
    except Exception as e:
        print(f"     open failed: {e}")
        return
    for msg in grbs:
        var = _norm_var(msg.shortName)
        if var is None:
            continue
        lvl = _norm_level(msg)
        if lvl is None:
            continue
        if member_id is not None:
            try:
                if msg.perturbationNumber != member_id:
                    continue
            except Exception:
                continue
        step = int(round(_step_hours(msg)))
        vt = anchor + timedelta(hours=step)
        if vt not in centers:
            continue
        if store.lat1d is None:
            lats, lons = msg.latlons()
            store.lat1d = lats[:, 0]
            store.lon1d = lons[0, :]
        rows, cols = _box_indices(store.lat1d, store.lon1d, centers[vt])
        if rows.size == 0 or cols.size == 0:
            continue
        box = msg.values[np.ix_(rows, cols)].astype(np.float32)
        mid = fixed_mid if fixed_mid is not None else member_id
        
        store.add(vt, lvl, var, mid, (box, rows))
    grbs.close()

 
def build_store(model, centers):
    print(f"reading level {model}")
    store = BoxStore()
    anchor = CONFIG['time']['anchor']
    if model == 'ECMWF':
        for mid in range(1, 51):
            _extract_file(ECMWF_UPPER, centers, store, member_id=mid, anchor=anchor)
            if mid % 10 == 0:
                print(f"       ECMWF member {mid}/50")
    elif model == 'GenCast':
        files = sorted(f for f in Path(CONFIG['dirs']['gencast']).glob('gencast_member_*.grib')
                       if not f.name.endswith('.idx'))[:50]
        for k, f in enumerate(files):
            _extract_file(f, centers, store, fixed_mid=k, anchor=anchor)
            if (k + 1) % 10 == 0:
                print(f"       GenCast member {k+1}/{len(files)}")
    else:
        key = 'pangu' if model == 'PanguWeather' else 'fourcast'
        path = Path(CONFIG['dirs'][key])
        mid = 0
        for it in CONFIG['time']['lagged_inits']:
            istr = it.strftime('%Y%m%d%H')
            for m in range(10):
                f = path / f"{istr}_{m}.grib"
                if f.exists():
                    
                    _extract_file(f, centers, store, fixed_mid=mid, anchor=it)
                    mid += 1
            print(f"       {model} init {istr} done")
    return store


# dte = 0.5 * ( var_ens(u) + var_ens(v) + KAPPA *var_ens(T) )
def compute_dte(store):
    out = {lvl: ([], []) for lvl in LEVELS}
    anchor = CONFIG['time']['anchor']
    for vt in sorted(store.data):
        lead = (vt - anchor).total_seconds() / 86400.0
        if lead < 0:
            continue
        for lvl in LEVELS:
            lev = store.data[vt].get(lvl)
            if not lev:
                continue
            if not all(v in lev for v in ('u', 'v', 't')):
                continue
            mids = set(lev['u']) & set(lev['v']) & set(lev['t'])
            if len(mids) < CONFIG['tracking']['min_members_for_mean']:
                continue
            mids = sorted(mids)
            rows = lev['u'][mids[0]][1]   
            cosw = np.cos(np.deg2rad(store.lat1d[rows]))[:, None]

            def stack(var):
                arrays = [lev[var][m][0] for m in mids]
                shape0 = arrays[0].shape
                good = [a for a in arrays if a.shape == shape0]
                if len(good) < CONFIG['tracking']['min_members_for_mean']:
                    return None
                return np.stack(good, axis=0)

            u = stack('u'); v = stack('v'); t = stack('t')
            if u is None or v is None or t is None:
                continue
            var_u = u.var(axis=0)
            var_v = v.var(axis=0)
            var_t = t.var(axis=0)
            dte_field = 0.5 * (var_u + var_v + KAPPA * var_t)
            w = np.broadcast_to(cosw, dte_field.shape)
            dte = float(np.sum(dte_field * w) / np.sum(w))
            out[lvl][0].append(lead)
            out[lvl][1].append(dte)

    res = {}
    for lvl in LEVELS:
        ld = np.array(out[lvl][0]); dt = np.array(out[lvl][1])
        order = np.argsort(ld)
        res[lvl] = (ld[order], dt[order])
    return res

#Exponential fit ln(DTE) = a + sigma*lead
def fit_doubling(lead, dte): 
    m = (lead >= FIT_LO_D) & (lead <= FIT_HI_D) & (dte > 0)
    if np.count_nonzero(m) < 3:
        return np.nan, np.nan, None, None
    x = lead[m]; y = np.log(dte[m])
    sigma, a = np.polyfit(x, y, 1)
    tau2 = np.log(2) / sigma if sigma > 0 else np.nan
    xx = np.linspace(x.min(), x.max(), 50)
    return sigma, tau2, xx, np.exp(a + sigma * xx)



def figure_4(dte_all, fits, output):
    print("  plotting ")
    ri_s = (RI_START - CONFIG['time']['anchor']).total_seconds() / 86400.0
    ri_e = (RI_END   - CONFIG['time']['anchor']).total_seconds() / 86400.0
    fig, axes = plt.subplots(3, 2, figsize=(11, 11),
                             gridspec_kw={'hspace': 0.30, 'wspace': 0.22})
    panel = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]

    for r, lvl in enumerate(LEVELS):
        ax_log, ax_lin = axes[r]
        for ax in (ax_log, ax_lin):
            ax.axvspan(ri_s, ri_e, color='#cccccc', alpha=0.30, zorder=0)
            ax.set_xlim(FIT_LO_D, MAX_LEAD_D)
        for mn in MODEL_ORDER:
            st = MODEL_STYLE[mn]
            ld, dte = dte_all[mn][lvl]
            if len(ld) == 0:
                continue
            disp = ld >= FIT_LO_D     # display from 12 h onward
            ax_log.semilogy(ld[disp], dte[disp], color=st['c'], lw=1.8,
                            marker='o', ms=3, alpha=0.9, label=st['lab'])
            ax_lin.plot(ld[disp], dte[disp], color=st['c'], lw=1.8,
                        marker='o', ms=3, alpha=0.9)
            sigma, tau2, fx, fy = fits[mn][lvl]

                                      #   ax_log.plot(fx, fy, color=st['c'], lw=1.0, ls='--', alpha=0.7)
        ax_log.set_ylabel(f'{lvl} hPa\nDTE (m² s⁻²)', fontsize=9)
        ax_log.set_title(f'{panel[r][0]}  {lvl} hPa — log (growth rate)',
                         fontsize=9, fontweight='bold')
        ax_lin.set_title(f'{panel[r][1]}  {lvl} hPa — linear (magnitude)',
                         fontsize=9, fontweight='bold')
        for ax in (ax_log, ax_lin):
            ax.grid(True, which='both', alpha=0.12, lw=0.4)
        if r == 2:
            ax_log.set_xlabel('Forecast lead time (days)', fontsize=9)
            ax_lin.set_xlabel('Forecast lead time (days)', fontsize=9)
        
        txt = [] # label doubling time
        for mn in MODEL_ORDER:
            _, tau2, _, _ = fits[mn][lvl]
            txt.append((mn, tau2))
        ylo, yhi = ax_log.get_ylim()
        for i, (mn, tau2) in enumerate(txt):
            if np.isfinite(tau2):
                ax_log.text(0.98, 0.30 - i * 0.075,
                            f"τ₂={tau2:.2f} d",
                            transform=ax_log.transAxes, fontsize=7,
                            color=MODEL_STYLE[mn]['c'], fontweight='bold',
                            ha='right', va='top')

    # legend + fit-window note
    handles = [mlines.Line2D([], [], color=MODEL_STYLE[m]['c'], lw=1.8,
                             marker='o', ms=4, label=MODEL_STYLE[m]['lab'])
               for m in MODEL_ORDER]
    handles += [
        plt.Rectangle((0, 0), 1, 1, fc='#cccccc', alpha=0.5, label='RI window'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Difference Total Energy (DTE) ensemble error growth ',
                 fontsize=10.5, fontweight='bold', y=0.97)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output}")
    plt.close()



def print_diagnostics(dte_all, fits):
    sep = '=' * 72
    print(f"\n{sep}\nDTE DOUBLING TIMES τ₂ (d) and GROWTH RATES σ (d⁻¹)\n"
          f"fit window {FIT_LO_D*24:.0f} h-day {FIT_HI_D:.0f};κ = Cp/Tr = {KAPPA:.3f}\n{sep}")
    header = f"  {'Model':14s}" + "".join(f"{lvl} hPa τ₂   σ      " for lvl in LEVELS)
    print(header)
    for mn in MODEL_ORDER:
        row = f"  {mn:14s}"
        for lvl in LEVELS:
            sigma, tau2, _, _ = fits[mn][lvl]
            row += f"{tau2:5.2f}  {sigma:5.2f}    " if np.isfinite(tau2) else "  n/a          "
        print(row)
    print(f"\n  RI-window DTE growth check (500 hPa — does growth accelerate during RI?):")
    ri_s = (RI_START - CONFIG['time']['anchor']).total_seconds() / 86400.0
    ri_e = (RI_END   - CONFIG['time']['anchor']).total_seconds() / 86400.0
    for mn in MODEL_ORDER:
        ld, dte = dte_all[mn][500]
        pre    = (ld >= FIT_LO_D) & (ld < ri_s) & (dte > 0)
        during = (ld >= ri_s) & (ld <= ri_e) & (dte > 0)
        if np.count_nonzero(pre) >= 2 and np.count_nonzero(during) >= 2:
            s_pre = np.polyfit(ld[pre],    np.log(dte[pre]),    1)[0]
            s_ri  = np.polyfit(ld[during], np.log(dte[during]), 1)[0]
            flag  = "ACCELERATES" if s_ri > 1.3 * s_pre else "no acceleration"
            print(f"    {mn:14s}: σ_pre={s_pre:.2f}  σ_RI={s_ri:.2f}  → {flag}")



def main():
    print("=" * 72)
    print("FIGURE 4 — DIFFERENCE TOTAL ENERGY (DTE)  | tracker: track.py")
    print("=" * 72)

    print("\nTracking members ")
    tracks = {
        'ECMWF':        load_ecmwf_ensemble(),
        'GenCast':      load_gencast_ensemble(),
        'PanguWeather': load_lagged_ensemble('PanguWeather', 'pangu'),
        'FourCastNet':  load_lagged_ensemble('FourCastNet',  'fourcast'),
    }
    centers = {m: ensemble_mean_centers(tracks[m]) for m in MODEL_ORDER}

    dte_all, fits = {}, {}
    for mn in MODEL_ORDER:
        store = build_store(mn, centers[mn])
        dte_all[mn] = compute_dte(store)
        fits[mn] = {lvl: fit_doubling(*dte_all[mn][lvl]) for lvl in LEVELS}
        del store   

    print_diagnostics(dte_all, fits)
    figure_4(dte_all, fits, OUT_FIG)
    print("\nCompleted\n")


if __name__ == "__main__":
    main()
