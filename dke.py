#!/usr/bin/env python3
"""
 Figure 6: 500 hPa KE and DKE spectra for TC Nargis (2008)

"""

from datetime import datetime, timedelta
from pathlib import Path
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pygrib

warnings.filterwarnings('ignore')

# details from track.py
from track import CONFIG, _step_hours, MODEL_ORDER

ECMWF_UPPER = "/scratch/apoorva/ECMWF/ecmwf_tigge_nargis_upper_perturbed.grib"
OUT_FIG     = "/scratch/apoorva/work1/codes/tracks/FIG6_SPECTRA.png"
CACHE       = OUT_FIG.replace('.png', '_data.pkl')


LEVEL      = 500      
HALF_DEG   = 10.0     
LEAD_HOURS = [24, 48, 96, 120, 144]
MAX_STEP   = 240      
DEG2KM     = 111.32   
 
MESO_LO  =   150.0    
MESO_HI  =  500.0    
SYNOP_LO =  500.0    
SYNOP_HI = 2000.0    
BREAK_KM =  500.0    


LAT_MIN = -90.0
LAT_MAX =  90.0
LON_MIN =   0.0
LON_MAX = 360.0


MODEL_STYLE = {
    'ECMWF':        {'lab': 'ECMWF IFS',    'color': '#d62728', 'p': '(a)'},
    'GenCast':      {'lab': 'GenCast',       'color': '#1f77b4', 'p': '(b)'},
    'PanguWeather': {'lab': 'Pangu-Weather', 'color': '#ff7f0e', 'p': '(c)'},
    'FourCastNet':  {'lab': 'FourCastNetv2', 'color': '#2ca02c', 'p': '(d)'},
}


LEAD_COLORS = {
    24:  '#2166ac',
    48:  '#4dac26',
    96:  '#d6604d',
    120: '#b2182b',
    144: '#000000',
}

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['DejaVu Serif'],
    'font.size':        9,
    'axes.linewidth':   0.8,
    'mathtext.fontset': 'dejavuserif',
})



def load_jtwc():
    
    pos = {}
    with open(CONFIG['dirs']['jtwc']) as f:
        for line in f:
            p = [x.strip() for x in line.split(',')]
            if len(p) < 10:
                continue
            try:
                dt = datetime.strptime(p[2][:10], '%Y%m%d%H')
            except Exception:
                continue
            las, los = p[6], p[7]
            lat = float(las[:-1]) / 10.0 * (-1 if las[-1] == 'S' else 1)
            lon = float(los[:-1]) / 10.0 * (-1 if los[-1] == 'W' else 1)
            pos[dt] = (lat, lon)
    return pos


def interp_center(jtwc, vt):
    if vt in jtwc:
        return jtwc[vt]
    times = sorted(jtwc)
    for i in range(len(times) - 1):
        if times[i] <= vt <= times[i + 1]:
            f = ((vt - times[i]).total_seconds() /
                 max((times[i + 1] - times[i]).total_seconds(), 1))
            a, b = jtwc[times[i]], jtwc[times[i + 1]]
            return (a[0] + f * (b[0] - a[0]),
                    a[1] + f * (b[1] - a[1]))
    return jtwc[min(times, key=lambda t: abs((t - vt).total_seconds()))]


#preventing spectral leakage from the box edges
def _hann2d(ny, nx):
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    w = np.outer(wy, wx)
    w /= np.sqrt(np.mean(w ** 2))   
    return w


def _field_spectrum(field, dx_km):
    
#2D FTT used   
    f = np.asarray(field, dtype=np.float64)
    ny, nx = f.shape
    n = min(ny, nx)
    f = f[:n, :n]
    f = f - f.mean()          
    f *= _hann2d(n, n)        

    F = np.fft.fftshift(np.fft.fft2(f))
    P = (np.abs(F) ** 2) / (n * n)


    ky = np.fft.fftshift(np.fft.fftfreq(n))
    kx = np.fft.fftshift(np.fft.fftfreq(n))
    KX, KY = np.meshgrid(kx, ky)
    Kr = np.sqrt(KX ** 2 + KY ** 2)

    #
    kbins = np.arange(0.5, n // 2) / n
    Pr = np.zeros(len(kbins) - 1)
    kc = np.zeros(len(kbins) - 1)
    for i in range(len(kbins) - 1):
        m = (Kr >= kbins[i]) & (Kr < kbins[i + 1])
        if m.any():
            Pr[i] = P[m].mean()
            kc[i] = 0.5 * (kbins[i] + kbins[i + 1])

    good = kc > 0
    kc, Pr = kc[good], Pr[good]

    k_per_km = kc / dx_km         
    wl = 1.0 / k_per_km           #
    order = np.argsort(wl)[::-1]  # large to small scale
    return wl[order], Pr[order]


def ke_and_dke_spectra(u_members, v_members, dx_km):
    U = np.stack(u_members, 0)
    V = np.stack(v_members, 0)
    umean = U.mean(0)
    vmean = V.mean(0)
    M = U.shape[0]

    wl_ref, ke_acc, dke_acc = None, None, None
    for m in range(M):
        wl, Pu  = _field_spectrum(U[m], dx_km)
        _,  Pv  = _field_spectrum(V[m], dx_km)
        _,  Pdu = _field_spectrum(U[m] - umean, dx_km)
        _,  Pdv = _field_spectrum(V[m] - vmean, dx_km)
        if wl_ref is None:
            wl_ref  = wl
            ke_acc  = np.zeros_like(Pu)
            dke_acc = np.zeros_like(Pu)
        if Pu.shape != ke_acc.shape:
            continue    # skip shape mismatch 
        ke_acc  += 0.5 * (Pu + Pv)
        dke_acc += 0.5 * (Pdu + Pdv)

    return wl_ref, ke_acc / M, dke_acc / M


def fit_slope(wl, E, lo, hi):
    m = (wl >= lo) & (wl <= hi) & np.isfinite(E) & (E > 0)
    if np.count_nonzero(m) < 3:
        return np.nan
    k = 1.0 / wl[m]
    p = np.polyfit(np.log10(k), np.log10(E[m]), 1)
    return float(p[0])


# Scale where model KE falls below reference
def effective_resolution(wl, E_ke, anchor_wl=MESO_HI,
                         ref_slope=-5.0 / 3.0, factor=0.33):
    
    k = 1.0 / wl
    ai = int(np.argmin(np.abs(wl - anchor_wl)))
    if E_ke[ai] <= 0:
        return np.nan
    C = E_ke[ai] / ((1.0 / wl[ai]) ** ref_slope)
    expected = C * (k ** ref_slope)
    short = (wl < wl[ai]) & (E_ke > 0)
    if not np.any(short):
        return np.nan
    deficit = E_ke[short] / expected[short]
    below = wl[short][deficit < factor]
    return float(np.max(below)) if below.size else np.nan


def saturation_scale(wl, E_ke, E_dke, frac=0.5):
    
    ok = np.isfinite(E_dke) & np.isfinite(E_ke) & (E_ke > 0)
    sat = ok & (E_dke >= frac * E_ke)
    return float(np.max(wl[sat])) if sat.any() else np.nan


def meso_energy_ratio(wl, E_ke, wl_ref, E_ref,
                      lo=MESO_LO, hi=MESO_HI):
    #mesoscale KE vs ECMWF.

    shared = np.array([w for w in wl if any(np.abs(w - wl_ref) < 1.0)])
    if len(shared) < 2:
        return np.nan

    def band_val(w_grid, E_grid):
        m = (np.isin(np.round(w_grid, 1), np.round(shared, 1)) &
             (w_grid >= lo) & (w_grid <= hi))
        if np.count_nonzero(m) < 2:
            return np.nan
        k = 1.0 / w_grid[m]
        o = np.argsort(k)
        trapfn = (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)
        return trapfn(E_grid[m][o], k[o])

    num = band_val(wl, E_ke)
    den = band_val(wl_ref, E_ref)
    return (float(num / den)
            if (np.isfinite(num) and np.isfinite(den) and den > 0)
            else np.nan)

def effective_resolution_ref(wl, E_ke, wl_ref, E_ref, factor=0.5):
    
    ok_wl = np.array([w for w in wl
                      if any(np.abs(w - wl_ref) < 1.0) and w < MESO_HI])
    if len(ok_wl) < 2:
        return np.nan

    def get_E(w_grid, E_grid, w):
        idx = np.argmin(np.abs(w_grid - w))
        return (E_grid[idx] if np.abs(w_grid[idx] - w) < 1.0 else np.nan)

    below = [w for w in ok_wl
             if get_E(wl, E_ke, w) < factor * get_E(wl_ref, E_ref, w)]
    return float(np.max(below)) if below else np.nan


#upper level 
def _ecmwf_boxes(jtwc):

    anchor = CONFIG['time']['anchor']
    centers = {lh: interp_center(jtwc, anchor + timedelta(hours=lh))
               for lh in LEAD_HOURS}
    tmp = {}
    lat1d = lon1d = None

    try:
        grbs = pygrib.open(str(ECMWF_UPPER))
    except Exception as e:
        print(f"   ECMWF open failed: {e}")
        return {}

    n = 0
    for msg in grbs:
        sn = msg.shortName.lower()
        if sn not in ('u', 'v'):
            continue
        tol = getattr(msg, 'typeOfLevel', '')
        if tol not in ('isobaricInhPa', 'isobaricInPa'):
            continue
        lvl = msg.level / (100.0 if tol == 'isobaricInPa' else 1.0)
        if int(round(lvl)) != LEVEL:
            continue
        sh = _step_hours(msg)
        lh = min(LEAD_HOURS, key=lambda L: abs(L - sh))
        if abs(lh - sh) > 3:
            continue
        try:
            mid = msg.perturbationNumber
        except Exception:
            continue
        if not (1 <= mid <= 50):
            continue
        if lat1d is None:
            lats, lons = msg.latlons()
            lat1d, lon1d = lats[:, 0], lons[0, :]
        clat, clon = centers[lh]
        li = np.where((lat1d >= clat - HALF_DEG) &
                      (lat1d <= clat + HALF_DEG))[0]
        lo = np.where((lon1d >= clon - HALF_DEG) &
                      (lon1d <= clon + HALF_DEG))[0]
        if len(li) < 10 or len(lo) < 10:
            continue
        sub = msg.values[li[0]:li[-1] + 1,
                         lo[0]:lo[-1] + 1].astype(np.float32)
        dlat = abs(lat1d[1] - lat1d[0])
        dlon = abs(lon1d[1] - lon1d[0])
        dx = (dlon * DEG2KM * np.cos(np.radians(clat)) + dlat * DEG2KM) / 2.0
        d = tmp.setdefault((lh, mid), {})
        d['u' if sn == 'u' else 'v'] = sub
        d['dx'] = dx
        n += 1
        if n % 100 == 0:
            print(f" ECMWF: {n} u/v messages binned ")

    grbs.close()
    boxes = {lh: {} for lh in LEAD_HOURS}
    for (lh, mid), d in tmp.items():
        if 'u' in d and 'v' in d:
            boxes[lh][mid] = (d['u'], d['v'], d['dx'])
    return boxes


#resding u and v
def _collect_uv(fp, step_h, member_id=None):
    try:
        grbs = pygrib.open(str(fp))
    except Exception:
        return None
    kw = dict(shortName='u', level=LEVEL)
    if member_id is not None:
        kw['perturbationNumber'] = member_id
    try:
        us = grbs.select(**kw)
    except Exception:
        us = []
    if not us:
        grbs.close()
        return None
    bu = min(us, key=lambda m: abs(_step_hours(m) - step_h))
    if abs(_step_hours(bu) - step_h) > 6:
        grbs.close()
        return None
    kw2 = dict(shortName='v', level=LEVEL)
    if member_id is not None:
        kw2['perturbationNumber'] = member_id
    try:
        vs = grbs.select(**kw2)
    except Exception:
        vs = []
    bv = next((m for m in vs
               if abs(_step_hours(m) - _step_hours(bu)) < 1), None)
    if bv is None:
        grbs.close()
        return None
    uf, vf = bu.values.copy(), bv.values.copy()
    lats, lons = bu.latlons()
    grbs.close()
    return uf, vf, lats, lons


def _subdomain(field, lats, lons, clat, clon):
    lat1d = lats[:, 0]
    lon1d = lons[0, :]
    li = np.where((lat1d >= clat - HALF_DEG) &
                  (lat1d <= clat + HALF_DEG))[0]
    lo = np.where((lon1d >= clon - HALF_DEG) &
                  (lon1d <= clon + HALF_DEG))[0]
    if len(li) < 10 or len(lo) < 10:
        return None, None
    sub = field[li[0]:li[-1] + 1, lo[0]:lo[-1] + 1]
    dlat = abs(lat1d[1] - lat1d[0])
    dlon = abs(lon1d[1] - lon1d[0])
    dx = (dlon * DEG2KM * np.cos(np.radians(clat)) + dlat * DEG2KM) / 2.0
    return sub, dx


# Compute KE/DKE spectra 
def model_spectra(model, jtwc, ecmwf_boxes=None):
    anchor = CONFIG['time']['anchor']
    out = {}

    for lh in LEAD_HOURS:
        vt = anchor + timedelta(hours=lh)
        clat, clon = interp_center(jtwc, vt)

        # checking 20°x20° box 
        if (clat - HALF_DEG < LAT_MIN or clat + HALF_DEG > LAT_MAX or
                clon - HALF_DEG < LON_MIN or clon + HALF_DEG > LON_MAX):
            print(f"     T+{lh}h: skipped (box clips domain edge)")
            continue

        u_list, v_list, dx_used = [], [], None

        
        if model == 'ECMWF':
            for mid, (us, vs, dx) in (ecmwf_boxes.get(lh, {})
                                      if ecmwf_boxes else {}).items():
                n = min(us.shape)
                u_list.append(us[:n, :n])
                v_list.append(vs[:n, :n])
                dx_used = dx

        # 
        elif model == 'GenCast':
            files = sorted(
                f for f in Path(CONFIG['dirs']['gencast']).glob(
                    'gencast_member_*.grib')
                if not f.name.endswith('.idx'))[:50]
            for fp in files:
                got = _collect_uv(fp, lh)
                if got is None:
                    continue
                uf, vf, lats, lons = got
                us, dx = _subdomain(uf, lats, lons, clat, clon)
                vs, _  = _subdomain(vf, lats, lons, clat, clon)
                if us is None or dx is None or dx <= 0:
                    continue
                n = min(us.shape)
                u_list.append(us[:n, :n])
                v_list.append(vs[:n, :n])
                dx_used = dx

        # 
        else:
            key = 'pangu' if model == 'PanguWeather' else 'fourcast'
            path = Path(CONFIG['dirs'][key])
            for it in CONFIG['time']['lagged_inits']:
                istr = it.strftime('%Y%m%d%H')
                sh = (vt - it).total_seconds() / 3600.0
                if sh < 0 or sh > MAX_STEP:
                    continue
                for m in range(10):
                    fp = path / f"{istr}_{m}.grib"
                    if not fp.exists():
                        continue
                    got = _collect_uv(fp, sh)
                    if got is None:
                        continue
                    uf, vf, lats, lons = got
                    us, dx = _subdomain(uf, lats, lons, clat, clon)
                    vs, _  = _subdomain(vf, lats, lons, clat, clon)
                    if us is None or dx is None or dx <= 0:
                        continue
                    n = min(us.shape)
                    u_list.append(us[:n, :n])
                    v_list.append(vs[:n, :n])
                    dx_used = dx

        if len(u_list) < 5:
            print(f"     T+{lh}h: <5 members — skip")
            continue

        #
        nmin = min(min(a.shape) for a in u_list)
        u_list = [a[:nmin, :nmin] for a in u_list]
        v_list = [a[:nmin, :nmin] for a in v_list]

        wl, E_ke, E_dke = ke_and_dke_spectra(u_list, v_list, dx_used)
        eff  = effective_resolution(wl, E_ke)
        sat  = saturation_scale(wl, E_ke, E_dke)
        beta = fit_slope(wl, E_ke, MESO_LO, MESO_HI)

        out[lh] = {
            'wl':       wl,
            'KE':       E_ke,
            'DKE':      E_dke,
            'eff_res':  eff,
            'sat':      sat,
            'beta_meso': beta,
        }
        print(f"     T+{lh}h: {len(u_list):2d} members  "
              f"beta_meso={beta:.2f}  eff_res~{eff:.0f}km  sat~{sat:.0f}km")
    return out


def print_diagnostics(spectra_all):
    print("\n" + "=" * 78)
    print("MESOSCALE KE DIAGNOSTICS "
          "Bonavita 2024; Li et al. 2025)")
    print(f"  KE band: {MESO_LO:.0f}–{MESO_HI:.0f} km | "
          f"DKE saturation")
    print("=" * 78)
    hdr = (f"  {'Model':14s}{'lead':>6}{'beta_meso':>11}"
           f"{'mesoKE/ECMWF':>14}{'eff_res_ref(km)':>16}{'sat(km)':>10}")
    print(hdr)

    ref_model = spectra_all.get('ECMWF', {})
    for mn in MODEL_ORDER:
        for lh in sorted(spectra_all.get(mn, {})):
            s   = spectra_all[mn][lh]
            ref = ref_model.get(lh)

            beta = s.get('beta_meso', np.nan)
            sat  = s.get('sat', np.nan)

            if ref is None:
                ratio = eff = np.nan
            else:
                ratio = meso_energy_ratio(
                    s['wl'], s['KE'], ref['wl'], ref['KE'])
                eff = effective_resolution_ref(
                    s['wl'], s['KE'], ref['wl'], ref['KE'])

            bs = f"{beta:.2f}" if np.isfinite(beta) else "  —"
            rs = f"{ratio:.2f}" if np.isfinite(ratio) else "  —"
            es = f"{eff:.0f}"  if np.isfinite(eff)   else "  —"
            ss = f"{sat:.0f}"  if np.isfinite(sat)   else "  —"
            print(f"  {mn:14s}{lh:>6}{bs:>11}{rs:>14}{es:>16}{ss:>10}")

    print()
    print("  beta_meso : mesoscale slope ; canonical = -5/3 = -1.67")
    print("  mesoKE/ECMWF : band-integrated KE ratio ")
    print("  eff_res_ref  : scale where model KE < 0.5 x ECMWF ")
    print("  sat: DKE saturation scale (Selz & Craig 2023)")
    print("  ECMWF rows are reference (ratio=1.00, eff_res_ref='—')")


#

def figure_6(spectra_all, output):
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                             sharex=True, sharey=True)

    # Anchor the reference lines to ECMWF's KE 
    ecmwf_ref = spectra_all.get('ECMWF', {})
    ref_anchor = None
    if ecmwf_ref:
        last_lh = sorted(ecmwf_ref)[-1]
        wlr = ecmwf_ref[last_lh]['wl']
        Er  = ecmwf_ref[last_lh]['KE']
        j   = np.argmin(np.abs(wlr - BREAK_KM))
        ref_anchor = Er[j]   # KE amplitude at the spectral break

    for idx, mn in enumerate(MODEL_ORDER):
        ax   = axes.flat[idx]
        ms   = MODEL_STYLE[mn]
        specs = spectra_all.get(mn, {})

        
        # KE curves 
        for lh in sorted(specs):
            s = specs[lh]
            col = LEAD_COLORS.get(lh, '#333333')
            ax.loglog(s['wl'], s['KE'],
                      color=col, lw=1.2, alpha=0.80,
                      label=f'KE T+{lh}h')
            
            ax.loglog(s['wl'], s['DKE'],
                      color=col, lw=1.4, ls='--', alpha=0.85,
                      label=f'DKE T+{lh}h')

        
        if specs:
            last = sorted(specs)[-1]
            beta = specs[last].get('beta_meso', np.nan)
            if np.isfinite(beta):
                ax.text(0.05, 0.08,
                        rf'$\beta_{{meso}}={beta:.2f}$' + '\n'
                        r'(ref: $-5/3=-1.67$)',
                        transform=ax.transAxes, fontsize=9,
                        fontweight='bold', color=ms['color'],
                        bbox=dict(facecolor='white', alpha=0.8,
                                  edgecolor='none', pad=1.5))

        

        
        ax.axvline(BREAK_KM, color='grey', ls='-', lw=0.7, alpha=0.45,
                   label='Spectral break (~500 km)')
        ax.axvspan(MESO_LO, MESO_HI,
                   color='orange', alpha=0.06, label='Mesoscale fit band')

        # panel labels
        ax.text(0.97, 0.97, ms['p'], transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='right', va='top')
        ax.set_title(ms['lab'], fontsize=12, fontweight='bold', pad=8)
        ax.invert_xaxis()   

        r, c = divmod(idx, 2)
        if r == 1:
            ax.set_xlabel('Wavelength (km)', fontsize=10)
        if c == 0:
            ax.set_ylabel(r'Spectral density (m$^2$ s$^{-2}$ per cycle km$^{-1}$)',
                          fontsize=9)
        ax.set_xlim(SYNOP_HI, MESO_LO)
        ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.4)

    #
    h, l = axes.flat[0].get_legend_handles_labels()

    seen, uh, ul = set(), [], []
    for hi, li in zip(h, l):
        if li not in seen:
            seen.add(li); uh.append(hi); ul.append(li)
    fig.legend(uh, ul, loc='lower center', ncol=4,
               fontsize=7.5, frameon=True, edgecolor='0.7',
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        '500 hPa kinetic-energy and difference-kinetic-energy spectra\n',
        fontsize=11, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Figure saved: {output}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 64)
    print("FIGURE 6 — KE / DKE SPECTRA  TC Nargis")
    print("Hann FFT | Dual ref slopes | beta fit 80-500 km | DKE saturation")
    print("=" * 64)

    # Quick path assertions so the run fails fast, not after an hour
    assert Path(ECMWF_UPPER).exists(), \
        f"ECMWF upper file not found:\n  {ECMWF_UPPER}"
    assert Path(OUT_FIG).parent.exists(), \
        f"Output directory does not exist:\n  {Path(OUT_FIG).parent}"

    print(f"\n  Spectral constants:")
    print(f"    Mesoscale band : {MESO_LO}–{MESO_HI} km")
    print(f"    Synoptic band  : {SYNOP_LO}–{SYNOP_HI} km")
    print(f"    Spectral break : {BREAK_KM} km ")
    print(f"    Domain bounds  : lat {LAT_MIN}–{LAT_MAX}, "
          f"lon {LON_MIN}–{LON_MAX}")

    jtwc = load_jtwc()
    print(f"\n  JTWC positions loaded: {len(jtwc)}")

    print("\n Pre-reading ECMWF upper-level GRIB")
    ecmwf_boxes = _ecmwf_boxes(jtwc)

    spectra_all = {}
    for mn in MODEL_ORDER:
        spectra_all[mn] = model_spectra(mn, jtwc, ecmwf_boxes)
        with open(CACHE, 'wb') as fh:
            pickle.dump(spectra_all, fh)
        print(f"  Checkpoint saved after {mn}")

    print_diagnostics(spectra_all)
    figure_6(spectra_all, OUT_FIG)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
