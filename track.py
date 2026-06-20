#!/usr/bin/env python3

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygrib
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')


CONFIG = {
    'dirs': {
        'gencast':  "/scratch/apoorva/gencast",
        'pangu':    "/scratch/apoorva/panguweather",
        'fourcast': "/scratch/apoorva/fourcastnet",
        
        'ecmwf':    "/scratch/apoorva/ECMWF/ecmwf_tigge_nargis_surface_perturbed.grib",
        'jtwc':     "/scratch/apoorva/JTWC/bio012008.txt",
        'output_tracks':  "/scratch/apoorva/work1/codes/tracks/FIG1a_TRACKS.png",
    },
    'time': {
                
        'anchor': datetime(2008, 4, 26, 12, 0),
        
        'start':  datetime(2008, 4, 26, 12, 0),
        'end':    datetime(2008, 5,  3,  0, 0),
        
        'lagged_inits': [
            datetime(2008, 4, 25, 12, 0),
            datetime(2008, 4, 25, 18, 0),
            datetime(2008, 4, 26,  0, 0),
            datetime(2008, 4, 26,  6, 0),
            datetime(2008, 4, 26, 12, 0),
        ],
    },

    
    'tracking': {
        
        'init_box': {'lat_min':  10, 'lat_max': 14,
                     'lon_min': 85, 'lon_max': 89},

          
        'field_smooth_sigma':   1.5,   
        'raw_refine_radius':    1.0,   

        
        'search_radius_6h':   2.5,     
        'search_radius_12h':  3.5,     

        
        'max_translation_ms': 20.0,    
        'pressure_threshold': 1010.0,  

        
        'max_consecutive_misses': 2,   
        'retry_expansion_factor': 1.0, 

        
        'min_track_length':   4,       
        
        'min_members_for_mean': 5,     
    },

    
    'display': {
        'landfall_lon': 94.8,
        'landfall_lat': 20.0,
        'domain': [80, 100, 6, 24],
        'figsize': (20, 16),
        'dpi':     300,
        'member_alpha':     0.15,
        'member_linewidth': 1.0,
    },

    
    
    'models': {
        'ECMWF':        {'color': '#d62728', 'marker': 'D',
                         'label': 'ECMWF IFS'},
        'GenCast':      {'color': '#1f77b4', 'marker': 'o',
                         'label': 'GenCast'},
        'PanguWeather': {'color': '#ff7f0e', 'marker': 's',
                         'label': 'Pangu-Weather'},
        'FourCastNet':  {'color': '#2ca02c', 'marker': '^',
                         'label': 'FourCastNetv2'},
    },
}

MODEL_ORDER = ['ECMWF', 'GenCast', 'PanguWeather', 'FourCastNet']

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.linewidth':    1.5,
    'axes.labelsize':    13,
    'axes.titlesize':    15,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'legend.framealpha': 0.95,
    'grid.linewidth':    1.0,
    'lines.linewidth':   2.0,
})
# 28 Apr :pre-RI, 1 May:RI, 2 May: near landfall
PHASE_MARKERS = [
    (datetime(2008, 4, 28, 12), '4/28'),   
    (datetime(2008, 5,  1,  0), '5/1'),     
    (datetime(2008, 5,  2,  0), '5/2'),    
]

def _pos_at_time(times, lats, lons, vt, tol_h=3.0):
    best, bd = None, None
    for i, t in enumerate(times):
        dh = abs((t - vt).total_seconds()) / 3600.0
        if dh <= tol_h and (bd is None or dh < bd):
            bd, best = dh, (float(lats[i]), float(lons[i]))
    return best


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _step_hours(msg):
    sh = msg.step
    if hasattr(sh, 'total_seconds'):
        return sh.total_seconds() / 3600.0
    return float(sh)


def _expand_box(box, expansion):
    cl = 0.5 * (box['lat_min'] + box['lat_max'])
    co = 0.5 * (box['lon_min'] + box['lon_max'])
    hl = expansion * 0.5 * (box['lat_max'] - box['lat_min'])
    ho = expansion * 0.5 * (box['lon_max'] - box['lon_min'])
    return {'lat_min': cl - hl, 'lat_max': cl + hl,
            'lon_min': co - ho, 'lon_max': co + ho}


def _update_search_box(lat, lon, radius_deg):
    return {'lat_min': lat - radius_deg, 'lat_max': lat + radius_deg,
            'lon_min': lon - radius_deg, 'lon_max': lon + radius_deg}


def find_pressure_minimum(mslp, lats, lons, search_box,
                          field_sigma, raw_refine_radius):
    
    mask = ((lats >= search_box['lat_min']) &
            (lats <= search_box['lat_max']) &
            (lons >= search_box['lon_min']) &
            (lons <= search_box['lon_max']))
    if not np.any(mask):
        return None

    #smoothing first to get a rough cnetre adn then refine on the raw field

    mslp_smooth = gaussian_filter(mslp, sigma=field_sigma)
    s1_idx = np.unravel_index(
        np.argmin(np.where(mask, mslp_smooth, np.inf)),
        mslp.shape)
    c_lat = float(lats[s1_idx])
    c_lon = float(lons[s1_idx])

    # 
    rm = ((lats >= c_lat - raw_refine_radius) &
          (lats <= c_lat + raw_refine_radius) &
          (lons >= c_lon - raw_refine_radius) &
          (lons <= c_lon + raw_refine_radius))
    if not np.any(rm):
        return {'lat': c_lat, 'lon': c_lon,
                'mslp': float(mslp[s1_idx])}

    s2_idx = np.unravel_index(
        np.argmin(np.where(rm, mslp, np.inf)),
        mslp.shape)
    return {'lat':  float(lats[s2_idx]),
            'lon':  float(lons[s2_idx]),
            'mslp': float(mslp[s2_idx])}



def track_single_member(filepath, init_time, is_ecmwf=False, member_id=None):
    
    cfg = CONFIG['tracking']
    sigma   = cfg['field_smooth_sigma']
    refine  = cfg['raw_refine_radius']
    sr_6h   = cfg['search_radius_6h']
    sr_12h  = cfg['search_radius_12h']
    vmax_ms = cfg['max_translation_ms']
    p_thr   = cfg['pressure_threshold']
    max_miss = cfg['max_consecutive_misses']
    retry_x  = cfg['retry_expansion_factor']
    min_len  = cfg['min_track_length']

    try:
        grbs = pygrib.open(str(filepath))

        
        msgs = []
        if is_ecmwf:
            for var in ['msl', 'Mean sea level pressure']:
                try:
                    found = grbs.select(shortName=var,
                                        perturbationNumber=member_id)
                    if found:
                        msgs = found
                        break
                except Exception:
                    continue
        else:
            for var in ['msl', 'prmsl', 'Mean sea level pressure']:
                try:
                    found = grbs.select(shortName=var)
                    if found:
                        msgs = found
                        break
                except Exception:
                    continue

        if not msgs:
            grbs.close()
            return None

        msgs = sorted(msgs, key=_step_hours)
        lats, lons = msgs[0].latlons()

        # forecast steps
        t_lats, t_lons, t_mslp, t_times = [], [], [], []
        search_box = CONFIG['tracking']['init_box'].copy()
        prev_step_h = None
        prev_time   = None
        misses      = 0

        for msg in msgs:
            step_h = _step_hours(msg)
            vt = init_time + timedelta(hours=step_h)

            
            if vt < CONFIG['time']['anchor']:
                prev_step_h = step_h
                continue
            if vt > CONFIG['time']['end']:
                break

            # expanding search radius
            dt_h = (step_h - prev_step_h) if prev_step_h is not None else 6.0
            sr = sr_12h if dt_h > 7.0 else sr_6h
            prev_step_h = step_h

            
            mslp = msg.values.copy()
            
            if np.nanmin(mslp) > 5000:   
                mslp = mslp / 100.0

            
            res = find_pressure_minimum(
                mslp, lats, lons, search_box, sigma, refine)

            
            if res is None and len(t_lats) > 0:
                res = find_pressure_minimum(
                    mslp, lats, lons,
                    _expand_box(search_box, retry_x),
                    sigma, refine)

            
            if res is None:
                if len(t_lats) > 0:
                    misses += 1
                    if misses > max_miss:
                        break
                continue

            
            if res['mslp'] > p_thr:
                if len(t_lats) > 0:
                    misses += 1
                    if misses > max_miss:
                        break
                continue

            # reject jumps that are too fast to be physical
            if len(t_lats) > 0 and prev_time is not None:
                d_km   = haversine_km(t_lats[-1], t_lons[-1],
                                      res['lat'], res['lon'])
                dt_sec = (vt - prev_time).total_seconds()
                if dt_sec > 0:
                    speed_ms = (d_km * 1000.0) / dt_sec
                    if speed_ms > vmax_ms:
                        misses += 1
                        if misses > max_miss:
                            break
                        continue

        
            t_lats.append(res['lat'])
            t_lons.append(res['lon'])
            t_mslp.append(res['mslp'])
            t_times.append(vt)
            search_box = _update_search_box(res['lat'], res['lon'], sr)
            prev_time = vt
            misses = 0

        grbs.close()

        if len(t_lats) < min_len:
            return None

        return {'lats':  np.array(t_lats),
                'lons':  np.array(t_lons),
                'mslp':  np.array(t_mslp),
                'times': t_times}

    except Exception:
        return None


def load_gencast_ensemble():
    print("   Loading GenCast")
    path = Path(CONFIG['dirs']['gencast'])
    files = sorted(f for f in path.glob("gencast_member_*.grib")
                   if not f.name.endswith('.idx'))[:50]
    tracks = []
    for f in files:
        t = track_single_member(f, CONFIG['time']['anchor'])
        if t is not None:
            tracks.append(t)
    print(f"     {len(tracks)}/{len(files)} members tracked")
    return tracks


def load_ecmwf_ensemble():
    print(" Loading ECMWF IFS")
    fp = Path(CONFIG['dirs']['ecmwf'])
    if not fp.exists():
        print(f"     ERROR: file not found: {fp}")
        return []
    tracks = []
    for mid in range(1, 51):
        t = track_single_member(fp, CONFIG['time']['anchor'],
                                is_ecmwf=True, member_id=mid)
        if t is not None:
            tracks.append(t)
    print(f"     {len(tracks)}/50 members tracked")
    return tracks


def load_lagged_ensemble(model_name, dir_key):
    print(f"Loading {model_name} "
          f"{len(CONFIG['time']['lagged_inits'])} inits × 10 members)")
    path = Path(CONFIG['dirs'][dir_key])
    tracks = []
    for init_time in CONFIG['time']['lagged_inits']:
        istr = init_time.strftime('%Y%m%d%H')
        for mid in range(10):
            fp = path / f"{istr}_{mid}.grib"
            if fp.exists():
                t = track_single_member(fp, init_time)
                if t is not None:
                    tracks.append(t)
    print(f"     {len(tracks)} members tracked")
    return tracks


def load_jtwc_best_track():
    print(" Loading JTWC best track")
    try:
        with open(CONFIG['dirs']['jtwc'], 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    rows = {}
    for line in lines:
        parts = line.split(',')
        if len(parts) < 10:
            continue
        try:
            dt = datetime.strptime(parts[2].strip()[:10], '%Y%m%d%H')
            if dt < CONFIG['time']['start'] or dt > CONFIG['time']['end']:
                continue
            lat_s = parts[6].strip()
            lon_s = parts[7].strip()
            lat = float(lat_s[:-1]) / 10.0
            if lat_s.endswith('S'):
                lat = -lat
            lon = float(lon_s[:-1]) / 10.0
            if lon_s.endswith('W'):
                lon = -lon
            rows[dt] = (lat, lon)
        except Exception:
            continue

    if not rows:
        return None
    times = sorted(rows.keys())
    return {'lats':  np.array([rows[t][0] for t in times]),
            'lons':  np.array([rows[t][1] for t in times]),
            'times': times}


def calculate_ensemble_mean_track(tracks, min_members=None):
    
    if not tracks:
        return None
    if min_members is None:
        min_members = CONFIG['tracking']['min_members_for_mean']

    
    all_times = set()
    for trk in tracks:
        for t in trk['times']:
            all_times.add(t)

    common_times = sorted(all_times)

    mean_lats, mean_lons, mean_times, n_mem = [], [], [], []
    for vt in common_times:
        la, lo = [], []
        for trk in tracks:
            if vt in trk['times']:
                idx = trk['times'].index(vt)
                la.append(trk['lats'][idx])
                lo.append(trk['lons'][idx])
        if len(la) >= min_members:
            mean_lats.append(float(np.mean(la)))
            mean_lons.append(float(np.mean(lo)))
            mean_times.append(vt)
            n_mem.append(len(la))

    if len(mean_lats) < CONFIG['tracking']['min_track_length']:
        return None

    return {'lats':      np.array(mean_lats),
            'lons':      np.array(mean_lons),
            'times':     mean_times,
            'n_members': np.array(n_mem)}


def truncate_for_display(lats, lons):
    
    cut_lon = CONFIG['display']['landfall_lon']
    cut_lat = CONFIG['display']['landfall_lat']

    lats = np.asarray(lats)
    lons = np.asarray(lons)
    over = np.where((lons > cut_lon) | (lats > cut_lat))[0]
    if len(over) == 0:
        return lats, lons
    cut = over[0] + 1
    if cut < 2:
        return None, None
    return lats[:cut], lons[:cut]


def print_diagnostics(model_data, jtwc):
    print(f"\n{'='*72}")
    print("TRACKING DIAGNOSTICS")
    print(f"{'='*72}")
    print("\n  Ensemble-mean temporal extent + members at lifecycle phases:")
    print(f"    {'Model':14s} {'first':>12} {'last':>12} {'n_pts':>6} "
          f"{'pre-RI':>7} {'RIon':>6} {'landfall':>9}")
    phase_t = [datetime(2008,4,28,12), datetime(2008,5,1,0), datetime(2008,5,2,0)]
    for mn in MODEL_ORDER:
        m = model_data[mn]['mean']
        if not m:
            print(f"    {mn:14s}  (no mean)"); continue
        def nmem_at(vt, tol_h=3.0):
            best, bd = 0, None
            for i, t in enumerate(m['times']):
                dh = abs((t - vt).total_seconds())/3600.0
                if dh <= tol_h and (bd is None or dh < bd):
                    bd, best = dh, int(m['n_members'][i])
            return best
        f = m['times'][0].strftime('%m-%d %HZ')
        l = m['times'][-1].strftime('%m-%d %HZ')
        p = [nmem_at(t) for t in phase_t]
        print(f"    {mn:14s} {f:>12} {l:>12} {len(m['times']):>6} "
              f"{p[0]:>7} {p[1]:>6} {p[2]:>9}")
    print(f"  {'Model':<18} {'N tracks':>9} {'Mean len':>10} {'Min len':>9} "
          f"{'Max len':>9}")
    print("  " + "-" * 60)
    for mn in MODEL_ORDER:
        tracks = model_data[mn]['tracks']
        if not tracks:
            print(f"  {mn:<18} {'—':>9} {'—':>10} {'—':>9} {'—':>9}")
            continue
        lengths = [len(t['lats']) for t in tracks]
        print(f"  {mn:<18} {len(tracks):>9d} "
              f"{np.mean(lengths):>10.1f} "
              f"{min(lengths):>9d} "
              f"{max(lengths):>9d}")

    if jtwc is not None:
        print(f"\n  JTWC best track: {len(jtwc['times'])} positions")
        print(f" from {jtwc['times'][0]} to {jtwc['times'][-1]}")



# 
def create_comparison_figure(model_data, jtwc_track, output_path):
    print(f"\n{'='*72}")
    print("FIGURE 1: ENSEMBLE TRACK COMPARISON")
    print(f"{'='*72}")

    fig = plt.figure(figsize=CONFIG['display']['figsize'])

    for i, model_name in enumerate(MODEL_ORDER):
        print(f"  Panel {i+1}: {model_name}")
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

        data = model_data.get(model_name, {'tracks': [], 'mean': None})
        conf = CONFIG['models'][model_name]

        # basemap
        ax.set_extent(CONFIG['display']['domain'], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,
                       facecolor='#f5f5dc', edgecolor='k',
                       lw=0.5, zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', zorder=0)
        ax.add_feature(cfeature.COASTLINE, lw=1.2, zorder=2)
        ax.add_feature(cfeature.BORDERS, ls=':', alpha=0.5, zorder=2)
        gl = ax.gridlines(draw_labels=True, lw=0.8,
                          color='gray', alpha=0.4, ls='--')
        gl.top_labels = False
        gl.right_labels = False

        # Individual members  
        for trk in data['tracks']:
            lat, lon = truncate_for_display(trk['lats'], trk['lons'])
            if lon is None or len(lon) < 2:
                continue
            ax.plot(lon, lat, '-',
                    color=conf['color'],
                    alpha=CONFIG['display']['member_alpha'],
                    lw=CONFIG['display']['member_linewidth'],
                    transform=ccrs.PlateCarree(),
                    zorder=3)

        # Ensemble mean
        if data['mean'] is not None:
            mlat, mlon = truncate_for_display(data['mean']['lats'],
                                              data['mean']['lons'])
            if mlon is not None and len(mlon) >= 2:
                
                ax.plot(mlon, mlat, '-',
                        color='white', lw=5,
                        transform=ccrs.PlateCarree(), zorder=4)
                ax.plot(mlon, mlat,
                        marker=conf['marker'], ls='-',
                        color=conf['color'], lw=3.5,
                        ms=7, mec='k', mew=1.5,
                        label='Ensemble Mean',
                        transform=ccrs.PlateCarree(), zorder=5)

        # JTWC best track 
        if jtwc_track is not None:
            jlat, jlon = truncate_for_display(jtwc_track['lats'],
                                              jtwc_track['lons'])
            if jlon is not None and len(jlon) >= 2:
                ax.plot(jlon, jlat,
                        marker='s', ls='-', color='k',
                        lw=3.5, ms=8, mfc='gold',
                        mec='k', mew=1.5,
                        label='JTWC Best Track',
                        transform=ccrs.PlateCarree(), zorder=6)
                # phase markers for each time
        cut_lon = CONFIG['display']['landfall_lon']
        cut_lat = CONFIG['display']['landfall_lat']
        for vt, lab in PHASE_MARKERS:
            pj = (_pos_at_time(jtwc_track['times'], jtwc_track['lats'],
                               jtwc_track['lons'], vt)
                  if jtwc_track is not None else None)
            pm = (_pos_at_time(data['mean']['times'], data['mean']['lats'],
                               data['mean']['lons'], vt)
                  if data['mean'] is not None else None)
            
            if pj and (pj[1] > cut_lon or pj[0] > cut_lat):
                pj = None
            if pm and (pm[1] > cut_lon or pm[0] > cut_lat):
                pm = None
            
            if pj and pm:
                ax.plot([pj[1], pm[1]], [pj[0], pm[0]],
                        color='0.35', lw=1.0, ls=':',
                        transform=ccrs.PlateCarree(), zorder=6)
            # JTWC phase ring and label
            if pj:
                ax.plot(pj[1], pj[0], marker='o', ms=14,
                        mfc='none', mec='k', mew=2.5,
                        transform=ccrs.PlateCarree(), zorder=9)
                txt = ax.text(pj[1] + 0.18, pj[0] - 0.40, lab,
                              fontsize=9, fontweight='bold',
                              color='k', ha='left', va='top',
                              transform=ccrs.PlateCarree(), zorder=10)
                txt.set_path_effects([
                    pe.withStroke(linewidth=2.5, foreground='white')])
            # each model's phase ring
            if pm:
                ax.plot(pm[1], pm[0], marker='o', ms=14,
                        mfc='none', mec=conf['color'], mew=2.5,
                        transform=ccrs.PlateCarree(), zorder=9)

        ax.set_title(conf['label'], fontsize=15, fontweight='bold', pad=12)

        h = [
            mlines.Line2D([], [], color='k', marker='s',
                          lw=3, ms=9, mfc='gold',
                          label='JTWC Best Track'),
            mlines.Line2D([], [], color=conf['color'],
                          marker=conf['marker'], lw=3, ms=8,
                          label='Ensemble Mean'),
            mlines.Line2D([], [], color=conf['color'], lw=1.5,
                          alpha=0.4, label='Individual Members'),
        ]
        ax.legend(handles=h, loc='lower left', fontsize=10,
                  framealpha=0.95, edgecolor='k', shadow=True)

    init_str = CONFIG['time']['anchor'].strftime('%Y-%m-%d %HZ')
    fig.suptitle(
        f"Multi-Model Ensemble Track Comparison",
        fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=CONFIG['display']['dpi'],
                bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 72)
    print("FIGURE 1-TC NARGIS (2008) MULTI-MODEL ENSEMBLE TRACK ANALYSIS")
    print("Two-stage Marchok-style tracker physical-speed validation")
    print("Mean track averaged by VALID TIME ")
    print("=" * 72)

    model_data = {}

    # Loading each ensemble order
    tracks = load_ecmwf_ensemble()
    model_data['ECMWF'] = {
        'tracks': tracks,
        'mean':   calculate_ensemble_mean_track(tracks),
    }

    tracks = load_gencast_ensemble()
    model_data['GenCast'] = {
        'tracks': tracks,
        'mean':   calculate_ensemble_mean_track(tracks),
    }

    tracks = load_lagged_ensemble('PanguWeather', 'pangu')
    model_data['PanguWeather'] = {
        'tracks': tracks,
        'mean':   calculate_ensemble_mean_track(tracks),
    }

    tracks = load_lagged_ensemble('FourCastNet', 'fourcast')
    model_data['FourCastNet'] = {
        'tracks': tracks,
        'mean':   calculate_ensemble_mean_track(tracks),
    }

    jtwc = load_jtwc_best_track()
    print_diagnostics(model_data, jtwc)
    create_comparison_figure(model_data, jtwc, CONFIG['dirs']['output_tracks'])
    print("\nDone.\n")


if __name__ == "__main__":
    main()
