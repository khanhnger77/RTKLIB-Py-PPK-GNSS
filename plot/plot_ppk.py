"""
Plot tools for RTKLIB-style .pos solution files.

Replaces the old accel.py / gndTrk.py / numberSat.py / position.py /
velocity.py scripts with one CLI that only needs numpy + matplotlib.

Examples (run from repo root):
    python plot/plot_ppk.py                          # all 2D plots, show
    python plot/plot_ppk.py track enu --save out/    # selected plots, save PNG
    python plot/plot_ppk.py all --ref data/u-blox/rtklib_demo5.pos
"""

import argparse
import os
import re
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# WGS84
RE_WGS84 = 6378137.0
FE_WGS84 = 1.0 / 298.257223563

# Solution quality flags (RTKLIB)
Q_STYLE = {1: ('Fix', '#2ca02c'), 2: ('Float', '#ff7f0e'),
           3: ('SBAS', '#9467bd'), 4: ('DGPS', '#1f77b4'),
           5: ('Single', '#d62728'), 6: ('PPP', '#8c564b')}

COLS = ['lat', 'lon', 'h', 'Q', 'ns', 'sdn', 'sde', 'sdu',
        'sdne', 'sdeu', 'sdun', 'age', 'ratio']


# ---------------------------------------------------------------- I/O
def read_pos(path):
    """Read an RTKLIB .pos file (lat/lon/height, GPST yyyy/mm/dd hh:mm:ss)."""
    t, rows = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('%') or not line.strip():
                continue
            v = line.split()
            # RTKLIB may print seconds as 60.000 -> roll over manually
            hh, mm, ss = v[1].split(':')
            base = datetime.strptime(v[0] + ' ' + hh + ':' + mm, '%Y/%m/%d %H:%M')
            t.append(base.timestamp() + float(ss))
            rows.append([float(x) for x in v[2:2 + len(COLS)]])
    d = np.array(rows)
    sol = {k: d[:, i] for i, k in enumerate(COLS[:d.shape[1]])}
    sol['t'] = np.array(t)
    sol['Q'] = sol['Q'].astype(int)
    sol['name'] = os.path.splitext(os.path.basename(path))[0]
    return sol


def read_obs_pos(path):
    """APPROX POSITION XYZ (ECEF) from a RINEX obs header."""
    with open(path) as f:
        for line in f:
            if 'APPROX POSITION XYZ' in line:
                return np.array([float(x) for x in re.findall(r'[-+\d.Ee]+', line[:60])])
            if 'END OF HEADER' in line:
                break
    raise ValueError('APPROX POSITION XYZ not found in ' + path)


# -------------------------------------------------------- coordinates
def pos2ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    e2 = FE_WGS84 * (2 - FE_WGS84)
    n = RE_WGS84 / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + h) * np.sin(lat)])


def ecef2pos(r):
    e2 = FE_WGS84 * (2 - FE_WGS84)
    r2 = r[0] ** 2 + r[1] ** 2
    z, zk = r[2], 0.0
    while abs(z - zk) >= 1e-4:
        zk = z
        sinp = z / np.sqrt(r2 + z ** 2)
        v = RE_WGS84 / np.sqrt(1 - e2 * sinp ** 2)
        z = r[2] + v * e2 * sinp
    lat = np.arctan(z / np.sqrt(r2))
    lon = np.arctan2(r[1], r[0])
    return np.degrees(lat), np.degrees(lon), np.sqrt(r2 + z ** 2) - v


def to_enu(sol, ref):
    """ENU (m) of a solution relative to geodetic ref (lat, lon, h)."""
    lat0, lon0 = np.radians(ref[0]), np.radians(ref[1])
    dr = pos2ecef(sol['lat'], sol['lon'], sol['h']) - pos2ecef(*ref)[:, None]
    sl, cl, so, co = np.sin(lat0), np.cos(lat0), np.sin(lon0), np.cos(lon0)
    e = -so * dr[0] + co * dr[1]
    n = -sl * co * dr[0] - sl * so * dr[1] + cl * dr[2]
    u = cl * co * dr[0] + cl * so * dr[1] + sl * dr[2]
    return e, n, u


def default_ref(sol, base_obs=None):
    if base_obs:
        return ecef2pos(read_obs_pos(base_obs))
    return sol['lat'][0], sol['lon'][0], sol['h'][0]


def fix_rate(sol):
    return 100.0 * np.mean(sol['Q'] == 1)


# -------------------------------------------------------------- plots
def _scatter_q(ax, x, y, q, s=4):
    for k in sorted(set(q)):
        m = q == k
        name, c = Q_STYLE.get(k, ('Q=%d' % k, 'k'))
        ax.scatter(x[m], y[m], s=s, c=c, label='%s (%.1f%%)' % (name, 100 * m.mean()))


def plot_track(sol, ref, ax=None):
    ax = ax or plt.figure(figsize=(8, 7)).gca()
    e, n, _ = to_enu(sol, ref)
    ax.plot(e, n, '-', color='0.8', lw=0.6, zorder=0)
    _scatter_q(ax, e, n, sol['Q'])
    ax.set(xlabel='East (m)', ylabel='North (m)', aspect='equal',
           title='Ground track - %s' % sol['name'])
    ax.grid(alpha=0.3)
    ax.legend(markerscale=3)
    return ax.figure


def plot_enu(sol, ref):
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    t = (sol['t'] - sol['t'][0]) / 60
    for ax, v, lab in zip(axs, to_enu(sol, ref), ['East', 'North', 'Up']):
        ax.plot(t, v, color='0.75', lw=0.6, zorder=0)
        _scatter_q(ax, t, v, sol['Q'], s=2)
        ax.set_ylabel(lab + ' (m)')
        ax.grid(alpha=0.3)
    axs[0].legend(markerscale=4, loc='upper right', ncol=3)
    axs[0].set_title('ENU position - %s' % sol['name'])
    axs[-1].set_xlabel('Time since start (min)')
    fig.tight_layout()
    return fig


def _deriv(sol, ref, order):
    enu = np.array(to_enu(sol, ref))
    dt = np.gradient(sol['t'])
    for _ in range(order):
        enu = np.gradient(enu, axis=1) / dt
    return enu


def plot_velocity(sol, ref, order=1):
    unit, name = ('m/s', 'Velocity') if order == 1 else ('m/s²', 'Acceleration')
    d = _deriv(sol, ref, order)
    t = (sol['t'] - sol['t'][0]) / 60
    fig, ax = plt.subplots(figsize=(11, 5))
    for v, lab in zip(d, ['East', 'North', 'Up']):
        ax.plot(t, v, lw=0.8, label=lab)
    ax.plot(t, np.linalg.norm(d, axis=0), 'k--', lw=0.8, label='Total')
    ax.set(xlabel='Time since start (min)', ylabel='%s (%s)' % (name, unit),
           title='%s - %s' % (name, sol['name']))
    ax.grid(alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    return fig


def plot_quality(sol):
    t = (sol['t'] - sol['t'][0]) / 60
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axs[0].step(t, sol['ns'], where='post', color='#1f77b4')
    axs[0].set_ylabel('Satellites')
    for k in ('sdn', 'sde', 'sdu'):
        axs[1].semilogy(t, sol[k], lw=0.8, label=k)
    axs[1].set_ylabel('Std dev (m)')
    axs[1].legend(ncol=3)
    axs[2].plot(t, sol['ratio'], color='#9467bd', lw=0.8)
    axs[2].axhline(3, color='r', ls='--', lw=0.8, label='thresar = 3')
    axs[2].set_ylabel('AR ratio')
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[0].set_title('Solution quality - %s (fix %.1f%%)' % (sol['name'], fix_rate(sol)))
    axs[-1].set_xlabel('Time since start (min)')
    fig.tight_layout()
    return fig


def plot_compare(sol, refsol, ref):
    """Horizontal/vertical difference vs a reference solution (common epochs, both fixed)."""
    ts, tr = np.round(sol['t'], 1), np.round(refsol['t'], 1)
    common, i, j = np.intersect1d(ts, tr, return_indices=True)
    a = {k: sol[k][i] for k in ('lat', 'lon', 'h')}
    b = {k: refsol[k][j] for k in ('lat', 'lon', 'h')}
    ea, na, ua = to_enu(a, ref)
    eb, nb, ub = to_enu(b, ref)
    both = (sol['Q'][i] == 1) & (refsol['Q'][j] == 1)
    t = (common - sol['t'][0]) / 60
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, d, lab in zip(axs, (ea - eb, na - nb, ua - ub), ['dE', 'dN', 'dU']):
        ax.plot(t, d * 100, color='0.7', lw=0.6, label='all')
        ax.plot(t[both], d[both] * 100, '.', ms=2, color='#2ca02c', label='both fixed')
        rms = np.sqrt(np.mean(d[both] ** 2)) * 100 if both.any() else np.nan
        ax.set_ylabel('%s (cm)' % lab)
        ax.text(0.01, 0.85, 'RMS (both fixed) = %.1f cm' % rms, transform=ax.transAxes)
        ax.grid(alpha=0.3)
    axs[0].legend(loc='upper right', markerscale=4)
    axs[0].set_title('%s minus %s' % (sol['name'], refsol['name']))
    axs[-1].set_xlabel('Time since start (min)')
    fig.tight_layout()
    return fig


PLOTS = ('track', 'enu', 'vel', 'acc', 'quality')


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('plots', nargs='*', default=['all'],
                   help='any of: %s, compare, all' % ', '.join(PLOTS))
    p.add_argument('--pos', default=os.path.join(root, 'data/u-blox/rover.pos'))
    p.add_argument('--base', default=os.path.join(root, 'data/u-blox/tmg23590.obs'),
                   help='base RINEX obs; ENU origin = its APPROX POSITION ('
                        'use "none" for first epoch)')
    p.add_argument('--ref', help='reference .pos for "compare" (e.g. RTKLIB demo5)')
    p.add_argument('--save', metavar='DIR', help='save PNGs instead of showing')
    a = p.parse_args()

    sol = read_pos(a.pos)
    ref = default_ref(sol, None if a.base == 'none' else a.base)
    want = list(PLOTS) + (['compare'] if a.ref else []) if 'all' in a.plots else a.plots
    print('%s: %d epochs, fix rate %.1f%%' % (sol['name'], len(sol['t']), fix_rate(sol)))

    figs = {}
    for w in want:
        if w == 'track':
            figs[w] = plot_track(sol, ref)
        elif w == 'enu':
            figs[w] = plot_enu(sol, ref)
        elif w == 'vel':
            figs[w] = plot_velocity(sol, ref, 1)
        elif w == 'acc':
            figs[w] = plot_velocity(sol, ref, 2)
        elif w == 'quality':
            figs[w] = plot_quality(sol)
        elif w == 'compare':
            figs[w] = plot_compare(sol, read_pos(a.ref), ref)
        else:
            p.error('unknown plot: ' + w)

    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for k, f in figs.items():
            out = os.path.join(a.save, '%s_%s.png' % (sol['name'], k))
            f.savefig(out, dpi=120)
            print('saved', out)
    else:
        plt.show()


if __name__ == '__main__':
    main()
