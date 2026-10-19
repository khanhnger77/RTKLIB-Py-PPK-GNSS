"""
Generate the README figures in docs/images (2D, 3D and GIF animations).

    python docs/make_figures.py [--stat data/u-blox/rover.pos.stat]

The skyplot needs a .pos.stat file (written by src/run_ppk.py); it is
skipped if the file is missing.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

# Some Linux setups ship an old system mpl_toolkits that shadows the one
# installed with matplotlib -> put matplotlib's own copy first.
import mpl_toolkits
_tk = os.path.join(os.path.dirname(matplotlib.__file__), '..', 'mpl_toolkits')
if os.path.isdir(_tk):
    mpl_toolkits.__path__.insert(0, os.path.abspath(_tk))
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'plot'))
from plot_ppk import (read_pos, read_obs_pos, ecef2pos, to_enu, fix_rate,  # noqa: E402
                      plot_track, plot_enu, plot_velocity, plot_quality,
                      plot_compare, Q_STYLE)

OUT = os.path.join(ROOT, 'docs', 'images')
QCOL = {k: v[1] for k, v in Q_STYLE.items()}


def qcolors(q):
    return [QCOL.get(int(k), 'k') for k in q]


# ------------------------------------------------------------ diagrams
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set(xlim=(0, 13), ylim=(0, 5.2))
    ax.axis('off')

    def box(x, y, w, h, title, sub, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.15',
                                    fc=c, ec='0.25', lw=1.2))
        ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center',
                fontsize=10.5, weight='bold')
        ax.text(x + w / 2, y + h * 0.3, sub, ha='center', va='center', fontsize=8.2)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                     mutation_scale=14, lw=1.3, color='0.3'))

    cin, cpy, cout = '#dbeafe', '#fef3c7', '#dcfce7'
    box(0.2, 3.7, 2.1, 1.0, 'Rover obs', 'RINEX 3 (.obs)', cin)
    box(0.2, 2.3, 2.1, 1.0, 'Base obs', 'RINEX 3 (.obs)', cin)
    box(0.2, 0.9, 2.1, 1.0, 'Navigation', 'ephemeris (.nav)', cin)
    box(3.0, 2.3, 2.1, 1.0, 'rinex.py', 'decode obs / nav', cpy)
    box(5.8, 3.7, 2.3, 1.0, 'ephemeris.py', 'sat pos + clock', cpy)
    box(5.8, 2.3, 2.3, 1.0, 'pntpos.py', 'single point (init)', cpy)
    box(5.8, 0.9, 2.3, 1.0, 'rtkpos.py', 'EKF + double diff', cpy)
    box(8.8, 0.9, 2.0, 1.0, 'mlambda.py', 'integer AR + ratio', cpy)
    box(8.8, 2.3, 2.0, 1.0, 'postpos.py', 'fwd / bwd / combine', cpy)
    box(11.2, 3.0, 1.6, 1.0, '.pos', 'solution', cout)
    box(11.2, 1.6, 1.6, 1.0, '.pos.stat', 'residuals', cout)
    box(8.8, 3.7, 2.0, 1.0, 'plot_ppk.py', '2D / 3D plots', cout)

    for y in (4.2, 2.8, 1.4):
        arrow(2.3, y, 3.0, 2.8)
    arrow(5.1, 2.9, 5.8, 4.1)
    arrow(5.1, 2.8, 5.8, 2.8)
    arrow(6.95, 2.3, 6.95, 1.9)
    arrow(8.1, 1.4, 8.8, 1.4)
    arrow(9.8, 1.9, 9.8, 2.3)
    arrow(10.8, 2.9, 11.2, 3.4)
    arrow(10.8, 2.7, 11.2, 2.1)
    arrow(12.0, 4.0, 10.8, 4.2)
    ax.annotate('', xy=(8.1, 1.15), xytext=(8.8, 1.15),
                arrowprops=dict(arrowstyle='-|>', color='#b45309', lw=1.2,
                                connectionstyle='arc3,rad=0.5'))
    ax.text(8.45, 0.55, 'fix & hold', ha='center', fontsize=8, color='#b45309')
    ax.text(6.5, 5.0, 'RTKLIB-Py PPK processing pipeline', ha='center',
            fontsize=13, weight='bold')
    fig.tight_layout()
    return fig


def fig_double_diff():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set(xlim=(0, 10), ylim=(0, 6.5), aspect='equal')
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 10, 0.8, color='#d6c7a1'))
    sats = {'i': (2.3, 5.6), 'j': (7.7, 5.9)}
    rx = {'Base (b)': (2.0, 0.8), 'Rover (r)': (7.0, 0.8)}
    col = {'i': '#2563eb', 'j': '#dc2626'}
    for s, (x, y) in sats.items():
        ax.add_patch(plt.Rectangle((x - 0.35, y - 0.12), 0.7, 0.24, color='0.35'))
        ax.add_patch(plt.Rectangle((x - 0.95, y - 0.06), 0.5, 0.12, color='#60a5fa'))
        ax.add_patch(plt.Rectangle((x + 0.45, y - 0.06), 0.5, 0.12, color='#60a5fa'))
        ax.text(x, y + 0.3, 'Satellite %s' % s, ha='center', fontsize=10, weight='bold')
        for (xr, yr) in rx.values():
            ax.plot([x, xr], [y - 0.12, yr + 0.35], color=col[s], lw=1.4, ls='--')
    for name, (x, y) in rx.items():
        ax.add_patch(plt.Polygon([[x - 0.3, y], [x + 0.3, y], [x, y + 0.35]], color='k'))
        ax.text(x, y - 0.45, name, ha='center', fontsize=10, weight='bold')
    ax.annotate('', xy=(6.7, 1.4), xytext=(2.3, 1.4),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(4.5, 1.55, 'baseline  (≈ 0–10 km)', ha='center', fontsize=9)
    ax.text(5.0, 3.5,
            r'$\nabla\Delta\phi_{rb}^{ij} = (\phi_r^i-\phi_b^i)-(\phi_r^j-\phi_b^j)$'
            '\n' r'$=\nabla\Delta\rho_{rb}^{ij} + \lambda\,\nabla\Delta N_{rb}^{ij} + \varepsilon$',
            ha='center', va='center', fontsize=12,
            bbox=dict(fc='white', ec='0.6', boxstyle='round'), zorder=5)
    ax.text(5.0, 2.45, 'receiver + satellite clocks cancel, atmosphere ≈ cancels\n'
            r'integer ambiguities $\nabla\Delta N$ solved by LAMBDA',
            ha='center', fontsize=9, color='0.25', zorder=5,
            bbox=dict(fc='white', ec='none', alpha=0.9))
    ax.set_title('Double-differenced carrier phase', fontsize=13, weight='bold')
    fig.tight_layout()
    return fig


def fig_skyplot(stat_file):
    sats = {}
    with open(stat_file) as f:
        for line in f:
            if not line.startswith('$SAT'):
                continue
            v = line.split(',')
            if v[4] != '1':
                continue
            sats.setdefault(v[3], []).append((float(v[5]), float(v[6])))
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    sys_col = {'G': '#2563eb', 'R': '#dc2626', 'E': '#16a34a', 'J': '#9333ea', 'C': '#ea580c'}
    for sat, ae in sorted(sats.items()):
        ae = np.array(ae)
        c = sys_col.get(sat[0], 'k')
        ax.plot(np.radians(ae[:, 0]), 90 - ae[:, 1], '.', ms=1.5, color=c)
        ax.text(np.radians(ae[-1, 0]), 90 - ae[-1, 1], sat, fontsize=7, color=c)
    ax.set_rlim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=7)
    for s, name in (('G', 'GPS'), ('R', 'GLONASS'), ('E', 'Galileo')):
        ax.plot([], [], 'o', color=sys_col[s], label=name)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.08, -0.08))
    ax.set_title('Skyplot - rover satellites used', pad=18, weight='bold')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ 3D
def _setup_3d(ax, e, n, u, zoom=1.0):
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)', labelpad=6)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.set_box_aspect((np.ptp(e), np.ptp(n), 0.35 * max(np.ptp(e), np.ptp(n))), zoom=zoom)


def fig_track3d(sol, ref):
    e, n, u = to_enu(sol, ref)
    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_axes([0, 0, 0.93, 0.94], projection='3d')
    ax.plot(e, n, np.full_like(u, u.min()), color='0.8', lw=0.8)  # ground shadow
    ax.scatter(e, n, u, c=qcolors(sol['Q']), s=3, depthshade=False)
    for k in sorted(set(sol['Q'])):
        ax.scatter([], [], [], c=QCOL[k], label=Q_STYLE[k][0])
    _setup_3d(ax, e, n, u, zoom=1.0)
    ax.view_init(28, -60)
    ax.legend(loc='upper left')
    ax.set_title('3D trajectory (height exaggerated) - fix %.1f%%' % fix_rate(sol))
    return fig


def gif_rotate3d(sol, ref, path, frames=60):
    e, n, u = to_enu(sol, ref)
    fig = plt.figure(figsize=(8, 5.5), dpi=72)
    ax = fig.add_axes([0, 0, 1, 0.93], projection='3d')
    ax.plot(e, n, np.full_like(u, u.min()), color='0.85', lw=0.6)
    ax.scatter(e, n, u, c=qcolors(sol['Q']), s=2, depthshade=False)
    _setup_3d(ax, e, n, u, zoom=1.15)
    ax.set_title('PPK trajectory 3D (green=fix, orange=float)', fontsize=10)

    def step(i):
        ax.view_init(25 + 10 * np.sin(2 * np.pi * i / frames), -60 + 360 * i / frames)
        return []

    FuncAnimation(fig, step, frames=frames).save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def gif_drive(sol, ref, path, frames=120):
    e, n, u = to_enu(sol, ref)
    t = (sol['t'] - sol['t'][0]) / 60
    idx = np.linspace(1, len(e) - 1, frames).astype(int)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=80,
                                 gridspec_kw=dict(width_ratios=[1.15, 1]))
    a1.plot(e, n, color='0.88', lw=1)
    a1.set(aspect='equal', xlabel='East (m)', ylabel='North (m)')
    a1.grid(alpha=0.3)
    trail = a1.scatter([], [], s=3, linewidths=0)
    head, = a1.plot([], [], 'o', ms=8, mfc='#facc15', mec='k')
    a2.set(xlim=(0, t[-1]), ylim=(-0.02, 1.02), xlabel='Time (min)',
           ylabel='Fix ratio (cumulative)')
    a2.grid(alpha=0.3)
    line, = a2.plot([], [], color='#2ca02c', lw=2)
    txt = a2.text(0.03, 0.08, '', transform=a2.transAxes, fontsize=10,
                  bbox=dict(fc='white', ec='0.7'))
    cum = np.cumsum(sol['Q'] == 1) / np.arange(1, len(e) + 1)
    fig.suptitle('Rover drive replay - %s' % sol['name'], weight='bold')
    fig.tight_layout()

    def step(k):
        i = idx[k]
        trail.set_offsets(np.c_[e[:i], n[:i]])
        trail.set_color(qcolors(sol['Q'][:i]))
        head.set_data([e[i]], [n[i]])
        line.set_data(t[:i], cum[:i])
        name = Q_STYLE.get(int(sol['Q'][i]), ('?',))[0]
        txt.set_text('t = %5.1f min   %s   ns = %d\nσ_N/E/U = %.3f / %.3f / %.3f m'
                     % (t[i], name, sol['ns'][i], sol['sdn'][i], sol['sde'][i], sol['sdu'][i]))
        return trail, head, line, txt

    FuncAnimation(fig, step, frames=frames).save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pos', default=os.path.join(ROOT, 'data/u-blox/rover.pos'))
    p.add_argument('--ref', default=os.path.join(ROOT, 'data/u-blox/rtklib_demo5.pos'))
    p.add_argument('--base', default=os.path.join(ROOT, 'data/u-blox/tmg23590.obs'))
    p.add_argument('--stat', default=os.path.join(ROOT, 'data/u-blox/rover.pos.stat'))
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    sol, refsol = read_pos(a.pos), read_pos(a.ref)
    ref = ecef2pos(read_obs_pos(a.base))

    figs = {
        'pipeline': fig_pipeline(),
        'double_difference': fig_double_diff(),
        'track_2d': plot_track(sol, ref),
        'track_2d_rtklib': plot_track(refsol, ref),
        'enu': plot_enu(sol, ref),
        'velocity': plot_velocity(sol, ref, 1),
        'acceleration': plot_velocity(sol, ref, 2),
        'quality': plot_quality(sol),
        'compare_rtklib': plot_compare(sol, refsol, ref),
        'track_3d': fig_track3d(sol, ref),
    }
    if os.path.exists(a.stat):
        figs['skyplot'] = fig_skyplot(a.stat)
    else:
        print('skip skyplot: no', a.stat)
    for k, f in figs.items():
        f.savefig(os.path.join(OUT, k + '.png'), dpi=110, bbox_inches='tight', pad_inches=0.3)
        plt.close(f)
        print('saved', k + '.png')

    gif_rotate3d(sol, ref, os.path.join(OUT, 'track_3d_rotate.gif'))
    print('saved track_3d_rotate.gif')
    gif_drive(sol, ref, os.path.join(OUT, 'drive_replay.gif'))
    print('saved drive_replay.gif')


if __name__ == '__main__':
    main()
