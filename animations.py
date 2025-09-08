#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
animations.py
-------------
Dual-panel animation driven by kv_flow_autoscale.simulate_series():

Left:
  - Arrivals / Throughput / Max-Flow (yellow) / p95 latency
  - SLA line (gray), vertical cursor, ▲/▼ autoscale markers

Right:
  - Per-edge bars: Flow / Capacity for
      S→P1, S→P2, P1→D1, P1→D2, P2→D1, P2→D2, D1→T, D2→T
    * Flow bar turns RED when saturated (flow == capacity)
    * Label INSIDE each bar: "flow/capacity" (yellow text if saturated)
  - Status box INSIDE the right chart (top-right corner)

CLI:
  python animations.py --ticks 300 --sla 50 --fps 6 --outfile lifecycle.gif
  python animations.py --ticks 300 --sla 50 --fps 6 --mp4 lifecycle.mp4
  python animations.py --ticks 300 --sla 50 --fps 6 --arr-boost 1.6
"""

import argparse
from pathlib import Path
import importlib.util
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Fixed edge ordering (bars + summaries)
EDGES = [
    ("S","P1"),("S","P2"),
    ("P1","D1"),("P1","D2"),
    ("P2","D1"),("P2","D2"),
    ("D1","T"),("D2","T"),
]
EDGE_LABELS = [f"{u}→{v}" for (u,v) in EDGES]

# ---------- import kv_flow_autoscale.py dynamically ----------
def _import_kv_flow_autoscale():
    here = Path(__file__).resolve().parent
    mod_path = here / "kv_flow_autoscale.py"
    if not mod_path.exists():
        raise FileNotFoundError("kv_flow_autoscale.py not found next to animations.py")
    spec = importlib.util.spec_from_file_location("kv_flow_autoscale", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kv_flow_autoscale"] = mod
    spec.loader.exec_module(mod)
    return mod

# ---------- helpers ----------
def _edge_key(u, v): return f"{u}→{v}"

def _bottlenecks(flow, cap):
    """Edges where flow == capacity > 0."""
    out = []
    for (u, v) in EDGES:
        f = flow.get((u, v), 0)
        c = cap.get((u, v), 0)
        if c > 0 and f >= c:
            out.append((u, v))
    return out

def _format_event(evt):
    if evt is None: return ""
    typ, updates = evt
    if not updates: return ""
    head = "UP" if typ == "up" else "DOWN"
    kvs = ", ".join(f"{k}={v}" for k, v in updates.items())
    return f"{head}: {kvs}"

def _format_capacities(cap_dict):
    """Multiline capacity list."""
    lines = ["Capacities:"]
    for (u, v) in EDGES:
        lines.append(f"{u}→{v}: {cap_dict.get((u,v),0)}")
    return "\n".join(lines)

def _format_flows_caps_inline(flow, cap):
    lines = ["Edges (flow/cap):"]
    for (u, v) in EDGES:
        f = flow.get((u, v), 0)
        c = cap.get((u, v), 0)
        lines.append(f"{u}→{v}: {f}/{c}")
    return "\n".join(lines)

# ---------- animation ----------
def draw_lifecycle(S, fps=6, outfile="lifecycle.gif", mp4=None):
    """
    S: dict from kv_flow_autoscale.simulate_series(...):
       keys: tick, arr, tp, p95, mf, flows[], caps[], events[], sla
       - events[t] is None or ("up"|"down", {"D1": new_cap, "D2": new_cap, ...})
    """
    T = len(S["tick"])
    plt.close("all")

    # Big canvas, crisp text
    fig = plt.figure(figsize=(18, 10))
    fig.set_dpi(140)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 3.2])

    # ----- LEFT: time series -----
    axL = fig.add_subplot(gs[0, 0])
    axR = axL.twinx()

    axL.set_xlim(0, S["tick"][-1])
    axL.set_ylim(0, max(S["mf"].max(), S["arr"].max(), S["tp"].max()) * 1.25 + 5)
    axR.set_ylim(0, max(S["p95"].max(), S["sla"]) * 1.35 + 10)

    # Max-flow yellow; others default; SLA gray to avoid clash
    (l_mf,)  = axL.plot([], [], linewidth=2.4, color="#f0c808", label="max-flow")
    (l_tp,)  = axL.plot([], [], linewidth=1.6, label="throughput")
    (l_arr,) = axL.plot([], [], linewidth=1.6, label="arrivals")
    (l_p95,) = axR.plot([], [], linewidth=2.0, label="p95 latency (ms)")
    sla_line = axR.axhline(S["sla"], linestyle="--", linewidth=1.8, color="#888888", label="SLA")

    # Cursor + autoscale markers
    cursor = axL.axvline(0, linestyle="-.", linewidth=1.0)
    (sc_up,)   = axR.plot([], [], marker="^", linestyle="None", markersize=8, label="▲ up")
    (sc_down,) = axR.plot([], [], marker="v", linestyle="None", markersize=8, label="▼ down")

    axL.set_title("Lifecycle: Requests ↑ → Bottleneck → ▲ Scale-Up → Stable (SLA) → Requests ↓ → ▼ Scale-Down")
    axL.set_xlabel("Tick")
    axL.set_ylabel("Req / cycle")
    axR.set_ylabel("Latency (ms)")

    # Legend
    lines_left  = [l_mf, l_tp, l_arr]
    lines_right = [l_p95, sla_line, sc_up, sc_down]
    labels = [h.get_label() for h in lines_left + lines_right]
    axL.legend(lines_left + lines_right, labels, loc="upper left")

    # ----- RIGHT: bars + inside labels + status box -----
    axB = fig.add_subplot(gs[0, 1])
    x = np.arange(len(EDGES))

    # Explicit colors to avoid backend/theme issues
    CAP_COLOR  = "#cbd5e1"  # light steel blue
    FLOW_COLOR = "#1f77b4"  # matplotlib default blue
    RED_COLOR  = "#d62728"  # red

    bars_cap  = axB.bar(x, [0] * len(EDGES), color=CAP_COLOR, alpha=0.95, label="capacity")
    bars_flow = axB.bar(x, [0] * len(EDGES), color=FLOW_COLOR, label="flow")

    # Text inside bars
    bar_texts = [
        axB.text(i, 0, "", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")
        for i in x
    ]

    axB.set_xticks(x)
    axB.set_xticklabels(EDGE_LABELS, rotation=45, ha="right")
    axB.set_ylim(0, 150)  # auto-grow later
    axB.set_ylabel("req / cycle")
    axB.set_title("Per-edge Flow / Capacity (red = saturated)")
    axB.legend()

    # Status box INSIDE the right bar chart (top-left corner)
    status = axB.text(
        0.015, 0.98, "", transform=axB.transAxes,
        ha="left", va="top", family="monospace", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.85, boxstyle="round,pad=0.35"),
        zorder=10
    )

    fig.tight_layout(pad=1.2)

    # Pre-compute autoscale markers
    up_X, up_Y, dn_X, dn_Y = [], [], [], []
    for t, evt in enumerate(S["events"]):
        if evt is None:
            continue
        typ, _ = evt
        y = max(S["p95"][t], S["sla"] * 0.9)
        if typ == "up":
            up_X.append(t); up_Y.append(y)
        else:
            dn_X.append(t); dn_Y.append(y)

    def update(frame: int):
        # Left curves
        xd = S["tick"][: frame + 1]
        l_mf.set_data(xd,  S["mf"][: frame + 1])
        l_arr.set_data(xd, S["arr"][: frame + 1])
        l_tp.set_data(xd,  S["tp"][: frame + 1])
        l_p95.set_data(xd, S["p95"][: frame + 1])
        cursor.set_xdata([frame, frame])

        ux = [t for t in up_X if t <= frame];  uy = [S["p95"][t] for t in ux]
        dx = [t for t in dn_X if t <= frame];  dy = [S["p95"][t] for t in dx]
        sc_up.set_data(ux, uy); sc_down.set_data(dx, dy)

        # Right bars
        cap  = S["caps"][frame]
        flow = S["flows"][frame]

        top = 0
        for i, (u, v) in enumerate(EDGES):
            c = cap.get((u, v), 0)
            f = flow.get((u, v), 0)

            bars_cap[i].set_height(c)

            # Explicit color each frame
            saturated = (c > 0 and f >= c)
            bars_flow[i].set_height(f)
            bars_flow[i].set_color(RED_COLOR if saturated else FLOW_COLOR)

            # Label inside bar
            bar_texts[i].set_text(f"{f}/{c}")
            if f > 0:
                bar_texts[i].set_y(f / 2)
            else:
                bar_texts[i].set_y(c / 2 if c > 0 else 1)
            bar_texts[i].set_color("yellow" if saturated else "white")

            top = max(top, max(f, c))

        if top * 1.10 + 5 > axB.get_ylim()[1] * 0.98:
            axB.set_ylim(0, top * 1.10 + 5)

        # Status inside RIGHT chart (axB)
        evt = S["events"][frame]
        bn  = _bottlenecks(flow, cap)
        bn_str = ", ".join(_edge_key(u, v) for (u, v) in bn) if bn else "None"
        evt_str = _format_event(evt)
        # caps_text = _format_capacities(cap)
        caps_text = _format_flows_caps_inline(flow, cap)

        status.set_text(
            f"STATUS @ tick {int(frame)}\n"
            "---------------------------\n"
            f"Arrivals: {int(S['arr'][frame]):>4} | Throughput: {int(S['tp'][frame]):>4}\n"
            f"p95 (ms): {float(S['p95'][frame]):>5.1f} | SLA: {int(S['sla']):>4}\n"
            f"Bottleneck edges: {bn_str}\n"
            f"{evt_str}\n"
            f"{caps_text}"
        )

        return [
            l_mf, l_arr, l_tp, l_p95, cursor, sc_up, sc_down,
            *bars_cap, *bars_flow, *bar_texts, sla_line, status
        ]

    anim = FuncAnimation(fig, update, frames=T, blit=False)

    # Save (avoid bbox_inches during animations to prevent frame-size warnings)
    if mp4:
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=140)
        except Exception:
            anim.save(outfile, writer=PillowWriter(fps=fps), dpi=140)
    else:
        anim.save(outfile, writer=PillowWriter(fps=fps), dpi=140)

    print("Saved:", mp4 or outfile)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks",     type=int, default=300, help="Timeline length in ticks")
    parser.add_argument("--sla",       type=int, default=50,  help="p95 SLA (ms)")
    parser.add_argument("--fps",       type=int, default=6,   help="Lower FPS => longer per-tick display")
    parser.add_argument("--outfile",   type=str, default="lifecycle.gif", help="Output GIF path")
    parser.add_argument("--mp4",       type=str, default=None, help="Output MP4 path (requires ffmpeg)")
    parser.add_argument("--seed",      type=int, default=2025, help="RNG seed")
    parser.add_argument("--arr-boost", type=float, default=1.6, help="Multiply arrivals (requests/sec)")
    args = parser.parse_args()

    kv = _import_kv_flow_autoscale()
    # Pass arrival boost through to the simulator
    S = kv.simulate_series(T=args.ticks, sla=args.sla, seed=args.seed, arrival_boost=args.arr_boost)
    draw_lifecycle(S, fps=args.fps, outfile=args.outfile, mp4=args.mp4)

if __name__ == "__main__":
    main()
