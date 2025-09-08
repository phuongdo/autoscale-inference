#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kv_flow_autoscale.py
--------------------

- Simple in-memory "API" for capacities:
    * Prefill:   S -> P1, P2
    * KV links:  P1,P2 -> D1,D2
    * Decode:    D1,D2 -> T

- Edmonds–Karp max-flow on the fixed S→P→D→T graph to compute:
    * total max-flow
    * per-edge flow assignments

- A simple autoscaler policy that:
    * scales decode up when utilization / latency is high
    * scales decode down when the system is idle
    * (optionally) nudges prefill when decode is far from saturated

- `simulate_series(...)` produces a time series used by animations:
    arrivals, throughput, p95, max-flow, per-edge flows/caps, and autoscale events.
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import random
import numpy as np

# =========================
# Fake API for capacities
# =========================

@dataclass
class PrefillConfig:
    S_to_P: Dict[str, int]      # e.g. {"P1": 60, "P2": 70}

@dataclass
class DecodeConfig:
    D_to_T: Dict[str, int]      # e.g. {"D1": 65, "D2": 55}

@dataclass
class KVConfig:
    P_to_D: Dict[Tuple[str, str], int]  # e.g. {("P1","D1"): 40, ("P1","D2"): 25, ...}

class FakeAPI:
    """In-memory stand-in for BASE/prefill and BASE/decode endpoints."""
    def __init__(self, prefill: PrefillConfig, decode: DecodeConfig, kv: KVConfig):
        self.prefill = prefill
        self.decode = decode
        self.kv = kv

    # GET
    def get_prefill(self) -> PrefillConfig: return self.prefill
    def get_decode(self) -> DecodeConfig:   return self.decode
    def get_kv(self) -> KVConfig:           return self.kv

    # POST/PUT
    def set_prefill(self, S_to_P: Dict[str, int]): self.prefill.S_to_P.update(S_to_P)
    def set_decode(self, D_to_T: Dict[str, int]):  self.decode.D_to_T.update(D_to_T)

# =========================
# Edmonds–Karp Max-Flow
# =========================

class MaxFlow:
    """
    Fixed DAG topology:
        S -> P1,P2
        P1,P2 -> D1,D2
        D1,D2 -> T
    We only store forward capacities; reverse edges are implied in residual graph.
    """
    def __init__(self, api: FakeAPI):
        self.api = api
        self.adj = {
            "S":  ["P1", "P2"],
            "P1": ["D1", "D2"],
            "P2": ["D1", "D2"],
            "D1": ["T"],
            "D2": ["T"],
            "T":  []
        }
        self.cap  = defaultdict(int)  # capacity(u,v)
        self.flow = defaultdict(int)  # flow(u,v)

    def build_capacities(self):
        """Load capacities from the API."""
        self.cap.clear()
        self.flow.clear()

        prefill = self.api.get_prefill().S_to_P
        decode  = self.api.get_decode().D_to_T
        kv      = self.api.get_kv().P_to_D

        # S -> P*
        for P, c in prefill.items():
            self.cap[("S", P)] = c

        # P* -> D*
        for (P, D), c in kv.items():
            self.cap[(P, D)] = c

        # D* -> T
        for D, c in decode.items():
            self.cap[(D, "T")] = c

    def residual(self, u, v) -> int:
        """Residual capacity on (u,v) considering forward/backward edges."""
        if (u, v) in self.cap:
            return self.cap[(u, v)] - self.flow[(u, v)]
        if (v, u) in self.cap:
            return self.flow[(v, u)]
        return 0

    def bfs(self, S="S", T="T"):
        """Find one augmenting path using BFS; return (path, bottleneck)."""
        parent = {S: None}
        bn = {S: float("inf")}
        q = deque([S])

        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if v not in parent and self.residual(u, v) > 0:
                    parent[v] = u
                    bn[v] = min(bn[u], self.residual(u, v))
                    if v == T:
                        # Reconstruct path
                        path = []
                        x = T
                        while x is not None:
                            path.append(x)
                            x = parent[x]
                        path.reverse()
                        return path, bn[T]
                    q.append(v)
        return None, 0

    def run(self, verbose: bool = True):
        """
        Returns:
            total (int): total max-flow
            steps (list): list of (path, bottleneck)
            flow (dict):  flow per edge (u,v)
            cap  (dict):  capacity per edge (u,v)
        """
        self.build_capacities()
        steps = []

        while True:
            path, b = self.bfs()
            if not path:
                break
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if (u, v) in self.cap:
                    self.flow[(u, v)] += b
                else:
                    self.flow[(v, u)] -= b
            if verbose:
                steps.append((path, b))

        total = self.flow[("S", "P1")] + self.flow[("S", "P2")]
        return total, steps, dict(self.flow), dict(self.cap)

# =========================
# Autoscaler policy (optional)
# =========================

@dataclass
class AutoScaleConfig:
    target_util: float = 0.85
    scale_up_factor: float = 1.15
    scale_down_factor: float = 0.90
    min_capacity_unit: int = 5

class InferenceAutoScaler:
    """A simple decode-first autoscaler using max-flow utilization signals."""
    def __init__(self, api: FakeAPI, cfg: AutoScaleConfig):
        self.api = api
        self.cfg = cfg

    def _round_unit(self, x: float) -> int:
        unit = self.cfg.min_capacity_unit
        return int(math.ceil(x / unit) * unit)

    def measure(self):
        total, _steps, flow, cap = MaxFlow(self.api).run(verbose=False)
        util = {}
        for d, c in self.api.get_decode().D_to_T.items():
            f = flow.get((d, "T"), 0)
            util[d] = (f / c) if c > 0 else 0.0
        return {"total": total, "util_decode": util, "flow": flow, "cap": cap}

    def step(self):
        rpt = self.measure()
        util = rpt["util_decode"]
        actions = {"decode": {}, "prefill": {}}

        hot = any(u >= self.cfg.target_util for u in util.values())
        if hot:
            # scale decode up
            new_dec = {}
            for d, cur in self.api.get_decode().D_to_T.items():
                if util[d] >= self.cfg.target_util:
                    new_dec[d] = self._round_unit(cur * self.cfg.scale_up_factor)
            if new_dec:
                self.api.set_decode(new_dec)
                actions["decode"] = new_dec
        else:
            # consider scaling down if far from target
            new_dec = {}
            for d, cur in self.api.get_decode().D_to_T.items():
                if util[d] < self.cfg.target_util * 0.6:
                    new_dec[d] = max(self._round_unit(cur * self.cfg.scale_down_factor),
                                     self.cfg.min_capacity_unit)
            if new_dec:
                self.api.set_decode(new_dec)
                actions["decode"] = new_dec

        return {"before": rpt, "actions": actions, "after": self.measure()}

# =========================
# Time-varying simulation for animations.py
# =========================

def _ramp_arrivals(t, T):
    """Smooth ramp-up → spike → decay profile for arrivals."""
    x = t / max(1, T)
    base = 20 + 80 * (1 / (1 + math.exp(-10 * (x - 0.35))))  # Adjusted to peak at 100
    if 0.30 <= x <= 0.36: base *= 1.15
    if 0.42 <= x <= 0.50: base *= 1.35  # main spike softer
    if 0.75 <= x <= 0.90: base *= 0.85
    return max(1.0, base)

def _token_length_factor(t, T):
    """Longer tokens around the spike for additional pressure."""
    x = t / max(1, T)
    return 1.0 + 0.45 * math.sin(min(1.0, max(0.0, (x - 0.25) / 0.4)) * math.pi)


def simulate_series(T: int = 300, sla: int = 50, seed: int = 2025, arrival_boost: float = 2.0):  # Increased arrival boost
    """
    Build a time series for animations:

    Returns a dict with:
      - "tick":  list[int] 0..T-1
      - "arr":   np.ndarray arrivals per tick
      - "tp":    np.ndarray throughput per tick (rough proxy)
      - "p95":   np.ndarray p95 latency proxy per tick (ms)
      - "mf":    np.ndarray max-flow per tick
      - "flows": list[dict] per-edge flow at each tick
      - "caps":  list[dict] per-edge capacity at each tick
      - "events":list[None | (str, dict)] autoscale events ("up"/"down", details)
      - "sla":   int SLA in ms
      - (Optional) arrival_boost: float multiplier for arrivals
    """
    random.seed(seed); np.random.seed(seed)

    # Initial capacities
    api = FakeAPI(
        prefill=PrefillConfig(S_to_P={"P1": 100, "P2": 100}),
        decode=DecodeConfig(D_to_T={"D1": 80, "D2": 80}),
        kv=KVConfig(P_to_D={
            ("P1","D1"): 60, ("P1","D2"): 60,
            ("P2","D1"): 60, ("P2","D2"): 60
        })
    )

    # Autoscale knobs (simple, decode-first)
    target_util = 0.85
    scale_up    = 1.20
    scale_down  = 0.88
    min_unit    = 10
    cooldown    = 12

    cd = 0

    ticks = list(range(T))
    arr, tp, p95, mf = [], [], [], []
    flows_series, caps_series, events = [], [], []

    for t in ticks:
        # --- Max flow under current capacities ---
        total, _steps, flow, cap = MaxFlow(api).run(verbose=False)  # <-- FIX: unpack 4 values
        mf.append(total)
        flows_series.append(flow)
        caps_series.append(cap)

        # --- Workload & rough throughput proxy ---
        # lam = arrival_boost * _ramp_arrivals(t, T)
        lam = _ramp_arrivals(t, T)
        tok = _token_length_factor(t, T)
        dem = int(np.random.poisson(lam))
        dec_cap = sum(api.get_decode().D_to_T.values())
        # thpt = min(dem, int(dec_cap / max(1e-6, tok * 1.1)))  # was tok
        thpt = min(dem, int(dec_cap / max(1e-6, tok * 0.9)))

        arr.append(dem)
        tp.append(thpt)

        # --- Latency proxy (very simple but effective for demo) ---
        rho = min(0.999, dem / max(1, dec_cap))
        mu = dec_cap / 10.0            # assume cycle_ms ~ 10
        lam_ms = dem / 10.0
        if mu > lam_ms:
            p95_ms = 1000.0 * (rho**2) / max(1e-6, (mu - lam_ms))
        else:
            p95_ms = 20.0 + 50.0 * rho * tok
        p95.append(p95_ms)

        # --- Autoscaling decisions (decode-first) ---
        evt = None
        if cd > 0:
            cd -= 1
        else:
            if p95_ms > sla or rho >= target_util:
                # scale-up decode
                new_dec = {}
                for d, cur in api.get_decode().D_to_T.items():
                    v = int(math.ceil(cur * scale_up / min_unit) * min_unit)
                    new_dec[d] = v
                api.set_decode(new_dec)
                evt = ("up", new_dec)
                cd = cooldown
            elif p95_ms < 0.6 * sla and rho < 0.5:
                # scale-down decode
                new_dec = {}
                for d, cur in api.get_decode().D_to_T.items():
                    v = int(math.ceil(cur * scale_down / min_unit) * min_unit)
                    v = max(v, min_unit)
                    new_dec[d] = v
                api.set_decode(new_dec)
                evt = ("down", new_dec)
                cd = cooldown

        events.append(evt)

    return {
        "tick": ticks,
        "arr":  np.array(arr, dtype=float),
        "tp":   np.array(tp,  dtype=float),
        "p95":  np.array(p95, dtype=float),
        "mf":   np.array(mf,  dtype=float),
        "flows": flows_series,
        "caps":  caps_series,
        "events": events,
        "sla": int(sla),
    }

# (Optional) quick manual test
if __name__ == "__main__":
    S = simulate_series(T=30, sla=50, seed=1)
    print("OK. Example keys:", list(S.keys()))
    print("First tick snapshot:", {
        "arr": S["arr"][0],
        "tp":  S["tp"][0],
        "p95": S["p95"][0],
        "mf":  S["mf"][0],
        "event": S["events"][0],
    })
