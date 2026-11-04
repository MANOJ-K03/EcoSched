"""Plotly figure builders. Pure functions: data in, figure out (no Streamlit calls)."""
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scheduler.insights import METRICS
from scheduler.models import SimulationResult

INK, PANEL, GRID, MUTED = "#0F141B", "#171E27", "#2A3441", "#8A96A6"
AMBER = "#FFB454"          # EMI / brand
IDLE_COLOR, SWITCH_COLOR = "#2A3340", "#E5484D"
PROCESS_PALETTE = ["#5B8DEF", "#3FB8A5", "#A78BFA", "#F472B6", "#7DD3FC",
                   "#86C56B", "#C792EA", "#4FD1C5", "#94A3F8", "#F28BA8"]
ALGO_COLORS = {"FCFS": "#5E6E85", "SJF": "#6F89B0", "Round Robin": "#7FA3D6", "Priority": "#4F9DA6",
               "EMI": AMBER, "SRTF (reference)": "#8E7FB5"}


def pid_colors(pids: Iterable[int]) -> Dict[int, str]:
    """One stable colour per process, identical in every chart."""
    return {pid: PROCESS_PALETTE[i % len(PROCESS_PALETTE)] for i, pid in enumerate(sorted(pids))}


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EAF0", size=12), margin=dict(l=10, r=10, t=40, b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def gantt(results: Dict[str, SimulationResult], colors: Dict[int, str], row_height: int = 64) -> go.Figure:
    """One row per algorithm. Processes keep their colour; idle = dark grey, switch = red."""
    fig = go.Figure()
    shown = set()
    for algo, res in results.items():
        for seg in res.trace:
            if seg.is_run:
                name, color, text = f"P{seg.pid}", colors[seg.pid], f"P{seg.pid}"
                note = "completes" if seg.completes else "preempted"
            elif seg.is_idle:
                name, color, text, note = "CPU idle", IDLE_COLOR, "", "no ready process"
            else:
                name, color, text, note = "context switch", SWITCH_COLOR, "", f"dispatching P{seg.pid}"
            fig.add_trace(go.Bar(
                y=[algo], x=[seg.duration], base=[seg.start], orientation="h", name=name, legendgroup=name,
                showlegend=name not in shown, text=[text], textposition="inside", insidetextanchor="middle",
                textfont=dict(color=INK, size=11), marker=dict(color=color, line=dict(color=INK, width=1)),
                hovertemplate=f"<b>{name}</b> ({note})<br>{algo}: t={seg.start} to {seg.end} "
                              f"(duration {seg.duration})<extra></extra>",
            ))
            shown.add(name)
    fig.update_layout(barmode="overlay", bargap=0.25)
    fig.update_yaxes(autorange="reversed", title_text="")
    fig.update_xaxes(title_text="time (ticks)", rangemode="tozero")
    return style(fig, height=max(200, row_height * len(results) + 90))


def _colors(algos: List[str]) -> List[str]:
    return [ALGO_COLORS.get(a, "#7A8699") for a in algos]


def time_metrics(summary: pd.DataFrame) -> go.Figure:
    """Waiting / turnaround / response / max wait share one unit (ticks), so one axis is honest."""
    fig = go.Figure()
    for key, shade in (("avg_wait", "#5B8DEF"), ("avg_turnaround", "#7DD3FC"),
                       ("avg_response", "#3FB8A5"), ("max_wait", "#F28BA8")):
        fig.add_trace(go.Bar(x=summary["algorithm"], y=summary[key], name=METRICS[key][0], marker_color=shade))
    fig.update_layout(barmode="group", title="Time metrics (ticks, lower is better)")
    return style(fig, 380)


def metric_bar(summary: pd.DataFrame, key: str, title: str) -> go.Figure:
    label, direction, unit = METRICS[key]
    fig = go.Figure(go.Bar(x=summary["algorithm"], y=summary[key], marker_color=_colors(list(summary["algorithm"])),
                           text=summary[key].round(3), textposition="outside", cliponaxis=False))
    arrow = "lower is better" if direction == "min" else "higher is better"
    fig.update_layout(title=f"{title} ({unit}, {arrow})")
    return style(fig, 300, legend=False)


def energy_stack(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name, color in (("energy_active_j", "active", "#5B8DEF"), ("energy_idle_j", "idle", "#6B7A90"),
                             ("energy_switch_j", "context switches", SWITCH_COLOR)):
        fig.add_trace(go.Bar(x=summary["algorithm"], y=summary[col], name=name, marker_color=color))
    fig.update_layout(barmode="stack", title="Simulated energy by component (J) - not measured power")
    return style(fig, 300)


def pareto(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for _, row in summary.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["avg_response"]], y=[row["energy_j"]], mode="markers+text", name=row["algorithm"],
            text=[row["algorithm"]], textposition="top center",
            marker=dict(size=16, color=ALGO_COLORS.get(row["algorithm"], "#7A8699"), line=dict(color=INK, width=1))))
    fig.update_layout(title="Responsiveness vs simulated energy (bottom-left is better)",
                      xaxis_title="average response time (ticks)", yaxis_title="simulated energy (J)")
    return style(fig, 360, legend=False)


def score_bar(names: List[str], scores: List[float]) -> go.Figure:
    order = sorted(range(len(names)), key=lambda i: scores[i])
    fig = go.Figure(go.Bar(x=[scores[i] for i in order], y=[names[i] for i in order], orientation="h",
                           marker_color=_colors([names[i] for i in order]),
                           text=[f"{scores[i]:.2f}" for i in order], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Weighted relative score (1 = best of the group on every weighted metric)")
    fig.update_xaxes(range=[0, 1.1])
    return style(fig, 280, legend=False)


def quantum_sweep(sweep: pd.DataFrame, emi_row: dict, emi_q: int) -> go.Figure:
    panels = [("avg_wait", "Average waiting (ticks)"), ("avg_response", "Average response (ticks)"),
              ("context_switches", "Context switches"), ("energy_j", "Simulated energy (J)")]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[1] for p in panels])
    for i, (key, _) in enumerate(panels):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(x=sweep["quantum"], y=sweep[key], mode="lines+markers", name="Round Robin, fixed quantum",
                                 line=dict(color=ALGO_COLORS["Round Robin"]), showlegend=(i == 0)), row=r, col=c)
        fig.add_trace(go.Scatter(x=[emi_q], y=[emi_row[key]], mode="markers", name="EMI (with grace / pressure rules)",
                                 marker=dict(size=13, color=AMBER, symbol="diamond", line=dict(color=INK, width=1)),
                                 showlegend=(i == 0)), row=r, col=c)
    fig.update_xaxes(title_text="quantum")
    return style(fig, 520)


def sensitivity_lines(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for algo, sub in df.groupby("algorithm", sort=False):
        fig.add_trace(go.Scatter(x=sub["switch_j"], y=sub["energy_j"], mode="lines+markers", name=algo,
                                 line=dict(color=ALGO_COLORS.get(algo, "#7A8699"), width=3 if algo == "EMI" else 2)))
    fig.update_layout(title="Simulated energy as energy-per-switch changes",
                      xaxis_title="energy per context switch (J)", yaxis_title="simulated energy (J)")
    return style(fig, 360)


def overhead_heatmap(grid: pd.DataFrame) -> go.Figure:
    piv = grid.pivot(index="cs_cost", columns="switch_j", values="diff_pct")
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[f"{c:g}" for c in piv.columns], y=[str(i) for i in piv.index], zmid=0,
        colorscale=[[0, "#3FB8A5"], [0.5, "#1B2430"], [1, "#E5484D"]],
        colorbar=dict(title="EMI vs RR (%)"),
        hovertemplate="switch time %{y}, switch energy %{x} J<br>EMI energy vs RR: %{z:.1f}%<extra></extra>"))
    fig.update_layout(title="EMI energy relative to Round Robin (negative = EMI uses less)",
                      xaxis_title="energy per switch (J)", yaxis_title="switch time cost (ticks)")
    return style(fig, 340, legend=False)


def seed_box(mc: pd.DataFrame, key: str) -> go.Figure:
    fig = go.Figure()
    for algo in mc["algorithm"].unique():
        fig.add_trace(go.Box(y=mc.loc[mc["algorithm"] == algo, key], name=algo,
                             marker_color=ALGO_COLORS.get(algo, "#7A8699"), boxmean=True))
    fig.update_layout(title=f"{METRICS[key][0]} across seeds")
    return style(fig, 360, legend=False)
