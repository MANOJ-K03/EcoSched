"""EcoSched - adaptive, energy-aware CPU scheduling simulator (Streamlit dashboard).

Run with:  python -m streamlit run app.py
All scheduling logic lives in scheduler/ and experiments/; this file is only the interface.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pandas as pd
import streamlit as st

import visuals as viz
from experiments.benchmark import (ALGORITHMS, SimConfig, energy_sensitivity, monte_carlo, overhead_grid,
                                   process_table, quantum_sweep, run_all, run_profiles, summary_table,
                                   win_rates, winners)
from experiments.workloads import (PRESETS, PROFILE_SEEDS, classic_demo, df_to_processes, load_preset,
                                   next_pid, processes_to_df)
from scheduler.adaptive import EMIConfig
from scheduler.energy import ENERGY_DISCLAIMER, EnergyModel
from scheduler.insights import METRICS, WEIGHT_PRESETS, composite_scores, interpret
from scheduler.metrics import verify_trace
from scheduler.models import CATEGORIES, ValidationError

st.set_page_config(page_title="EcoSched", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
h1 {font-size: 2.1rem !important; letter-spacing: -0.02em;}
div[data-testid="stMetric"] {background:#171E27; border:1px solid #2A3441; border-radius:6px; padding:12px 14px;}
div[data-testid="stMetricValue"] {font-family: "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace;}
.disclaimer {border:1px solid #FFB454; color:#FFB454; border-radius:6px; padding:6px 12px; font-size:0.85rem; display:inline-block;}
.muted {color:#8A96A6;}
.q {font-size:1.15rem; line-height:1.5; max-width: 62rem;}
</style>
""", unsafe_allow_html=True)

COLUMN_CONFIG = {
    "pid": st.column_config.NumberColumn("PID", min_value=0, step=1, format="%d", help="Unique process ID."),
    "arrival": st.column_config.NumberColumn("Arrival", min_value=0, step=1, format="%d",
                                             help="Tick at which the process enters the ready queue."),
    "burst": st.column_config.NumberColumn("Burst", min_value=1, step=1, format="%d",
                                           help="CPU time the process needs in total (ticks)."),
    "priority": st.column_config.NumberColumn("Priority", min_value=0, max_value=99, step=1, format="%d",
                                              help="Lower number = higher priority."),
    "category": st.column_config.SelectboxColumn("Category", options=list(CATEGORIES), required=True,
                                                 help="Used by EcoMode Intelligence to read the workload mix."),
}


# ------------------------------------------------------------------ state helpers
def canon(df: pd.DataFrame) -> list:
    """Order/format-insensitive snapshot of the table (ints and 3.0 compare equal)."""
    rows = []
    for rec in df.to_dict("records"):
        rows.append(tuple(int(v) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)
                          for v in rec.values()))
    return rows


def signature(df: pd.DataFrame, cfg: SimConfig, include_ref: bool) -> str:
    return json.dumps({"rows": canon(df), "cfg": asdict(cfg), "ref": include_ref}, sort_keys=True, default=str)


def init_state() -> None:
    if "procs_base" not in st.session_state:
        df = processes_to_df(classic_demo())
        st.session_state.update(procs_base=df, procs_edit=df, editor_ver=0, profile="Classic demo", sim=None,
                                bench=None, mc=None)


def set_table(df: pd.DataFrame, profile: str) -> None:
    st.session_state.procs_base = df
    st.session_state.procs_edit = df
    st.session_state.profile = profile
    st.session_state.editor_ver += 1


def effective_profile() -> str:
    """Profile name only counts as a hint while the table still matches the loaded preset."""
    same = canon(st.session_state.procs_edit) == canon(st.session_state.procs_base)
    return st.session_state.profile if same else "Custom"


def execute(df: pd.DataFrame, cfg: SimConfig, include_ref: bool) -> dict:
    procs = df_to_processes(df)
    if not procs:
        raise ValidationError("Add at least one process before running the simulation.")
    profile = effective_profile()
    results, decision = run_all(procs, cfg, profile, include_ref)
    return {
        "sig": signature(df, cfg, include_ref), "procs": procs, "results": results, "decision": decision,
        "summary": summary_table(results, cfg.energy), "ptable": process_table(results), "cfg": cfg,
        "profile": profile, "problems": {n: verify_trace(r) for n, r in results.items()},
        "colors": viz.pid_colors([p.pid for p in procs]),
    }


def sidebar_config() -> tuple[SimConfig, bool]:
    with st.sidebar:
        st.markdown("### Simulation settings")
        quantum = st.number_input("Round Robin time quantum", 1, 100, 4, 1,
                                  help="Fixed slice a process may run before being preempted (baseline Round Robin).")
        cs_cost = st.number_input("Context-switch overhead (ticks)", 0, 10, 1, 1,
                                  help="CPU time lost saving/restoring process state each time the CPU switches process.")
        include_ref = st.toggle("Add SRTF reference scheduler", value=False,
                                help="Shortest Remaining Time First: a classic lower bound for waiting time.")
        with st.expander("Energy model (simulated)"):
            st.caption(ENERGY_DISCLAIMER)
            active = st.number_input("Active CPU power (W)", 0.0, 500.0, 10.0, 0.5)
            idle = st.number_input("Idle CPU power (W)", 0.0, 500.0, 2.0, 0.5)
            sw = st.number_input("Energy per context switch (J)", 0.0, 10.0, 0.05, 0.01, format="%.3f")
            unit = st.number_input("Seconds per tick", 0.0001, 10.0, 0.01, 0.001, format="%.4f",
                                   help="Converts ticks to seconds so power (W) x time (s) gives joules.")
        with st.expander("EcoMode Intelligence rules"):
            override = st.selectbox("Mode", ["Auto (rules decide)", "latency", "balanced", "efficiency"],
                                    help="Auto lets the rules vote; the others force a mode for what-if tests.")
            short_t = st.number_input("A job is 'short' if burst <=", 1, 100, 5, 1)
            f_lat = st.number_input("Latency factor (x median burst)", 0.1, 5.0, 0.5, 0.1)
            f_bal = st.number_input("Balanced factor", 0.1, 5.0, 1.0, 0.1)
            f_eff = st.number_input("Efficiency factor", 0.1, 5.0, 1.5, 0.1)
            amort = st.number_input("Min quantum = k x switch cost, k =", 0, 20, 4, 1,
                                    help="Keeps switch overhead to roughly 1/(k+1) of each slice.")
            grace = st.number_input("Completion grace (x quantum)", 1.0, 3.0, 1.5, 0.1,
                                    help="A job within this multiple of the quantum of finishing is allowed to finish. 1.0 = off.")
            pressure = st.number_input("Queue-pressure threshold (0 = off)", 0, 50, 8, 1,
                                       help="If at least this many jobs are waiting, the quantum is halved.")
    emi = EMIConfig(short_burst_threshold=int(short_t), factor_latency=float(f_lat), factor_balanced=float(f_bal),
                    factor_efficiency=float(f_eff), switch_amortization=int(amort), completion_grace=float(grace),
                    queue_pressure_threshold=int(pressure),
                    mode_override=None if override.startswith("Auto") else override)
    cfg = SimConfig(int(quantum), int(cs_cost), EnergyModel(active, idle, sw, unit), emi)
    return cfg, include_ref


def csv_button(label: str, df: pd.DataFrame, name: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv", key=key)


def table(df: pd.DataFrame, decimals: int = 3) -> None:
    st.dataframe(df.round(decimals), width="stretch", hide_index=True)


# ------------------------------------------------------------------ app
init_state()
cfg, include_ref = sidebar_config()

st.title("EcoSched")
st.markdown('<p class="q">When does an adaptive scheduler actually beat the classic ones, and when does it lose? '
            'EcoSched runs five policies on identical workloads and measures the answer from their execution traces.</p>',
            unsafe_allow_html=True)

tab_names = ["Overview", "Process Lab", "Scheduler Arena", "EcoMode Intelligence", "Performance Observatory",
             "Experiment Lab", "OS Concepts", "About and Limitations"]
tabs = st.tabs(tab_names)

# ---- Process Lab first in code so edits are known before other tabs render
with tabs[1]:
    st.subheader("Process Lab")
    c1, c2, c3 = st.columns([2, 1, 3])
    preset = c1.selectbox("Workload preset", PRESETS, help="Built-in profiles use fixed seeds, so they are reproducible.")
    seed = c2.number_input("Seed (0 = preset default)", 0, 10_000, 0, 1)
    c3.caption("Presets: " + ", ".join(f"{k} (seed {v})" for k, v in PROFILE_SEEDS.items()))

    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button("Load preset"):
        set_table(processes_to_df(load_preset(preset, int(seed) or None)), preset)
    if b2.button("Restore default demo"):
        set_table(processes_to_df(classic_demo()), "Classic demo")
    if b3.button("Reset table"):
        set_table(pd.DataFrame([{"pid": 1, "arrival": 0, "burst": 4, "priority": 5, "category": "batch"}]), "Custom")
    if b4.button("Add process"):
        cur = st.session_state.procs_edit
        row = {"pid": next_pid(cur), "arrival": 0, "burst": 4, "priority": 5, "category": "batch"}
        set_table(pd.concat([cur, pd.DataFrame([row])], ignore_index=True), st.session_state.profile)
    if b5.button("Remove last process"):
        set_table(st.session_state.procs_edit.iloc[:-1].reset_index(drop=True), st.session_state.profile)

    edited = st.data_editor(st.session_state.procs_base, key=f"editor_{st.session_state.editor_ver}",
                            hide_index=True, width="stretch", num_rows="fixed", column_config=COLUMN_CONFIG)
    st.session_state.procs_edit = edited

    try:
        procs_now = df_to_processes(edited)
        valid, msg = True, ""
    except ValidationError as exc:
        procs_now, valid, msg = [], False, str(exc)
    if not valid:
        st.error(f"Fix the table before running: {msg}")
    elif not procs_now:
        st.info("The table is empty. Add a process or load a preset to begin.")
    else:
        total = sum(p.burst for p in procs_now)
        st.caption(f"{len(procs_now)} processes, {total} ticks of CPU demand, arrivals between "
                   f"{min(p.arrival for p in procs_now)} and {max(p.arrival for p in procs_now)}. "
                   f"Priority convention: lower number = higher priority.")
    if st.button("Run simulation", type="primary", disabled=not (valid and procs_now)):
        try:
            st.session_state.sim = execute(edited, cfg, include_ref)
            st.success("Simulation complete. Open Scheduler Arena to compare the schedules.")
        except ValidationError as exc:
            st.error(str(exc))

if st.session_state.sim is None:
    try:
        st.session_state.sim = execute(st.session_state.procs_edit, cfg, include_ref)
    except ValidationError:
        pass
sim = st.session_state.sim
current_sig = None
try:
    current_sig = signature(st.session_state.procs_edit, cfg, include_ref)
except Exception:
    pass


def need_sim() -> bool:
    if sim is None:
        st.info("No results yet. Build a valid workload in Process Lab and press Run simulation.")
        return False
    if current_sig != sim["sig"]:
        st.warning("Inputs or settings changed since the last run. Press Run simulation in Process Lab to refresh.")
    return True


# ---- Overview
with tabs[0]:
    if need_sim():
        d = sim["decision"]
        st.plotly_chart(viz.gantt(sim["results"], sim["colors"]), width="stretch", key="ov_gantt")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Workload", sim["profile"], help="Preset name, or Custom once the table is edited.")
        k2.metric("Processes", len(sim["procs"]))
        bad = {n: p for n, p in sim["problems"].items() if p}
        k3.metric("Trace verification", "all passed" if not bad else f"{len(bad)} failed",
                  help="Every trace is re-checked: no overlaps, no run before arrival, full burst executed once.")
        k4.metric("EMI decision", f"{d.mode}, q={d.quantum}")
        st.markdown(f'<span class="disclaimer">{ENERGY_DISCLAIMER}</span>', unsafe_allow_html=True)
        st.markdown("""
**What happens here.** Each scheduler gets the same processes and the same context-switch cost.
The simulator records every CPU interval (run, idle, context switch) and computes all metrics from that record.
EcoMode Intelligence reads the workload, chooses a Round Robin quantum with transparent rules,
and is judged by the same measurements as the classic algorithms, including the cases where it loses.
""")
        for line in interpret(sim["summary"])[:3]:
            st.markdown(f"- {line}")

# ---- Scheduler Arena
with tabs[2]:
    st.subheader("Scheduler Arena")
    if need_sim():
        res, summ = sim["results"], sim["summary"]
        cols = st.columns(len(res))
        for col, (_, row) in zip(cols, summ.iterrows()):
            with col:
                st.markdown(f"**{row['algorithm']}**")
                st.metric("Avg waiting", f"{row['avg_wait']:.2f}", help="Time spent ready but not running.")
                st.metric("Avg response", f"{row['avg_response']:.2f}", help="Arrival until first CPU use.")
                st.metric("Switches", int(row["context_switches"]))
                st.metric("Energy (sim.)", f"{row['energy_j']:.2f} J")
        st.plotly_chart(viz.gantt(res, sim["colors"], 74), width="stretch", key="arena_gantt")
        st.caption("Colours identify processes in every chart. Dark grey = CPU idle, red = context-switch overhead. "
                   "Hover a block for exact timestamps.")
        pick = st.selectbox("Inspect one schedule", list(res), key="arena_pick")
        seg_rows = []
        for s in res[pick].trace:
            seg_rows.append({"interval": f"{s.start} to {s.end}", "start": s.start, "end": s.end, "duration": s.duration,
                             "what": f"P{s.pid}" if s.is_run else ("CPU idle" if s.is_idle else f"switch to P{s.pid}"),
                             "event": ("completes" if s.completes else "preempted") if s.is_run else ""})
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Timeline**")
            st.dataframe(pd.DataFrame(seg_rows), width="stretch", hide_index=True)
        with t2:
            st.markdown("**Per-process metrics**")
            pt = sim["ptable"]
            table(pt[pt["algorithm"] == pick].drop(columns=["algorithm", "pid"]))

# ---- EcoMode Intelligence
with tabs[3]:
    st.subheader("EcoMode Intelligence")
    if need_sim():
        d = sim["decision"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mode", d.mode + (" (override)" if d.overridden else ""))
        m2.metric("Quantum", d.quantum, help="Ticks a process may run before preemption.")
        m3.metric("Raw quantum", d.raw_quantum, help="factor x median burst, before the floor and cap.")
        m4.metric("Switch-cost floor", d.quantum_floor, help="Minimum quantum so that overhead stays a small share of each slice.")
        st.info(d.explanation)
        l, r = st.columns(2)
        with l:
            st.markdown("**Workload analysis**")
            feats = pd.DataFrame({"feature": list(d.features), "value": [float(v) for v in d.features.values()]})
            table(feats, 3)
            st.caption("Points per mode: " + ", ".join(f"{k} {v}" for k, v in d.scores.items())
                       + ". Ties go to balanced.")
        with r:
            st.markdown("**Rule evidence**")
            ev = pd.DataFrame(d.rules)
            ev["fired"] = ev["fired"].map({True: "yes", False: "no"})
            st.dataframe(ev, width="stretch", hide_index=True)
        st.markdown("**Trade-offs of each mode**")
        st.markdown("""
- **Latency**: short quantum. Jobs get the CPU sooner (lower response), at the price of more context switches and more switch energy.
- **Balanced**: moderate quantum. A compromise when job lengths are mixed.
- **Efficiency**: long quantum. Fewer switches and less overhead, but short jobs may wait behind long ones (FCFS-like behaviour).
""")
        top_q = int(min(30, max(p.burst for p in sim["procs"])))
        sweep = quantum_sweep(sim["procs"], sim["cfg"], top_q)
        emi_row = sim["summary"].set_index("algorithm").loc["EMI"].to_dict()
        st.markdown("**Where does EMI's choice sit among every fixed quantum?**")
        st.plotly_chart(viz.quantum_sweep(sweep, emi_row, d.quantum), width="stretch", key="emi_sweep")
        st.caption("The diamond is EMI's measured result. It can differ from the Round Robin line at the same quantum "
                   "because of the completion-grace and queue-pressure rules.")

# ---- Performance Observatory
with tabs[4]:
    st.subheader("Performance Observatory")
    if need_sim():
        summ = sim["summary"]
        st.plotly_chart(viz.time_metrics(summ), width="stretch", key="obs_time")
        row1 = st.columns(3)
        row1[0].plotly_chart(viz.metric_bar(summ, "throughput", "Throughput"), width="stretch", key="obs_thr")
        row1[1].plotly_chart(viz.metric_bar(summ, "utilization", "CPU utilization"), width="stretch", key="obs_util")
        row1[2].plotly_chart(viz.metric_bar(summ, "context_switches", "Context switches"), width="stretch", key="obs_cs")
        row2 = st.columns(3)
        row2[0].plotly_chart(viz.metric_bar(summ, "fairness", "Jain fairness"), width="stretch", key="obs_fair")
        row2[1].plotly_chart(viz.energy_stack(summ), width="stretch", key="obs_energy")
        row2[2].plotly_chart(viz.pareto(summ), width="stretch", key="obs_pareto")
        st.caption("Fairness: Jain's index over each job's CPU share, burst / (burst + waiting). 1 means all jobs were "
                   "slowed equally; it does not account for priorities. Utilization counts idle and switch time as elapsed time.")
        st.markdown("**Decision advisor**")
        preset_w = st.selectbox("What matters most?", list(WEIGHT_PRESETS), key="weights",
                                help="Each metric is rescaled 0-1 across the compared schedulers, then weighted.")
        scores = composite_scores(summ, WEIGHT_PRESETS[preset_w])
        st.plotly_chart(viz.score_bar(list(summ["algorithm"]), list(scores)), width="stretch", key="obs_score")
        top = summ["algorithm"][scores.idxmax()]
        st.caption(f"Highest relative score under '{preset_w}': {top}. This ranks only the schedulers shown, on this workload.")
        st.markdown("**Findings computed from these results**")
        for line in interpret(summ):
            st.markdown(f"- {line}")
        table(summ.drop(columns=["energy_active_j", "energy_idle_j", "energy_switch_j"]))
        csv_button("Download comparison CSV", summ, "ecosched_comparison.csv", "dl_summary")
        csv_button("Download process-level CSV", sim["ptable"], "ecosched_process_metrics.csv", "dl_proc")

# ---- Experiment Lab
with tabs[5]:
    st.subheader("Experiment Lab")
    st.markdown("**1. Benchmark every built-in workload**")
    if st.button("Run benchmark on all profiles"):
        st.session_state.bench = run_profiles(cfg)
    bench = st.session_state.bench
    if bench is None:
        st.info("Press the button to run 4 profiles x 5 schedulers with the current settings.")
    else:
        table(bench.drop(columns=["energy_active_j", "energy_idle_j", "energy_switch_j"]))
        st.markdown("Best scheduler per profile and metric (ties listed together)")
        st.dataframe(winners(bench), width="stretch", hide_index=True)
        csv_button("Download benchmark CSV", bench, "ecosched_benchmark.csv", "dl_bench")
        inspect = st.selectbox("Inspect a profile", list(PROFILE_SEEDS), key="inspect_profile")
        sub_res, _ = run_all(load_preset(inspect), cfg, inspect)
        st.plotly_chart(viz.gantt(sub_res, viz.pid_colors([p.pid for p in load_preset(inspect)]), 60),
                        width="stretch", key="bench_gantt")
        st.dataframe(bench[bench["profile"] == inspect][["algorithm", "avg_wait", "avg_response", "context_switches",
                                                          "fairness", "energy_j"]].round(3),
                     width="stretch", hide_index=True)

    st.markdown("**2. Robustness across random seeds**")
    mc1, mc2 = st.columns(2)
    mc_profile = mc1.selectbox("Profile", list(PROFILE_SEEDS), index=2, key="mc_profile")
    mc_n = mc2.slider("Number of seeds", 5, 100, 30, 5, help="Each seed generates a different workload of the same type.")
    if st.button("Run Monte-Carlo experiment"):
        st.session_state.mc = (mc_profile, mc_n, monte_carlo(mc_profile, mc_n, cfg))
    if st.session_state.mc is not None:
        prof, n, mc = st.session_state.mc
        st.caption(f"{n} seeds of the {prof} profile, seeds 1000 to {1000 + n - 1}.")
        wr = win_rates(mc)
        table(wr, 2)
        metric_key = st.selectbox("Distribution of", list(METRICS), format_func=lambda k: METRICS[k][0], key="mc_metric")
        st.plotly_chart(viz.seed_box(mc, metric_key), width="stretch", key="mc_box")
        csv_button("Download seed-level CSV", mc, "ecosched_monte_carlo.csv", "dl_mc")

    st.markdown("**3. Sensitivity of the energy comparison (current workload)**")
    if sim is None:
        st.info("Run a simulation first.")
    else:
        max_sj = st.slider("Maximum energy per switch (J)", 0.1, 5.0, 1.0, 0.1)
        sj_values = [round(max_sj * i / 10, 4) for i in range(0, 11)]
        sens = energy_sensitivity(sim["procs"], sim["cfg"], sj_values, sim["profile"])
        st.plotly_chart(viz.sensitivity_lines(sens), width="stretch", key="sens_lines")
        low_e = sens.loc[sens.groupby("switch_j")["energy_j"].idxmin()][["switch_j", "algorithm"]]
        changes = low_e["algorithm"].ne(low_e["algorithm"].shift()).sum() - 1
        st.caption("Lowest-energy scheduler as switch energy grows: "
                   + ", ".join(f"{r.switch_j:g} J: {r.algorithm}" for r in low_e.itertuples())
                   + f". The leader changed {max(int(changes), 0)} time(s) across this range.")
        grid = overhead_grid(sim["procs"], sim["cfg"], [0, 1, 2, 3, 4], [0.0, 0.05, 0.1, 0.25, 0.5, 1.0], sim["profile"])
        st.plotly_chart(viz.overhead_heatmap(grid), width="stretch", key="sens_heat")
        csv_button("Download sensitivity CSV", sens, "ecosched_sensitivity.csv", "dl_sens")

# ---- OS Concepts
CONCEPTS = {
    "First Come, First Served (FCFS)": "Runs processes in arrival order until each finishes. Simple and fair in order, but a long job at the front makes everything behind it wait (the convoy effect).",
    "Shortest Job First (SJF)": "Among the processes that have arrived, picks the one with the shortest burst and runs it to completion. Minimises average waiting time for non-preemptive scheduling, but long jobs can starve.",
    "Round Robin": "Each process gets a time quantum, then goes to the back of the ready queue. Good response time; too small a quantum wastes time on switches, too large turns it into FCFS.",
    "Priority scheduling": "The ready process with the highest priority runs. Here a lower number means higher priority, and a newly arrived higher-priority process preempts the running one. Low-priority jobs can starve.",
    "Preemption": "The OS takes the CPU away from a running process before it finishes, because its quantum expired or a more important process arrived.",
    "Context switching": "Saving one process's CPU state and loading another's. It does no useful work, costs time, and in this model also costs energy. EcoSched counts a switch whenever the CPU moves directly between two different processes.",
    "Waiting time": "Turnaround minus burst: the time a process sat ready but not running.",
    "Turnaround time": "Completion time minus arrival time: how long the user waits for a job from submission to finish.",
    "Response time": "Time from arrival to the first moment on the CPU. This is what makes a system feel responsive.",
    "CPU utilization": "Busy time divided by elapsed time. Idle gaps and context-switch overhead count as elapsed time, so heavy switching lowers it.",
    "Energy-model assumptions": "Energy = active power x run time + idle power x idle time + energy per switch x switches. The numbers are adjustable assumptions. Nothing is measured from hardware, and real CPUs add effects such as frequency scaling and sleep states that this model ignores.",
}
with tabs[6]:
    st.subheader("OS Concepts")
    for title, body in CONCEPTS.items():
        with st.expander(title):
            st.write(body)

# ---- About
with tabs[7]:
    st.subheader("About and Limitations")
    st.markdown("**Architecture**")
    st.code("""app.py (Streamlit UI)  ->  visuals.py (Plotly)
        |
        +-> experiments/benchmark.py, workloads.py   (runs, profiles, Monte-Carlo, sensitivity)
                 |
                 +-> scheduler/algorithms.py   FCFS, SJF, Round Robin, Priority, SRTF
                 +-> scheduler/adaptive.py     EcoMode Intelligence
                 +-> scheduler/metrics.py      metrics + trace verifier   <- computed only from traces
                 +-> scheduler/energy.py       simulated energy model
                 +-> scheduler/insights.py     composite score + generated findings""", language="text")
    st.markdown("""
**Conventions.** Ties break by arrival time, then process ID. The clock starts at the earliest arrival. A context switch is
counted when the CPU moves directly from one process to a different one with no idle gap; the first dispatch is never a switch.
Utilization = run time / elapsed time. Throughput = completed jobs / elapsed time. Waiting = turnaround - burst (no I/O modelled).

**EMI rules.** Six transparent rules vote for latency, balanced or efficiency; the quantum is a factor times the median burst,
raised to a switch-cost floor and capped. At run time, a job close to finishing may finish (completion grace), and the quantum is
halved when many jobs wait (queue pressure). All thresholds are editable in the sidebar.

**Limitations.** Single CPU core; no I/O blocking, memory, cache, or multi-core effects; integer ticks; synthetic workloads;
the energy figures are assumptions rather than measurements. A win on one workload is not a general proof, which is why the
Experiment Lab repeats runs over many seeds and reports where EMI loses.

**Reproduce.** Presets use fixed seeds (shown in Process Lab). Monte-Carlo runs use seeds 1000 upward. The test suite runs with
`python -m pytest -q`.
""")
