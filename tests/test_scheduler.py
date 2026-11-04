"""EcoSched test-suite. Expected values below were calculated by hand (see comments)."""
import pytest

from experiments.benchmark import ALGORITHMS, SimConfig, monte_carlo, run_all, win_rates
from experiments.workloads import PROFILE_SEEDS, classic_demo, generate
from scheduler.adaptive import EMIConfig, decide, emi_schedule
from scheduler.algorithms import fcfs, priority_preemptive, round_robin, sjf, srtf
from scheduler.energy import EnergyModel, estimate_energy
from scheduler.insights import WEIGHT_PRESETS, composite_scores, interpret, ranks
from scheduler.metrics import count_context_switches, jain_index, process_metrics, safe_div, summarize, verify_trace
from scheduler.models import Process, ValidationError, validate_processes

ENERGY = EnergyModel(active_w=10, idle_w=2, switch_j=0.5, time_unit_s=1.0)


def P(pid, arrival, burst, priority=5, category="batch"):
    return Process(pid, arrival, burst, priority, category)


def runs(result):
    """[(pid, start, end)] for run segments."""
    return [(s.pid, s.start, s.end) for s in result.trace if s.is_run]


def by_pid(result):
    return {r["pid"]: r for r in process_metrics(result)}


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- FCFS
def test_fcfs_order_and_metrics():
    # P1(0,5) P2(1,3) P3(2,8): runs 0-5, 5-8, 8-16 -> completion 5,8,16; waiting 0,4,6
    res = fcfs([P(3, 2, 8), P(1, 0, 5), P(2, 1, 3)])
    assert runs(res) == [(1, 0, 5), (2, 5, 8), (3, 8, 16)]
    m = by_pid(res)
    assert [m[i]["completion"] for i in (1, 2, 3)] == [5, 8, 16]
    assert [m[i]["turnaround"] for i in (1, 2, 3)] == [5, 7, 14]
    assert [m[i]["waiting"] for i in (1, 2, 3)] == [0, 4, 6]
    assert [m[i]["response"] for i in (1, 2, 3)] == [0, 4, 6]
    assert close(summarize(res, ENERGY)["avg_wait"], 10 / 3)


# ---------------------------------------------------------------- SJF
def test_sjf_is_non_preemptive_and_picks_shortest_arrived():
    # P1(0,7) P2(2,4) P3(4,1) P4(5,4): P1 0-7 (not preempted), then P3(1), P2(4, earlier), P4
    res = sjf([P(1, 0, 7), P(2, 2, 4), P(3, 4, 1), P(4, 5, 4)])
    assert runs(res) == [(1, 0, 7), (3, 7, 8), (2, 8, 12), (4, 12, 16)]
    assert all(not s.preempted for s in res.run_segments)


# ---------------------------------------------------------------- Round Robin
def test_round_robin_preemption_requeue_and_boundary_arrival():
    # q=2: P1(0,5) P2(1,3) P3(2,1). P3 arrives exactly when P1's slice ends and is queued BEFORE P1.
    res = round_robin([P(1, 0, 5), P(2, 1, 3), P(3, 2, 1)], quantum=2)
    assert runs(res) == [(1, 0, 2), (2, 2, 4), (3, 4, 5), (1, 5, 7), (2, 7, 8), (1, 8, 9)]
    assert [s.preempted for s in res.run_segments] == [True, True, False, True, False, False]
    m = by_pid(res)
    assert [m[i]["completion"] for i in (1, 2, 3)] == [9, 8, 5]
    assert [m[i]["waiting"] for i in (1, 2, 3)] == [4, 4, 2]
    assert count_context_switches(res) == 5


def test_round_robin_single_process_has_no_switch_and_merges():
    res = round_robin([P(1, 0, 7)], quantum=2, cs_cost=3)
    assert runs(res) == [(1, 0, 7)]
    assert count_context_switches(res) == 0


# ---------------------------------------------------------------- Priority
def test_priority_preemption_hand_calculated():
    # P1(0,4,prio3) P2(1,2,prio1) P3(2,1,prio2). Lower number = higher priority.
    # 0-1 P1 | 1-3 P2 (preempts P1) | 3-4 P3 | 4-7 P1
    res = priority_preemptive([P(1, 0, 4, 3), P(2, 1, 2, 1), P(3, 2, 1, 2)])
    assert runs(res) == [(1, 0, 1), (2, 1, 3), (3, 3, 4), (1, 4, 7)]
    assert res.run_segments[0].preempted is True
    m = by_pid(res)
    assert [m[i]["waiting"] for i in (1, 2, 3)] == [3, 0, 1]
    assert [m[i]["response"] for i in (1, 2, 3)] == [0, 0, 1]


def test_srtf_reference_preempts_on_shorter_remaining():
    res = srtf([P(1, 0, 8), P(2, 1, 2)])
    assert runs(res) == [(1, 0, 1), (2, 1, 3), (1, 3, 10)]


# ---------------------------------------------------------------- arrivals, idle, edge cases
def test_simultaneous_arrivals_break_ties_by_pid():
    res = fcfs([P(2, 0, 3), P(1, 0, 3)])
    assert [r[0] for r in runs(res)] == [1, 2]
    res = priority_preemptive([P(2, 0, 3, 4), P(1, 0, 3, 4)])
    assert runs(res)[0][0] == 1


def test_multiple_processes_at_time_zero_sjf():
    res = sjf([P(1, 0, 6), P(2, 0, 2), P(3, 0, 4)])
    assert [r[0] for r in runs(res)] == [2, 3, 1]


def test_idle_period_and_late_arrival():
    res = fcfs([P(1, 0, 2), P(2, 5, 2)])
    assert [(s.kind, s.start, s.end) for s in res.trace] == [("run", 0, 2), ("idle", 2, 5), ("run", 5, 7)]
    s = summarize(res, ENERGY)
    assert close(s["utilization"], 4 / 7) and s["idle_time"] == 3
    assert count_context_switches(res) == 0  # idle gap -> not a switch


def test_single_process_arriving_late_starts_clock_at_arrival():
    res = fcfs([P(1, 3, 2)])
    assert [(s.kind, s.start, s.end) for s in res.trace] == [("run", 3, 5)]
    assert close(summarize(res, ENERGY)["utilization"], 1.0)


def test_preemptive_idle_gap():
    res = priority_preemptive([P(1, 0, 2, 1), P(2, 6, 1, 1)])
    assert [(s.kind, s.start, s.end) for s in res.trace] == [("run", 0, 2), ("idle", 2, 6), ("run", 6, 7)]


# ---------------------------------------------------------------- context switches / overhead
def test_context_switch_overhead_segment_and_metrics():
    res = fcfs([P(1, 0, 2), P(2, 0, 2)], cs_cost=1)
    assert [(s.kind, s.start, s.end) for s in res.trace] == [("run", 0, 2), ("switch", 2, 3), ("run", 3, 5)]
    s = summarize(res, ENERGY)
    assert s["context_switches"] == 1 and s["switch_time"] == 1
    assert close(s["utilization"], 4 / 5)
    assert close(s["energy_j"], 10 * 4 + 0.5)       # 40 J active + 0.5 J for one switch
    assert by_pid(res)[2]["response"] == 3          # overhead delays the second process


def test_no_switch_before_first_process():
    for algo in (fcfs, sjf, priority_preemptive):
        assert not algo([P(1, 0, 3)], cs_cost=2).trace[0].is_switch


# ---------------------------------------------------------------- invalid input / zero denominators
def test_invalid_processes_rejected():
    bad = [dict(pid=1, arrival=-1, burst=2), dict(pid=1, arrival=0, burst=0), dict(pid=1, arrival=0, burst=1.5),
           dict(pid=-1, arrival=0, burst=2), dict(pid=1, arrival=0, burst=2, priority=100),
           dict(pid=1, arrival=0, burst=2, category="gaming"), dict(pid=1, arrival=0, burst=True),
           dict(pid=1, arrival=0, burst=float("nan")), dict(pid=1, arrival=0, burst="abc")]
    for kwargs in bad:
        with pytest.raises(ValidationError):
            Process(**kwargs)


def test_duplicate_ids_and_bad_parameters():
    with pytest.raises(ValidationError):
        fcfs([P(1, 0, 2), P(1, 1, 2)])
    with pytest.raises(ValidationError):
        round_robin([P(1, 0, 2)], quantum=0)
    with pytest.raises(ValidationError):
        fcfs([P(1, 0, 2)], cs_cost=-1)
    with pytest.raises(ValidationError):
        validate_processes(["not a process"])


def test_integer_like_floats_and_category_aliases_accepted():
    p = Process(1.0, 2.0, 3.0, 4.0, "CPU_intensive")
    assert (p.pid, p.arrival, p.burst, p.priority, p.category) == (1, 2, 3, 4, "cpu-intensive")


def test_empty_workload_and_zero_denominators():
    for algo in (fcfs, sjf, priority_preemptive, srtf):
        res = algo([])
        assert res.trace == []
        s = summarize(res, ENERGY)
        assert s["avg_wait"] == 0 and s["utilization"] == 0 and s["throughput"] == 0 and s["fairness"] == 0
    assert round_robin([], 3).trace == []
    assert safe_div(1, 0) == 0.0 and jain_index([]) == 0.0
    assert emi_schedule([], 1)[0].trace == []


# ---------------------------------------------------------------- fairness & energy arithmetic
def test_jain_index_values():
    assert close(jain_index([1, 1, 1]), 1.0)
    assert close(jain_index([1.0, 0.5]), 0.9)        # (1.5^2) / (2 * 1.25)
    assert close(jain_index([1, 0, 0, 0]), 0.25)     # one hog: 1/n


def test_energy_arithmetic():
    model = EnergyModel(active_w=10, idle_w=2, switch_j=0.5, time_unit_s=0.5)
    e = estimate_energy(model, run_time=4, idle_time=3, switches=2)
    assert close(e.active_j, 20) and close(e.idle_j, 3) and close(e.switch_j, 1) and close(e.total_j, 24)
    with pytest.raises(ValueError):
        EnergyModel(active_w=-1)


# ---------------------------------------------------------------- adaptive policy
def test_adaptive_mode_selection():
    cfg = EMIConfig()
    interactive = [P(i, i, 2, 1, "interactive") for i in range(1, 7)]
    batch = [P(i, 0, 20, 5, "cpu-intensive") for i in range(1, 5)]
    assert decide(interactive, 0, cfg).mode == "latency"
    assert decide(batch, 0, cfg).mode == "efficiency"
    # 5 interactive + 5 batch (latency 2 pts) but wide burst spread (balanced 1) + 'Mixed' hint (balanced 1)
    mixed = [P(i, 0, 6, 1, "interactive") for i in range(1, 6)] + [P(i, 0, 6, 5, "batch") for i in range(6, 10)] \
        + [P(10, 0, 200, 5, "cpu-intensive")]
    d = decide(mixed, 0, cfg, profile_hint="Mixed")
    assert d.scores == {"latency": 2, "balanced": 2, "efficiency": 0}
    assert d.mode == "balanced"                          # documented tie-break: balanced wins ties
    assert decide(mixed, 0, cfg).mode == "latency"       # without the hint latency leads 2 vs 1


def test_adaptive_quantum_rules_override_and_floor():
    batch = [P(i, 0, 20, 5, "cpu-intensive") for i in range(1, 5)]
    d = decide(batch, 0)
    assert d.quantum == 30 and d.raw_quantum == 30       # ceil(1.5 * median 20)
    d = decide(batch, 0, EMIConfig(mode_override="latency"))
    assert d.mode == "latency" and d.overridden and d.quantum == 10
    d = decide([P(1, 0, 2, 1, "interactive")], cs_cost=3)  # floor = 4 * 3 = 12 beats raw quantum 1
    assert d.quantum == 12 and d.quantum_floor == 12
    with pytest.raises(ValidationError):
        decide(batch, 0, EMIConfig(mode_override="turbo"))
    assert len(d.rules) == 6 and d.explanation


def test_emi_completion_grace_skips_pointless_switch():
    # quantum forced to 4 with grace 1.5: remaining 6 <= 6 -> job finishes in one slice
    cfg = EMIConfig(mode_override="balanced", factor_balanced=1.0, completion_grace=1.5)
    res, d = emi_schedule([P(1, 0, 6, 5, "batch"), P(2, 0, 4, 5, "batch")], 0, cfg)
    assert d.quantum == 5                    # ceil(1.0 * median(6,4)=5)
    assert runs(res)[0] == (1, 0, 6)         # 6 <= 1.5*5, no preemption
    cfg_off = EMIConfig(mode_override="balanced", completion_grace=1.0)
    res_off, _ = emi_schedule([P(1, 0, 6, 5, "batch"), P(2, 0, 4, 5, "batch")], 0, cfg_off)
    assert runs(res_off)[0] == (1, 0, 5)     # grace disabled -> preempted after quantum


def test_emi_queue_pressure_shrinks_quantum():
    procs = [P(i, 0, 12, 5, "batch") for i in range(1, 6)]
    base = EMIConfig(mode_override="balanced", completion_grace=1.0, queue_pressure_threshold=0)
    pressed = EMIConfig(mode_override="balanced", completion_grace=1.0, queue_pressure_threshold=3)
    q = emi_schedule(procs, 0, base)[1].quantum
    first_base = runs(emi_schedule(procs, 0, base)[0])[0]
    first_pressed = runs(emi_schedule(procs, 0, pressed)[0])[0]
    assert first_base[2] - first_base[1] == q
    assert first_pressed[2] - first_pressed[1] == q // 2


# ---------------------------------------------------------------- invariants for every scheduler
def _all_results(procs, cs, quantum=3):
    cfg = SimConfig(quantum=quantum, cs_cost=cs)
    res, _ = run_all(procs, cfg, include_reference=True)
    return res


def _check_invariants(procs, res):
    by_id = {p.pid: p for p in procs}
    trace = res.trace
    # trace temporally consistent: strictly positive, contiguous, no overlap
    for a, b in zip(trace, trace[1:]):
        assert a.end == b.start, f"{res.algorithm}: gap/overlap {a} -> {b}"
    assert all(s.duration > 0 for s in trace)
    executed = {pid: 0 for pid in by_id}
    completions = {pid: 0 for pid in by_id}
    for s in res.run_segments:
        assert s.start >= by_id[s.pid].arrival, f"{res.algorithm}: P{s.pid} ran before arrival"
        executed[s.pid] += s.duration
        completions[s.pid] += int(s.completes)
    for pid, p in by_id.items():
        assert executed[pid] == p.burst
        assert completions[pid] == 1
    for r in process_metrics(res):
        assert r["completion"] >= r["arrival"] + r["burst"]
        assert r["waiting"] >= 0 and r["response"] >= 0 and r["turnaround"] >= r["burst"]
    assert verify_trace(res) == []


def test_invariants_hold_for_all_schedulers_and_workloads():
    workloads = [classic_demo()] + [generate(p) for p in PROFILE_SEEDS] + [generate("Mixed", s) for s in range(50, 56)]
    for procs in workloads:
        for cs in (0, 1, 3):
            for name, res in _all_results(procs, cs).items():
                _check_invariants(procs, res)


def test_verify_trace_detects_corruption():
    res = fcfs([P(1, 0, 4)])
    from scheduler.models import Segment
    res.trace[0] = Segment("run", 0, 3, 1, completes=True)  # executes 3 of 4
    assert any("executed 3 of 4" in p for p in verify_trace(res))


# ---------------------------------------------------------------- workloads, benchmark, insights
def test_workloads_are_reproducible():
    for name in PROFILE_SEEDS:
        assert generate(name) == generate(name)
        assert generate(name, 7) == generate(name, 7)
    assert generate("Mixed", 1) != generate("Mixed", 2)
    assert len(generate("Interactive")) == 12 and len(generate("High load")) == 20
    with pytest.raises(ValidationError):
        generate("nonsense")


def test_ranks_and_composite_scores():
    assert ranks({"a": 1.0, "b": 2.0, "c": 1.0}, "min") == {"a": 1, "b": 3, "c": 1}
    assert ranks({"a": 1.0, "b": 2.0}, "max") == {"a": 2, "b": 1}
    res, _ = run_all(classic_demo(), SimConfig())
    from experiments.benchmark import summary_table
    df = summary_table(res, EnergyModel())
    for weights in WEIGHT_PRESETS.values():
        sc = composite_scores(df, weights)
        assert ((sc >= -1e-9) & (sc <= 1 + 1e-9)).all()


def test_interpretation_is_derived_from_results():
    res, _ = run_all(classic_demo(), SimConfig())
    from experiments.benchmark import summary_table
    df = summary_table(res, EnergyModel())
    text = " ".join(interpret(df))
    assert "EMI was best or tied-best on" in text and "Round Robin" in text
    row = df.set_index("algorithm")
    assert f"{row.loc['EMI', 'avg_response']:.3g}" in text


def test_monte_carlo_and_win_rates_shape():
    mc = monte_carlo("Mixed", 5, SimConfig())
    assert mc["seed"].nunique() == 5 and set(mc["algorithm"]) == set(ALGORITHMS)
    wr = win_rates(mc)
    assert len(wr) == 9 and wr["EMI beats RR (% seeds)"].between(0, 100).all()
