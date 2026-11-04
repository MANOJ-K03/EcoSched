"""Review 1 demo: the same workload through four schedulers (no UI needed).
Run:  python demo_review1.py
"""
from experiments.workloads import classic_demo
from scheduler.algorithms import fcfs, priority_preemptive, round_robin, sjf
from scheduler.energy import EnergyModel
from scheduler.metrics import summarize, verify_trace

CS_COST, QUANTUM = 1, 4           # context-switch overhead and Round Robin quantum (ticks)
procs = classic_demo()

print("WORKLOAD (PID, arrival, burst, priority)")
for p in procs:
    print(f"  P{p.pid}: arrival={p.arrival:<2} burst={p.burst:<2} priority={p.priority}")

runs = [fcfs(procs, CS_COST), sjf(procs, CS_COST), round_robin(procs, QUANTUM, CS_COST),
        priority_preemptive(procs, CS_COST)]

rows = []
for r in runs:
    print(f"\n{r.algorithm.upper()}  (S = context switch, idle = CPU idle)")
    print("  " + " ".join(f"[{'P'+str(s.pid) if s.is_run else ('S' if s.is_switch else 'idle')} {s.start}-{s.end}]"
                          for s in r.trace))
    print("  trace check:", "OK" if not verify_trace(r) else verify_trace(r))
    rows.append(summarize(r, EnergyModel()))

print(f"\n{'Algorithm':<13}{'AvgWait':>9}{'AvgTurn':>9}{'AvgResp':>9}{'Switches':>10}{'CPU util':>10}")
for s in rows:
    print(f"{s['algorithm']:<13}{s['avg_wait']:>9.2f}{s['avg_turnaround']:>9.2f}{s['avg_response']:>9.2f}"
          f"{s['context_switches']:>10}{s['utilization']:>10.1%}")
