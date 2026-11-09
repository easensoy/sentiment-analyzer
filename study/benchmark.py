"""Training, model-only inference and HTTP serving benchmarks for the frozen candidates, and cost estimates.

    python -m study.benchmark train        # complete Pipeline fits, one fresh process per run (about 25 minutes)
    python -m study.benchmark inference    # artefact loading, warm single-request latency, batch throughput
    python -m study.benchmark http         # uvicorn + FastAPI over loopback, closed-loop concurrency sweep
    python -m study.benchmark costs        # parameterised compute-cost estimates from the measurements

Integrity. Configurations come from protocol/protocol_v1.json and are not modified. Every workload is drawn from
the official training partition (a seeded sample recorded in workload_ids.csv); the official test partition is
never read. Training re-fits are checked against the frozen artefacts on the same training reviews.

Conditions. Numerical-library threads are pinned to one (OMP, OpenBLAS, MKL), so every model computation is
single-threaded; liblinear and libsvm are single-threaded regardless. The host is a shared workstation: on Windows,
every benchmark process (training children, the inference process, the HTTP client and the uvicorn server) is
restricted to the performance cores at above-normal priority, and other processes are left unchanged. The CPU time
other processes used during each measured interval is recorded with the measurement (other_busy_cpus). Wall-clock
time uses time.perf_counter; CPU time uses process counters.

Measurement boundaries.
  train      cleaning + TfidfVectorizer.fit_transform + estimator.fit on in-memory texts (= Pipeline.fit); data
             loading excluded. Peak memory is the process's peak working set (Windows) or VmHWM (Linux), with
             the baseline after data loading reported separately.
  inference  in-process, warm artefact: cleaning + TF-IDF transform + one decision_function call; no HTTP, no JSON.
  http       client -> loopback TCP -> uvicorn (one worker, default thread pool) -> FastAPI validation -> the same
             model path -> JSON response -> client. The client runs on the same host.

Outputs in results/benchmarks/: training_runs.csv, inference.csv (every batch pass in inference_passes.csv),
http.csv, costs.csv, workload_ids.csv, environment.json.
"""
import argparse
import asyncio
import ctypes
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmarks"
PROTOCOL = ROOT / "protocol" / "protocol_v1.json"
DERIVED = ROOT / "data" / "derived" / "imdb_official.csv.gz"
MEMBERSHIP = ROOT / "results" / "data" / "split_membership.csv.gz"
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
WORKLOAD_SIZE, WORKLOAD_SEED = 2_000, 7
TRAIN_REPEATS, SLOW_FIT_REPEATS = 3, 1  # the full-partition SVC fit takes about 17 minutes
LATENCY_RUNS, LATENCY_REQUESTS, LATENCY_WARMUP = 3, 1_000, 200
BATCH_SIZES, BATCH_REPEATS = (64, 1_024), 5
LOAD_REPEATS = 5
HTTP_CANDIDATES = ("svc_uni_8k_historical", "linsvc_uni_full", "linsvc_bi_v2_full")
CONCURRENCY = (1, 2, 4, 8, 16, 32)
HTTP_WARMUP_REQUESTS, LEVEL_WARMUP_S, LEVEL_MEASURE_S, HTTP_SWEEPS, HTTP_TIMEOUT_S = 200, 2.0, 15.0, 2, 30.0
# Hypothetical prices for the cost model; they are parameters, not quotes from any provider.
RATES_USD_PER_VCPU_HOUR = (0.02, 0.05, 0.10)
UTILISATION = (0.25, 0.50, 1.00)


# ----------------------------------------------------------------------------- process measurements

def _windows_memory() -> tuple[int, int]:
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.WorkingSetSize, counters.PeakWorkingSetSize


def memory_bytes() -> tuple[int, int]:
    """(current, peak) resident memory of this process in bytes."""
    if sys.platform == "win32":
        return _windows_memory()
    if sys.platform.startswith("linux"):
        # VmRSS and VmHWM come from one snapshot, in which the kernel reports the high-water mark as at least the
        # current RSS; /proc/self/statm and ru_maxrss are separate counters and can disagree by hundreds of kB.
        fields = {}
        with open("/proc/self/status") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmHWM"):
                    fields[key] = int(value.split()[0]) * 1024  # reported in kB
        return fields["VmRSS"], fields["VmHWM"]
    import resource  # other POSIX systems expose only the peak (ru_maxrss, bytes on macOS)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak, peak


def process_cpu_seconds(pid: int) -> float:
    """User + kernel CPU seconds consumed so far by another process."""
    if sys.platform == "win32":
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            raise OSError(f"cannot open process {pid}")
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                raise OSError("GetProcessTimes failed")
            to_s = lambda ft: ((ft.dwHighDateTime << 32) + ft.dwLowDateTime) / 1e7
            return to_s(times[2]) + to_s(times[3])
        finally:
            kernel32.CloseHandle(handle)
    with open(f"/proc/{pid}/stat") as handle:
        fields = handle.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def background_load(seconds: float = 2.0) -> dict:
    """Busy logical CPUs across the whole host, sampled over an interval (Windows: performance counter)."""
    if sys.platform == "win32":
        query = ("(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval "
                 f"{max(1, int(seconds))} -MaxSamples 1).CounterSamples.CookedValue")
        try:
            value = float(subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True,
                                         text=True, timeout=60).stdout.strip())
            return {"host_cpu_percent": round(value, 1), "busy_logical_cpus": round(value / 100 * os.cpu_count(), 1)}
        except (OSError, ValueError, subprocess.SubprocessError):
            return {"host_cpu_percent": None}
    load1 = os.getloadavg()[0]
    return {"load_average_1min": load1}


def host_cpu_seconds() -> tuple[float, float]:
    """(busy, total) CPU seconds summed over every logical processor of the host since boot."""
    if sys.platform == "win32":
        from ctypes import wintypes
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise OSError("GetSystemTimes failed")
        to_s = lambda ft: ((ft.dwHighDateTime << 32) + ft.dwLowDateTime) / 1e7
        total = to_s(kernel) + to_s(user)  # kernel time includes idle time
        return total - to_s(idle), total
    with open("/proc/stat") as handle:
        values = [int(v) for v in handle.readline().split()[1:]]
    idle = values[3] + values[4]
    return (sum(values) - idle) / os.sysconf("SC_CLK_TCK"), sum(values) / os.sysconf("SC_CLK_TCK")


def other_busy_cpus(busy0: float, busy1: float, own_cpu_s: float, wall_s: float) -> float:
    """Mean number of logical CPUs kept busy by other processes over an interval."""
    return max(0.0, (busy1 - busy0 - own_cpu_s) / wall_s)


def performance_cpus() -> list[int]:
    """Logical processors of the highest efficiency class: the performance cores of a hybrid CPU, else all."""
    if sys.platform != "win32":
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count()))
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    length = wintypes.ULONG(0)
    kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(length), None, 0)
    buffer = (ctypes.c_ubyte * length.value)()
    if not kernel32.GetSystemCpuSetInformation(buffer, length, ctypes.byref(length), None, 0):
        return list(range(os.cpu_count()))
    raw, entries, offset = bytes(buffer), [], 0
    while offset < length.value:  # SYSTEM_CPU_SET_INFORMATION records of variable size
        size = int.from_bytes(raw[offset:offset + 4], "little")
        entries.append((raw[offset + 14], raw[offset + 18]))  # LogicalProcessorIndex, EfficiencyClass
        offset += size
    top = max(efficiency for _, efficiency in entries)
    return sorted(cpu for cpu, efficiency in entries if efficiency == top)


def place(pid: int | None = None) -> None:
    """Restrict a process to the performance cores and raise it to above-normal priority, so that a loaded host
    neither migrates it to efficiency cores nor delays it behind normal-priority work. Other processes are not
    changed. Windows only; elsewhere the process keeps its affinity and priority."""
    if sys.platform != "win32":
        return
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    handle = kernel32.OpenProcess(0x0200 | 0x1000, False, pid if pid is not None else os.getpid())
    if not handle:
        raise OSError(f"cannot open process {pid}")
    try:
        mask = sum(1 << cpu for cpu in performance_cpus())
        if not (kernel32.SetProcessAffinityMask(handle, mask) and kernel32.SetPriorityClass(handle, 0x8000)):
            raise OSError("cannot set affinity or priority")
    finally:
        kernel32.CloseHandle(handle)


def hardware() -> dict:
    info = {"machine": platform.machine(), "system": f"{platform.system()} {platform.release()}",
            "logical_cpus": os.cpu_count(), "python": platform.python_version()}
    if sys.platform == "win32":
        query = ("$c=Get-CimInstance Win32_Processor|Select-Object -First 1;$o=Get-CimInstance Win32_OperatingSystem;"
                 "\"$($c.Name)|$($c.NumberOfCores)|$($c.MaxClockSpeed)|$([math]::Round($o.TotalVisibleMemorySize/1MB,1))\"")
        try:
            name, cores, mhz, ram = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True,
                                                   text=True, timeout=60).stdout.strip().split("|")
            info.update(cpu=name.strip(), physical_cores=int(cores), base_clock_mhz=int(mhz), ram_gb=float(ram))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    import fastapi
    import scipy
    import sklearn
    import uvicorn
    info.update(numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__,
                fastapi=fastapi.__version__, uvicorn=uvicorn.__version__)
    return info


def record_environment(section: str, extra: dict) -> None:
    path = OUT / "environment.json"
    env = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    env.setdefault("hardware", hardware())
    env["thread_settings"] = THREAD_ENV
    env["placement"] = ({"affinity_logical_cpus": performance_cpus(), "priority_class": "above normal",
                         "note": "benchmark processes only; other processes on the host were not changed"}
                        if sys.platform == "win32" else {"note": "default affinity and priority"})
    env[section] = {"recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}
    path.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- workloads and candidates

def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def artefact_dir(cid: str, spec: dict) -> Path:
    base = "models/artefacts" if cid == "svc_uni_8k_historical" else "models/study/artefacts"
    return ROOT / base / spec["model_version"]


def workload() -> pd.DataFrame:
    """A seeded sample of official TRAINING-partition reviews; its ids are written once and then reused."""
    ids_path = OUT / "workload_ids.csv"
    texts = pd.read_csv(DERIVED, usecols=["example_id", "partition", "text"])
    train = texts[texts["partition"] == "train"]
    if ids_path.exists():
        ids = pd.read_csv(ids_path)["example_id"]
    else:
        ids = train["example_id"].sample(WORKLOAD_SIZE, random_state=WORKLOAD_SEED).sort_values()
        OUT.mkdir(parents=True, exist_ok=True)
        ids.to_frame().to_csv(ids_path, index=False, lineterminator="\n")
    frame = ids.to_frame().merge(train, on="example_id", validate="one_to_one")
    if (frame["partition"] != "train").any() or len(frame) != WORKLOAD_SIZE:
        raise SystemExit("the benchmark workload must be 2,000 training-partition reviews")
    return frame


def summary(values) -> dict:
    values = list(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


# ----------------------------------------------------------------------------- training

def train_one(cid: str, repeat: int, data_path: Path, out_path: Path) -> None:
    """Child process: fit one frozen configuration with step timings, peak memory and a consistency check.

    The training texts arrive as a compact pickle written by the parent, so the process's peak working set is
    reached while fitting rather than while parsing the full dataset (a process's peak cannot be reset)."""
    import gc
    import pickle

    from src.pipeline import build_pipeline, load_artefact

    spec = protocol()["candidates"][cid]
    with open(data_path, "rb") as handle:
        texts, labels = pickle.load(handle)
    y = np.asarray(labels)
    gc.collect()
    baseline, _ = memory_bytes()
    pipeline = build_pipeline(spec["config"])
    cleaner, tfidf, clf = (pipeline.named_steps[k] for k in ("clean", "tfidf", "clf"))
    host0, _ = host_cpu_seconds()
    cpu0, t0 = time.process_time(), time.perf_counter()
    cleaned = cleaner.fit(texts).transform(texts)
    t1 = time.perf_counter()
    x = tfidf.fit_transform(cleaned)
    t2 = time.perf_counter()
    clf.fit(x, y)
    t3, cpu1 = time.perf_counter(), time.process_time()
    host1, _ = host_cpu_seconds()
    _, peak = memory_bytes()
    frozen, _ = load_artefact(artefact_dir(cid, spec))
    probe = texts[:500]  # training reviews: the check never reads the test partition
    gap = float(np.max(np.abs(pipeline.decision_function(probe) - frozen.decision_function(probe))))
    result = {"candidate": cid, "repeat": repeat, "n_train": len(texts), "n_features": x.shape[1],
              "cleaning_s": t1 - t0, "vectoriser_fit_transform_s": t2 - t1, "estimator_fit_s": t3 - t2,
              "pipeline_fit_s": t3 - t0, "pipeline_fit_cpu_s": cpu1 - cpu0,
              "baseline_memory_mb": baseline / 2**20, "peak_memory_mb": peak / 2**20,
              "fit_memory_increment_mb": (peak - baseline) / 2**20,
              "max_abs_score_difference_vs_frozen": gap,
              "other_busy_cpus": other_busy_cpus(host0, host1, cpu1 - cpu0, t3 - t0)}
    out_path.write_text(json.dumps(result), encoding="utf-8")


def run_training(only: list[str] | None) -> None:
    import pickle

    from study.fit import training_frame

    proto = protocol()
    cids = [c for c in proto["candidates"] if proto["candidates"][c]["config"] is not None]
    cids = [c for c in cids if not only or c in only]
    before = background_load()
    rows = []
    path = OUT / "training_runs.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["candidate", "repeat"])
    env = {**os.environ, **THREAD_ENV}
    with tempfile.TemporaryDirectory() as data_dir:
        data_paths = {}
        for name in sorted({proto["candidates"][c]["training_set"] for c in cids}):
            frame = training_frame(name)  # example-id order, as study.fit fitted the frozen artefacts
            data_paths[name] = Path(data_dir) / f"{name}.pkl"
            with open(data_paths[name], "wb") as handle:
                pickle.dump((frame["text"].tolist(), frame["label"].tolist()), handle)
        failures = []
        schedule = [(repeat, cid) for repeat in range(TRAIN_REPEATS) for cid in cids
                    if repeat < (SLOW_FIT_REPEATS if cid == "svc_uni_full" else TRAIN_REPEATS)]
        for repeat, cid in schedule:  # repeats interleaved across configurations, so drifting load affects all alike
            if ((done["candidate"] == cid) & (done["repeat"] == repeat)).any():
                continue
            out_path = Path(data_dir) / f"{cid}-{repeat}.json"
            command = [sys.executable, "-m", "study.benchmark", "_train_one", cid, str(repeat),
                       str(data_paths[proto["candidates"][cid]["training_set"]]), str(out_path)]
            for attempt in (1, 2):  # a crashed process is retried once; every crash is recorded
                code = subprocess.run(command, cwd=ROOT, env=env).returncode
                if code == 0:
                    break
                failures.append({"candidate": cid, "repeat": repeat, "attempt": attempt, "exit_code": code})
                print(f"{cid} #{repeat}: process exited with code {code}")
            else:
                raise SystemExit(f"{cid} #{repeat}: the fit failed twice")
            row = json.loads(out_path.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"{cid} #{repeat}: fit {row['pipeline_fit_s']:.2f} s, increment {row['fit_memory_increment_mb']:.0f} MB, "
                  f"score gap {row['max_abs_score_difference_vs_frozen']:.2e}, other load {row['other_busy_cpus']:.1f} CPUs")
            frame = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            frame = frame.astype({"repeat": int, "n_train": int, "n_features": int})  # exact, not "%.6g"
            frame.to_csv(path, index=False, lineterminator="\n", float_format="%.6g")
    record_environment("train", {"background_before": before, "background_after": background_load(),
                                 "repeats": TRAIN_REPEATS, "repeats_svc_uni_full": SLOW_FIT_REPEATS,
                                 "process_per_run": True, "order": "repeats interleaved across configurations",
                                 "process_failures": failures})


# ----------------------------------------------------------------------------- model-only inference

def run_inference() -> None:
    """Every configuration is loaded once and then measured in rounds: each round runs one latency run (or one batch
    pass) for every configuration, in alternating order, so that drifting host load reaches all of them alike."""
    from src.pipeline import load_artefact

    proto = protocol()
    reviews = workload()["text"].tolist()
    before = background_load()
    cids = [cid for cid, spec in proto["candidates"].items() if spec["config"] is not None]
    rounds = lambda k: cids if k % 2 == 0 else cids[::-1]
    loads = {cid: [] for cid in cids}
    pipelines = {}
    for k in range(LOAD_REPEATS):  # SHA-256 check + joblib load + validation, warm file cache
        for cid in rounds(k):
            t0 = time.perf_counter()
            pipelines[cid], _ = load_artefact(artefact_dir(cid, proto["candidates"][cid]))
            loads[cid].append(time.perf_counter() - t0)
    parts = {cid: (pipelines[cid][:-1], pipelines[cid][-1]) for cid in cids}
    for cid in cids:
        features, clf = parts[cid]
        for text in reviews[:LATENCY_WARMUP]:
            clf.decision_function(features.transform([text]))
    host = {cid: [0.0, 0.0, 0.0] for cid in cids}  # host busy CPU s, own CPU s, wall s while measuring each one

    def measured(cid, fn):
        h0, c0, w0 = host_cpu_seconds()[0], time.process_time(), time.perf_counter()
        result = fn()
        h1, c1, w1 = host_cpu_seconds()[0], time.process_time(), time.perf_counter()
        for i, v in enumerate((h1 - h0, c1 - c0, w1 - w0)):
            host[cid][i] += v
        return result

    def latency_run(cid, run):
        features, clf = parts[cid]
        lat = []
        for text in reviews[run * LATENCY_REQUESTS % len(reviews):][:LATENCY_REQUESTS]:
            t0 = time.perf_counter()
            clf.decision_function(features.transform([text]))
            lat.append((time.perf_counter() - t0) * 1000)
        return np.quantile(lat, [0.5, 0.95, 0.99])

    def batch_pass(cid, size):
        features, clf = parts[cid]
        c0, t0 = time.process_time(), time.perf_counter()
        for start in range(0, len(reviews), size):
            clf.decision_function(features.transform(reviews[start:start + size]))
        return time.perf_counter() - t0, time.process_time() - c0

    quantiles = {cid: [] for cid in cids}
    for run in range(LATENCY_RUNS):
        for cid in rounds(run):
            quantiles[cid].append(measured(cid, lambda: latency_run(cid, run)))
    passes = {(cid, size): [] for cid in cids for size in BATCH_SIZES}
    for k in range(BATCH_REPEATS):
        for cid in rounds(k):
            for size in BATCH_SIZES:
                passes[(cid, size)].append(measured(cid, lambda: batch_pass(cid, size)))
    rows, pass_rows = [], []
    for cid in cids:
        directory, pipeline, clf = artefact_dir(cid, proto["candidates"][cid]), pipelines[cid], parts[cid][1]
        p50s, p95s, p99s = zip(*quantiles[cid])
        row = {"candidate": cid, "artefact_mb": (directory / "pipeline.joblib").stat().st_size / 2**20,
               "n_features": len(pipeline.named_steps["tfidf"].vocabulary_),
               # a kernel SVC scores against every support vector; the linear estimators use one weight vector
               "n_support_vectors": int(clf.n_support_.sum()) if hasattr(clf, "n_support_") else 0,
               "load_s_median": statistics.median(loads[cid]), "load_s_min": min(loads[cid]),
               "load_s_max": max(loads[cid]),
               "latency_ms_p50": statistics.median(p50s), "latency_ms_p50_min": min(p50s), "latency_ms_p50_max": max(p50s),
               "latency_ms_p95": statistics.median(p95s), "latency_ms_p95_min": min(p95s), "latency_ms_p95_max": max(p95s),
               "latency_ms_p99": statistics.median(p99s)}
        for size in BATCH_SIZES:
            walls, cpus = zip(*passes[(cid, size)])
            rates = [len(reviews) / w for w in walls]
            per_1000 = [1000 * c / len(reviews) for c in cpus]  # process clock ticks are 15.6 ms on Windows
            row.update({f"throughput_b{size}_per_s": statistics.median(rates),
                        f"throughput_b{size}_min": min(rates), f"throughput_b{size}_max": max(rates),
                        f"cpu_s_per_1000_b{size}": statistics.median(per_1000),
                        f"cpu_s_per_1000_b{size}_min": min(per_1000), f"cpu_s_per_1000_b{size}_max": max(per_1000)})
            pass_rows += [{"candidate": cid, "round": k, "batch_size": size, "reviews": len(reviews), "wall_s": w,
                           "cpu_s": c} for k, (w, c) in enumerate(passes[(cid, size)])]
        row["cpu_s_per_1000_batch"] = row[f"cpu_s_per_1000_b{BATCH_SIZES[-1]}"]
        row["wall_s_per_1000_batch"] = 1000 / row[f"throughput_b{BATCH_SIZES[-1]}_per_s"]
        busy, own, wall = host[cid]
        row["other_busy_cpus"] = max(0.0, (busy - own) / wall)
        rows.append(row)
        print(f"{cid}: load {row['load_s_median']:.2f} s, p50 {row['latency_ms_p50']:.2f} ms, "
              f"batch {row[f'throughput_b{BATCH_SIZES[-1]}_per_s']:.0f}/s, CPU {row['cpu_s_per_1000_batch']:.3f} s/1,000, "
              f"other load {row['other_busy_cpus']:.1f} CPUs")
    pd.DataFrame(rows).to_csv(OUT / "inference.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.DataFrame(pass_rows).to_csv(OUT / "inference_passes.csv", index=False, lineterminator="\n",
                                   float_format="%.6g")
    lengths = pd.Series([len(t) for t in reviews])
    record_environment("inference", {"background_before": before, "background_after": background_load(),
                                     "workload": f"{len(reviews)} official training-partition reviews (seed {WORKLOAD_SEED})",
                                     "workload_chars": {"median": int(lengths.median()), "p95": int(lengths.quantile(0.95))},
                                     "order": "all configurations loaded once; loads, latency runs and batch passes in "
                                              "rounds over every configuration, alternating direction",
                                     "load_repeats": LOAD_REPEATS, "latency": f"{LATENCY_WARMUP} warm-up requests, then "
                                     f"{LATENCY_RUNS} runs of {LATENCY_REQUESTS} sequential requests (batch size 1)",
                                     "batch": f"batch sizes {list(BATCH_SIZES)}, {BATCH_REPEATS} passes over the workload each"})


# ----------------------------------------------------------------------------- HTTP serving

def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def listener_pid(port: int, fallback: int) -> int:
    """PID of the process listening on the port. On Windows a venv's python.exe is a launcher that runs the real
    interpreter as a child, so the Popen PID would measure the idle launcher instead of the server."""
    if sys.platform != "win32":
        return fallback
    query = f"(Get-NetTCPConnection -LocalPort {port} -State Listen | Select-Object -First 1).OwningProcess"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True,
                         timeout=60).stdout.strip()
    return int(out) if out.isdigit() else fallback


def stop(proc: subprocess.Popen) -> None:
    """Stop the server and any child it started (the venv launcher's interpreter on Windows)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    proc.wait(timeout=30)


async def closed_loop(base_url: str, payloads: list[str], concurrency: int, warmup_s: float, measure_s: float,
                      timeout_s: float, transport=None, on_window=None) -> list[tuple[float, float, bool, str]]:
    """Closed-loop load: `concurrency` workers each send the next request when the previous one returns.
    Returns (start, end, ok, status) for requests that started inside the measurement window. If given,
    on_window("start") and on_window("end") are called at the window's boundaries (for CPU sampling)."""
    import httpx

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    records: list[tuple[float, float, bool, str]] = []
    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=timeout_s, transport=transport) as client:
        begin = time.perf_counter()
        measure_from, measure_to = begin + warmup_s, begin + warmup_s + measure_s

        async def worker(k: int) -> None:
            i = k
            while time.perf_counter() < measure_to:
                t0 = time.perf_counter()
                try:
                    response = await client.post("/predict", json={"text": payloads[i % len(payloads)]})
                    ok, status = response.status_code == 200, str(response.status_code)
                except Exception as exc:  # timeouts and connection errors count as failed requests
                    ok, status = False, type(exc).__name__
                t1 = time.perf_counter()
                if t0 >= measure_from:
                    records.append((t0 - measure_from, t1 - measure_from, ok, status))
                i += concurrency

        async def window() -> None:
            await asyncio.sleep(max(0.0, measure_from - time.perf_counter()))
            on_window("start")
            await asyncio.sleep(max(0.0, measure_to - time.perf_counter()))
            on_window("end")

        await asyncio.gather(*(worker(k) for k in range(concurrency)), *([window()] if on_window else []))
    return records


def summarise_level(records, measure_s: float) -> dict:
    done = [r for r in records if r[1] <= measure_s]
    lat = np.array([(end - start) * 1000 for start, end, ok, _ in done if ok])
    failures = sum(1 for r in done if not r[2])
    q = np.quantile(lat, [0.5, 0.95, 0.99]) if lat.size else [np.nan] * 3
    return {"completed": len(done), "failed": failures, "failure_rate": failures / len(done) if done else np.nan,
            "throughput_rps": sum(1 for r in done if r[2]) / measure_s,
            "latency_ms_p50": q[0], "latency_ms_p95": q[1], "latency_ms_p99": q[2]}


def serve(cid: str, spec: dict, models_dir: Path, port: int) -> tuple[subprocess.Popen, float]:
    """Start uvicorn on the frozen artefact; return the process and its time to first successful /model."""
    import httpx

    target = models_dir / "artefacts" / spec["model_version"]
    shutil.copytree(artefact_dir(cid, spec), target)
    (models_dir / "serving.json").write_text(json.dumps(
        {"schema_version": 1, "artefact": f"artefacts/{spec['model_version']}", "model_version": spec["model_version"]}),
        encoding="utf-8")
    env = {**os.environ, **THREAD_ENV, "SENTIMENT_MODELS_DIR": str(models_dir)}
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", str(port),
                             "--workers", "1", "--log-level", "warning", "--no-access-log"], cwd=ROOT, env=env)
    while time.perf_counter() - t0 < 120:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/model", timeout=2).status_code == 200:
                return proc, time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            raise SystemExit(f"{cid}: server exited with code {proc.returncode}")
        time.sleep(0.2)
    stop(proc)
    raise SystemExit(f"{cid}: server not ready within 120 s")


def run_http() -> None:
    import httpx

    proto = protocol()
    reviews = workload()["text"].tolist()
    before = background_load()
    rows = []
    for sweep in range(HTTP_SWEEPS):
        for cid in HTTP_CANDIDATES:
            spec = proto["candidates"][cid]
            port = free_port()
            with tempfile.TemporaryDirectory() as tmp:
                proc, ready_s = serve(cid, spec, Path(tmp) / "models", port)
                server_pid = listener_pid(port, proc.pid)
                place(server_pid)  # start-up ran with default placement; the measured load runs placed
                try:
                    base = f"http://127.0.0.1:{port}"
                    with httpx.Client(base_url=base, timeout=HTTP_TIMEOUT_S) as client:
                        for text in reviews[:HTTP_WARMUP_REQUESTS]:
                            client.post("/predict", json={"text": text}).raise_for_status()
                    for level in CONCURRENCY:
                        marks = {}

                        def mark(event: str, marks=marks) -> None:
                            marks[event] = (process_cpu_seconds(server_pid), time.process_time(), time.perf_counter(),
                                            host_cpu_seconds()[0])

                        records = asyncio.run(closed_loop(base, reviews, level, LEVEL_WARMUP_S, LEVEL_MEASURE_S,
                                                          HTTP_TIMEOUT_S, on_window=mark))
                        server_cpu, client_cpu, window_s, host_busy = (end - start for start, end
                                                                       in zip(marks["start"], marks["end"]))
                        stats = summarise_level(records, LEVEL_MEASURE_S)
                        served = stats["throughput_rps"] * LEVEL_MEASURE_S
                        rows.append({"candidate": cid, "sweep": sweep, "concurrency": level, "server_ready_s": ready_s,
                                     **stats, "server_cpu_s_per_1000": 1000 * server_cpu / served if served else np.nan,
                                     "server_cpu_utilisation": server_cpu / window_s,
                                     "client_cpu_utilisation": client_cpu / window_s,
                                     "other_busy_cpus": max(0.0, (host_busy - server_cpu - client_cpu) / window_s)})
                        print(f"{cid} sweep {sweep} c={level}: {stats['throughput_rps']:.0f} rps, "
                              f"p50 {stats['latency_ms_p50']:.1f} ms, p95 {stats['latency_ms_p95']:.1f} ms, "
                              f"failed {stats['failed']}, other load {rows[-1]['other_busy_cpus']:.1f} CPUs")
                finally:
                    stop(proc)
            pd.DataFrame(rows).to_csv(OUT / "http.csv", index=False, lineterminator="\n", float_format="%.6g")
    from importlib import metadata

    def version(dist: str) -> str | None:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None

    record_environment("http", {"background_before": before, "background_after": background_load(),
                                "server": "uvicorn, one worker process, FastAPI default thread pool for the sync handler",
                                "http_stack": {d: version(d) for d in ("uvicorn", "h11", "httptools", "uvloop", "fastapi",
                                                                      "starlette", "pydantic", "anyio", "httpx")},
                                "client": "httpx.AsyncClient on the same host, keep-alive connections, closed loop",
                                "concurrency": list(CONCURRENCY), "warmup_requests": HTTP_WARMUP_REQUESTS,
                                "level_warmup_s": LEVEL_WARMUP_S, "level_measure_s": LEVEL_MEASURE_S,
                                "sweeps": HTTP_SWEEPS, "timeout_s": HTTP_TIMEOUT_S,
                                "cpu_note": "server and client CPU are sampled at the boundaries of each level's "
                                            "measurement window; server CPU per 1,000 requests divides the server's "
                                            "CPU in the window by the requests completed in it; a client CPU "
                                            "utilisation near 1.0 would mean the single-threaded client limited "
                                            "throughput"})


# ----------------------------------------------------------------------------- cost estimates

def run_costs() -> None:
    """Compute cost per 1,000 predictions = rate x CPU-seconds per 1,000 / 3,600 / utilisation, for each mode."""
    inference = pd.read_csv(OUT / "inference.csv").set_index("candidate")
    http = pd.read_csv(OUT / "http.csv")
    peak = http.loc[http.groupby(["candidate", "sweep"])["throughput_rps"].idxmax()]
    interactive = peak.groupby("candidate")["server_cpu_s_per_1000"].median()
    rows = []
    for cid in inference.index:
        bases = [("batch (in-process, batch size 1,024, one thread)", inference.loc[cid, "cpu_s_per_1000_batch"])]
        if cid in interactive.index:
            bases.append(("interactive HTTP (server CPU at peak throughput)", interactive[cid]))
        for mode, cpu_s in bases:
            for rate in RATES_USD_PER_VCPU_HOUR:
                for util in UTILISATION:
                    rows.append({"candidate": cid, "mode": mode, "cpu_s_per_1000": cpu_s,
                                 "rate_usd_per_vcpu_hour": rate, "utilisation": util,
                                 "usd_per_1000_predictions": rate * cpu_s / 3600 / util,
                                 "usd_per_million_predictions": 1000 * rate * cpu_s / 3600 / util,
                                 "status": "estimate: hypothetical rate and utilisation; measured CPU time"})
    pd.DataFrame(rows).to_csv(OUT / "costs.csv", index=False, lineterminator="\n", float_format="%.6g")
    print(pd.DataFrame(rows).query("rate_usd_per_vcpu_hour == 0.05 and utilisation == 0.5")
          [["candidate", "mode", "cpu_s_per_1000", "usd_per_million_predictions"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--only", nargs="*")
    sub.add_parser("inference")
    sub.add_parser("http")
    sub.add_parser("costs")
    child = sub.add_parser("_train_one")
    child.add_argument("candidate")
    child.add_argument("repeat", type=int)
    child.add_argument("data", type=Path)
    child.add_argument("out", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command in ("_train_one", "inference", "http"):
        place()
    if args.command == "_train_one":
        train_one(args.candidate, args.repeat, args.data, args.out)
    elif args.command == "train":
        run_training(args.only)
    elif args.command == "inference":
        run_inference()
    elif args.command == "http":
        run_http()
    else:
        run_costs()


if __name__ == "__main__":
    main()
