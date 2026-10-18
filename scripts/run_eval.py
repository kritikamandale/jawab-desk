"""Run the evaluation: every model x every eval case, then write the reports.

Usage:
    python scripts/run_eval.py

Models evaluated (override with JAWAB_MODELS, comma separated):
    Qwen/Qwen2.5-0.5B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct   (default single-reply model, JAWAB_MODEL)
    Qwen/Qwen2.5-3B-Instruct     (only if replies are fast enough on this machine)
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jawab.guardrail import is_supported  # noqa: E402
from jawab.metrics import summarize, worst_failures  # noqa: E402
from jawab.model import HFModel, unload_model  # noqa: E402
from jawab.reply import generate_reply  # noqa: E402

OUT_JSON = ROOT / "outputs" / "eval_report.json"
OUT_MD = ROOT / "outputs" / "eval_report.md"

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    os.environ.get("JAWAB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
]
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
FAST_ENOUGH_SECONDS = 20.0

MODEL_LABELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
}


def label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def machine_specs() -> dict:
    import torch

    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1) if hasattr(os, "sysconf") else None
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_model": _cpu_model(),
        "cores_logical": os.cpu_count(),
        "ram_gb": ram_gb,
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def _cpu_model() -> str:
    if platform.system() != "Windows":
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return platform.processor() or "unknown"
        return "unknown"
    try:
        import ctypes

        class SYSTEM_INFO(ctypes.Structure):
            _fields_ = [
                ("wProcessorArchitecture", ctypes.c_ushort),
                ("wReserved", ctypes.c_ushort),
                ("dwPageSize", ctypes.c_ulong),
                ("lpMinimumApplicationAddress", ctypes.c_void_p),
                ("lpMaximumApplicationAddress", ctypes.c_void_p),
                ("dwActiveProcessorMask", ctypes.c_ulong),
                ("dwNumberOfProcessors", ctypes.c_ulong),
                ("dwProcessorType", ctypes.c_ulong),
                ("dwAllocationGranularity", ctypes.c_ulong),
                ("wProcessorLevel", ctypes.c_ushort),
                ("wProcessorRevision", ctypes.c_ushort),
        ]

        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        winreg.CloseKey(key)
        return str(name).strip()
    except Exception:
        return platform.processor() or "unknown"


def load_data():
    with open(ROOT / "data" / "item_sheets.json", encoding="utf-8") as fh:
        sheets = {item["id"]: item for item in json.load(fh)["items"]}
    with open(ROOT / "data" / "eval_cases.json", encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    return sheets, cases


def run_model(model_name: str, sheets: dict, cases: list) -> list:
    model = HFModel(model_name)
    results = []
    for i, case in enumerate(cases, 1):
        outcome = generate_reply(sheets[case["item_id"]], case["message"], model)
        raw_ok, _ = is_supported(outcome["raw_reply"], sheets[case["item_id"]], case["message"])
        results.append(
            {
                "id": case["id"],
                "item_id": case["item_id"],
                "message": case["message"],
                "language": case["language"],
                "expected": case["expected"],
                "must_contain": case["must_contain"],
                "reply": outcome["reply"],
                "raw_reply": outcome["raw_reply"],
                "deferred": outcome["deferred"],
                "blocked": outcome["blocked"],
                "reasons": outcome["reasons"],
                "raw_supported": raw_ok,
                "seconds": outcome["seconds"],
            }
        )
        print(f"  [{label(model_name)}] {i}/{len(cases)} {case['id']} "
              f"({outcome['seconds']:.1f}s, {'deferred' if outcome['deferred'] else 'blocked' if outcome['blocked'] else 'answered'})",
              flush=True)
    return results


def main() -> int:
    if os.environ.get("JAWAB_MOCK") == "1":
        print("JAWAB_MOCK=1 is set; evaluation always uses the real models. Continuing.")
    sheets, cases = load_data()
    specs = machine_specs()

    models = os.environ.get("JAWAB_MODELS")
    models = [m.strip() for m in models.split(",") if m.strip()] if models else list(DEFAULT_MODELS)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": specs,
        "data": {
            "item_sheets": "3 synthetic sheets (written by the coding agent)",
            "eval_cases": "24 synthetic buyer messages (written by the coding agent), 12 answerable / 12 unanswerable",
        },
        "models": {},
        "skipped_models": [],
        "notes": [
            "Prompts and guardrail were fixed before this single evaluation run; no tuning after seeing results.",
            "raw_unsupported_rate is measured on the RAW model reply, before the guardrail. "
            "The unsupported rate AFTER the guardrail is zero by construction and must not be "
            "presented as a model quality result. The honest model numbers are raw_unsupported_rate and blocked_count.",
        ],
    }

    for model_name in models:
        if model_name == MODEL_3B and not os.environ.get("JAWAB_MODELS"):
            if os.environ.get("JAWAB_SKIP_3B") == "1":
                report["skipped_models"].append(
                    {"model": MODEL_3B, "reason": "skipped via JAWAB_SKIP_3B=1"}
                )
                continue
            print(f"\n=== {model_name} (single-reply probe) ===", flush=True)
            probe_model = HFModel(model_name)
            probe_out = generate_reply(sheets[cases[0]["item_id"]], cases[0]["message"], probe_model)
            probe_secs = probe_out["seconds"]
            unload_model()
            if probe_secs > FAST_ENOUGH_SECONDS:
                report["skipped_models"].append(
                    {
                        "model": MODEL_3B,
                        "reason": f"skipped: one reply took {probe_secs:.1f}s, over the ~20s per-reply budget on this machine",
                    }
                )
                print(f"  probe reply took {probe_secs:.1f}s -> skipping 3B", flush=True)
                continue
            print(f"  probe reply took {probe_secs:.1f}s -> running full evaluation", flush=True)

        print(f"\n=== {model_name} ===", flush=True)
        started = time.perf_counter()
        results = run_model(model_name, sheets, cases)
        wall = time.perf_counter() - started
        metrics = summarize(results)
        metrics["wall_seconds"] = round(wall, 1)
        report["models"][model_name] = {
            "label": label(model_name),
            "metrics": metrics,
            "worst_failures": worst_failures(results, 5),
            "results": results,
        }
        print(f"  done in {wall:.1f}s: {json.dumps(metrics)}", flush=True)
        unload_model()
        write_reports(report)  # checkpoint after every model

        # 3B only if replies are fast enough on this machine (probed before its run).
        default_done = all(m in report["models"] for m in DEFAULT_MODELS)
        if default_done and MODEL_3B not in models and not report["skipped_models"]:
            slowest = max(
                report["models"],
                key=lambda m: report["models"][m]["metrics"]["latency_seconds"]["mean"],
            )
            gate = report["models"][slowest]["metrics"]["latency_seconds"]["mean"]
            if gate < FAST_ENOUGH_SECONDS:
                models.append(MODEL_3B)
            else:
                report["skipped_models"].append(
                    {
                        "model": MODEL_3B,
                        "reason": f"skipped: {label(slowest)} averaged {gate:.1f}s per reply, "
                                  f"so 3B would exceed the ~20s per-reply budget on this machine",
                    }
                )

    write_reports(report)
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


def write_reports(report: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    specs = report["machine"]
    lines = []
    lines.append("# Jawab Desk evaluation report")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("## Machine")
    lines.append("")
    lines.append(f"- CPU: {specs.get('cpu_model') or specs.get('cpu')} ({specs.get('cores_logical')} logical cores)")
    lines.append(f"- RAM: {specs.get('ram_gb')} GB")
    lines.append(f"- CUDA available: {specs.get('cuda_available')} (all evaluation ran on CPU)"
                 if not specs.get("cuda_available") else f"- CUDA available: {specs.get('cuda_available')}")
    lines.append(f"- Python {specs.get('python')}, torch {specs.get('torch')}")
    lines.append(f"- Platform: {specs.get('platform')}")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Item sheets: {report['data']['item_sheets']}")
    lines.append(f"- Eval cases: {report['data']['eval_cases']}")
    lines.append("- **These are synthetic messages written by the coding agent, not real buyer chats.**")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    header = (
        "| model | answerable_correct | unanswerable_deferred_by_model | unanswerable_safe "
        "| raw_unsupported_rate | blocked_count | latency mean (s) | latency median (s) |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")
    for model_name, data in report["models"].items():
        m = data["metrics"]
        lat = m["latency_seconds"]
        lines.append(
            f"| {data['label']} | {m['answerable_correct']:.0%} | {m['unanswerable_deferred_by_model']:.0%} "
            f"| {m['unanswerable_safe']:.0%} | {m['raw_unsupported_rate']:.0%} | {m['blocked_count']} "
            f"| {lat['mean']:.2f} | {lat['median']:.2f} |"
        )
    lines.append("")
    lines.append(f"Cases per model: {next(iter(report['models'].values()))['metrics']['n_cases']} "
                 f"({next(iter(report['models'].values()))['metrics']['n_answerable']} answerable, "
                 f"{next(iter(report['models'].values()))['metrics']['n_unanswerable']} unanswerable).")
    lines.append("")
    for note in report["notes"]:
        lines.append(f"> {note}")
        lines.append("")
    if report["skipped_models"]:
        lines.append("## Skipped models")
        lines.append("")
        for skip in report["skipped_models"]:
            lines.append(f"- {skip['model']}: {skip['reason']}")
        lines.append("")
    lines.append("## 5 worst failures (per model)")
    lines.append("")
    for model_name, data in report["models"].items():
        lines.append(f"### {data['label']}")
        lines.append("")
        failures = data["worst_failures"]
        if not failures:
            lines.append("No failures on this evaluation set.")
            lines.append("")
            continue
        for i, f in enumerate(failures, 1):
            lines.append(f"{i}. **{f['id']}** ({f['expected']}) - {f['message']}")
            lines.append(f"   - raw reply: {f['raw_reply']}")
            lines.append(f"   - why it failed: {f['why_it_failed']}")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
