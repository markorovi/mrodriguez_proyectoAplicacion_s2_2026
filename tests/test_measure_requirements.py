"""Evidence script for the performance requirements, with fake data and no inference."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from scripts import measure_requirements as mr


def _pair(name="sintético: desplazamiento", kind="sintetico", cold=1.0, warm_max=0.4, error=None, depth_ok=True):
    pair = {"name": name, "kind": kind, "size": [640, 480], "expected": "x", "changes": ["caja: DESPLAZAMIENTO"]}
    if error:
        pair["error"] = error
        return pair
    pair["cold"] = {"total": cold, "stages": {"depth": 0.1, "loading": 0.5}, "includes_loading": True}
    pair["warm"] = {"count": 3, "total": {"mean": warm_max - 0.1, "min": warm_max - 0.2, "max": warm_max},
                    "stages": {"depth": {"mean": 0.05, "min": 0.04, "max": 0.06}}}
    pair["depth"] = [{"input_size": [640, 480], "shape": [480, 640] if depth_ok else [240, 320],
                      "finite": True, "min": 1.0, "max": 3.5}]
    return pair


def _child(device="cuda", pairs=None, **extra):
    pairs = pairs if pairs is not None else [_pair(name=f"p{i}") for i in range(3)]
    child = {"device_requested": device, "pairs": pairs, "resolved_devices": {"depth": "CPU" if device == "cpu" else "RTX"}}
    if device == "cuda":
        child["gpu"] = {"smi_available": True, "baseline": {"util_median": 3, "mem_median_mib": 1600, "samples": 5},
                        "max_util": 80, "max_mem_mib": 2500, "mem_increase_mib": 900, "torch_max_allocated_mib": 870,
                        "samples": 40}
    if device == "cpu":
        child.update(cpu_notice="aviso", cpu_notice_in_warnings=True, cuda_available=False)
    child.update(extra)
    return child


def test_stats_mean_min_max():
    assert mr.stats([1.0, 2.0, 6.0]) == {"mean": 3.0, "min": 1.0, "max": 6.0}
    assert mr.stats([]) is None
    per_stage = mr.stage_stats([{"depth": 1.0, "loading": 4.0}, {"depth": 3.0}])
    assert per_stage["depth"] == {"mean": 2.0, "min": 1.0, "max": 3.0}
    assert per_stage["loading"] == {"mean": 4.0, "min": 4.0, "max": 4.0}


def test_rr001_passes_within_budget_and_fails_over_it():
    assert mr.evaluate_rr001(_child()).status == "CUMPLE"
    slow = _child(pairs=[_pair(name="a", cold=16.0), _pair(name="b"), _pair(name="c")])
    assert mr.evaluate_rr001(slow).status == "NO CUMPLE"
    broken = _child(pairs=[_pair(name="a"), _pair(name="b"), _pair(name="c", error="x")])
    assert mr.evaluate_rr001(broken).status == "NO CUMPLE"
    assert mr.evaluate_rr001(None).status == "NO EVALUADO"


def test_rf009_needs_a_memory_increase_over_the_idle_baseline():
    assert mr.evaluate_rf009(_child()).status == "CUMPLE"
    small = _child()
    small["gpu"]["mem_increase_mib"] = 60
    assert mr.evaluate_rf009(small).status == "NO CUMPLE"
    no_torch = _child()
    no_torch["gpu"]["torch_max_allocated_mib"] = 0
    assert mr.evaluate_rf009(no_torch).status == "NO CUMPLE"


def test_rf009_without_nvidia_smi_relies_on_pytorch():
    child = _child()
    child["gpu"] = {"smi_available": False, "baseline": None, "torch_max_allocated_mib": 500}
    verdict = mr.evaluate_rf009(child)
    assert verdict.status == "CUMPLE" and "monitoreo externo no disponible" in verdict.evidence
    child["gpu"]["torch_max_allocated_mib"] = 0
    assert mr.evaluate_rf009(child).status == "NO CUMPLE"


def test_rf010_requires_cpu_completion_and_the_notice():
    assert mr.evaluate_rf010(_child("cpu")).status == "CUMPLE"
    assert mr.evaluate_rf010(_child("cpu", cpu_notice_in_warnings=False)).status == "NO CUMPLE"
    on_gpu = _child("cpu", resolved_devices={"depth": "NVIDIA"})
    assert mr.evaluate_rf010(on_gpu).status == "NO CUMPLE"
    gpu_still_visible = _child("cpu", cuda_available=True)
    verdict = mr.evaluate_rf010(gpu_still_visible)
    assert verdict.status == "NO CUMPLE" and "GPU oculta al proceso: no" in verdict.evidence
    assert mr.evaluate_ra002(_child(), gpu_still_visible).status == "NO CUMPLE"


def test_ra002_requires_both_environments():
    assert mr.evaluate_ra002(_child(), _child("cpu")).status == "CUMPLE"
    failed_cpu = _child("cpu", pairs=[_pair(name="a"), _pair(name="b", error="x"), _pair(name="c")])
    assert mr.evaluate_ra002(_child(), failed_cpu).status == "NO CUMPLE"
    assert mr.evaluate_ra002(_child(), None).status == "NO EVALUADO"


def test_rr002_checks_shape_finiteness_and_range():
    assert mr.evaluate_rr002([_child()]).status == "CUMPLE"
    distorted = _child(pairs=[_pair(name="a", depth_ok=False), _pair(name="b"), _pair(name="c")])
    assert mr.evaluate_rr002([distorted]).status == "NO CUMPLE"
    bad_range = _child()
    bad_range["pairs"][0]["depth"][0]["min"] = 0.0
    assert mr.evaluate_rr002([bad_range]).status == "NO CUMPLE"


def test_markdown_has_the_requirements_environment_and_timing_tables():
    results = {"cuda": _child(), "cpu": _child("cpu"), "environment": {"gpu": "RTX", "os": "Windows"},
               "meta": {"date": "2026-11-20 10:00", "commit": "abc1234", "dirty": False},
               "observations": ["nota"]}
    page = mr.render_markdown(results, mr.evaluate(results))
    for text in ("## Requisitos", "| RR-001 | **CUMPLE** |", "| RF-009 |", "## Entorno", "| GPU | RTX |",
                 "## Tiempos en GPU", "## Tiempos en CPU (sin GPU)", "1600 MiB", "abc1234",
                 mr.REGENERATE_COMMAND, "- nota", "una sola máquina"):
        assert text in page, text


def test_nvidia_smi_output_is_parsed_and_its_absence_handled():
    assert mr.parse_smi_samples("12, 1650\n 3 ,1600\nbasura\n") == [(12.0, 1650.0), (3.0, 1600.0)]
    ok = lambda *a, **k: SimpleNamespace(returncode=0, stdout="7, 1700\n")  # noqa: E731
    assert mr.query_smi(ok) == (7.0, 1700.0)
    missing = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi"))  # noqa: E731
    assert mr.query_smi(missing) is None
    assert mr.baseline([(3, 1600), (5, 1610), (2, 1590), None]) == {"util_median": 3, "mem_median_mib": 1600, "samples": 3}
    assert mr.baseline([None, None]) is None


def test_gpu_monitor_keeps_the_maxima():
    readings = iter([(5.0, 1600.0), (90.0, 2400.0), (40.0, 2000.0)])
    monitor = mr.GpuMonitor(query=lambda: next(readings, None), interval=0.001)
    with monitor:
        while len(monitor.samples) < 3:
            pass
    assert (monitor.max_util, monitor.max_mem_mib) == (90.0, 2400.0)


def test_parent_launches_one_fresh_process_per_device(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, env, cwd, check):
        calls.append((command, env))
        out = command[command.index("--json-out") + 1]
        device = command[command.index("--device") + 1]
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(_child(device), handle)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mr, "metadata", lambda: {"date": "d", "commit": "c", "dirty": False})
    results = mr.run_parent(["cuda", "cpu"], tmp_path, run=fake_run)

    (cuda_cmd, cuda_env), (cpu_cmd, cpu_env) = calls
    assert cuda_cmd[:3] == [sys.executable, "-m", "scripts.measure_requirements"]
    assert cuda_cmd[3:5] == ["--device", "cuda"] and cpu_cmd[3:5] == ["--device", "cpu"]
    # An empty value does not hide the GPU on Windows; an invalid index does.
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == "-1" and "CUDA_VISIBLE_DEVICES" not in cuda_env
    assert cpu_env["HF_HUB_OFFLINE"] == "1" and cuda_env["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"  # The session environment is untouched.
    assert (tmp_path / "rendimiento.md").is_file() and (tmp_path / "rendimiento.json").is_file()
    assert [r["status"] for r in results["requirements"]] == ["CUMPLE"] * 5


def test_errors_are_sanitized_of_absolute_paths():
    message = mr.sanitize(f"No se encontró {mr.PROJECT_ROOT / 'data' / 'x.jpg'}")
    assert str(mr.PROJECT_ROOT) not in message and "x.jpg" in message


def test_real_pairs_keep_only_their_name_and_timings():
    real = {"name": "real_2", "kind": "real", "files": ["real_before_2.jpg", "real_after_2.jpg"],
            "changes": ["caja: DESPLAZAMIENTO (6.6 cm)"], "cold": {"total": 1.0}, "size": [1920, 1080]}
    assert mr.public_pair(real) == {"name": "real_2", "kind": "real", "cold": {"total": 1.0}, "size": [1920, 1080]}
    synthetic = _pair()
    assert mr.public_pair(synthetic) is synthetic  # Synthetic scenes are not personal.
