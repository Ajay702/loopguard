from __future__ import annotations

from loopguard.cli import main


def test_demo_trace_runs_and_crash_resume(tmp_path, capsys):
    runs = str(tmp_path / "runs")
    assert main(["--runs-dir", runs, "demo"]) == 0
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "injection detected" in out and "retry" in out

    assert main(["--runs-dir", runs, "demo", "--crash-after", "6"]) == 1
    out = capsys.readouterr().out
    run_id = out.split("loopguard resume ")[-1].split()[0]
    assert main(["--runs-dir", runs, "resume", run_id]) == 0
    assert "Resumed after crash" in capsys.readouterr().out

    assert main(["--runs-dir", runs, "trace", run_id]) == 0
    assert (tmp_path / "runs" / run_id / "trace.html").exists()
    assert main(["--runs-dir", runs, "runs"]) == 0


def test_eval_subset(tmp_path, capsys):
    assert main(["--runs-dir", str(tmp_path), "eval", "--only", "refund_happy_path", "--markdown"]) == 0
    assert "1/1 scenarios passed" in capsys.readouterr().out
