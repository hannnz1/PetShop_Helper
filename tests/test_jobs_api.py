"""The admin job API can only launch registered Makefile targets."""

from pathlib import Path
import signal
import subprocess

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.jobs import router
from app.core import jobs


class FakeProcess:
    def __init__(self, pid=4321):
        self.pid = pid
        self.returncode = None
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signal):
        self.signals.append(signal)

    def wait(self, timeout=None):
        self.returncode = -1
        return self.returncode


def _client(tmp_path, monkeypatch, *, platform_name="nt"):
    process = FakeProcess()
    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return process

    monkeypatch.setattr(jobs.subprocess, "Popen", popen)
    monkeypatch.setattr(jobs.JobRunner, "_make_executable", lambda self: "make-test")
    app = FastAPI()
    app.state.jobs = jobs.JobRunner(root=tmp_path, platform_name=platform_name)
    app.include_router(router)
    return TestClient(app), process, calls


def test_unknown_job_is_rejected(tmp_path, monkeypatch):
    client, _, calls = _client(tmp_path, monkeypatch)
    with client:
        assert client.post("/api/jobs/echo%20pwned").status_code == 404
        assert client.get("/api/jobs/not-a-job").status_code == 404
        assert client.post("/api/jobs/not-a-job/stop").status_code == 404
    assert calls == []


def test_running_job_refuses_reentry_and_uses_fixed_argv(tmp_path, monkeypatch):
    client, process, calls = _client(tmp_path, monkeypatch)
    with client:
        first = client.post("/api/jobs/kb-preview")
        assert first.status_code == 200
        assert first.json()["state"] == "running"
        assert client.post("/api/jobs/kb-preview").status_code == 409
        assert client.get("/api/jobs/kb-preview").json()["state"] == "running"
        assert client.get("/api/jobs").json()["jobs"]
        process.returncode = 0
        assert client.get("/api/jobs/kb-preview").json()["state"] == "succeeded"
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == ["make-test", "kb-preview"]
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == tmp_path
    assert kwargs["creationflags"] == jobs.subprocess.CREATE_NEW_PROCESS_GROUP


def test_every_job_target_exists_in_makefile():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    for spec in jobs.JOB_SPECS.values():
        assert f"\n{spec.target}:" in "\n" + makefile


def test_mining_job_is_registered_and_requires_confirmation():
    spec = jobs.JOB_SPECS["kb-mine"]
    assert spec.target == "kb-mine"
    assert spec.heavy is True


def test_windows_stop_terminates_only_registered_process_tree(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    stopped = []

    def run(argv, **kwargs):
        stopped.append((argv, kwargs))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(jobs.subprocess, "run", run)
    with client:
        assert client.post("/api/jobs/kb-preview").status_code == 200
        result = client.post("/api/jobs/kb-preview/stop")
    assert result.status_code == 200
    assert result.json()["state"] == "stopped"
    assert stopped == [(["taskkill", "/PID", "4321", "/T", "/F"], {"check": False, "capture_output": True})]


def test_windows_stop_waits_for_exit_before_allowing_reentry(tmp_path, monkeypatch):
    client, process, calls = _client(tmp_path, monkeypatch)
    wait_calls = []

    def wait(timeout=None):
        wait_calls.append(timeout)
        raise subprocess.TimeoutExpired(cmd="make", timeout=timeout)

    process.wait = wait
    monkeypatch.setattr(
        jobs.subprocess, "run",
        lambda argv, **kwargs: type("Result", (), {"returncode": 0})(),
    )
    with client:
        assert client.post("/api/jobs/kb-preview").status_code == 200
        assert client.post("/api/jobs/kb-preview/stop").status_code == 503
        assert client.get("/api/jobs/kb-preview").json()["state"] == "running"
        assert client.post("/api/jobs/kb-preview").status_code == 409
    assert wait_calls == [5]
    assert len(calls) == 1


def test_windows_taskkill_failure_does_not_allow_reentry(tmp_path, monkeypatch):
    client, _, calls = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        jobs.subprocess, "run",
        lambda argv, **kwargs: type("Result", (), {"returncode": 1})(),
    )
    with client:
        assert client.post("/api/jobs/kb-preview").status_code == 200
        assert client.post("/api/jobs/kb-preview/stop").status_code == 503
        assert client.post("/api/jobs/kb-preview").status_code == 409
    assert len(calls) == 1


def test_log_tail_is_bounded_and_unknown_job_never_reads_path(tmp_path):
    runner = jobs.JobRunner(root=tmp_path, max_log_bytes=10)
    log = tmp_path / "log" / "acceptance" / "kb-preview.log"
    log.parent.mkdir(parents=True)
    log.write_text("1234567890last", encoding="utf-8")
    assert runner.tail("kb-preview") == "567890last"
    try:
        runner.tail("../secrets")
    except jobs.UnknownJob:
        pass
    else:
        raise AssertionError("unregistered paths must be rejected")


def test_posix_start_and_stop_use_process_group(tmp_path, monkeypatch):
    client, _, calls = _client(tmp_path, monkeypatch, platform_name="posix")
    signals = []
    monkeypatch.setattr(jobs.os, "killpg", lambda pid, sig: signals.append((pid, sig)), raising=False)
    with client:
        assert client.post("/api/jobs/kb-preview").status_code == 200
        assert client.post("/api/jobs/kb-preview/stop").json()["state"] == "stopped"
    assert calls[0][1]["start_new_session"] is True
    assert "creationflags" not in calls[0][1]
    assert signals == [(4321, signal.SIGTERM)]


def test_heavy_job_requires_explicit_confirmation(tmp_path, monkeypatch):
    client, _, calls = _client(tmp_path, monkeypatch)
    monkeypatch.setitem(jobs.JOB_SPECS, "kb-preview", jobs.JobSpec("kb-preview", "kb-preview", heavy=True))
    with client:
        assert client.post("/api/jobs/kb-preview").status_code == 400
        assert calls == []
        assert client.post("/api/jobs/kb-preview", json={"confirm": True}).status_code == 200
