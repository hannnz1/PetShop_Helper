"""Run only registered Makefile targets for the local knowledge-base admin UI."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
from threading import RLock


ROOT = Path(__file__).resolve().parents[2]
WINDOWS_MAKE = Path(r"C:\msys64\usr\bin\make.exe")


@dataclass(frozen=True)
class JobSpec:
    name: str
    target: str
    heavy: bool = False


# Register a target only after its Makefile recipe and referenced script exist.
JOB_SPECS = {
    spec.name: spec for spec in (
        JobSpec("kb-preview", "kb-preview"),
        JobSpec("kb-build", "kb-build"),
        JobSpec("seed-conv", "seed-conv"),
        JobSpec("kb-mine", "kb-mine", heavy=True),
        JobSpec("eval-mining", "eval-mining"),
    )
}


class UnknownJob(LookupError):
    pass


class JobAlreadyRunning(RuntimeError):
    pass


class ConfirmationRequired(ValueError):
    pass


class JobRunner:
    """Track one process per registered target in the current application process."""

    def __init__(
        self,
        root: Path = ROOT,
        *,
        platform_name: str | None = None,
        max_log_bytes: int = 8192,
    ) -> None:
        self.root = Path(root)
        self.platform_name = platform_name or os.name
        self.max_log_bytes = max_log_bytes
        self._lock = RLock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stopped: set[str] = set()

    def _spec(self, name: str) -> JobSpec:
        try:
            return JOB_SPECS[name]
        except KeyError:
            raise UnknownJob(name) from None

    def _make_executable(self) -> str:
        if self.platform_name == "nt":
            if not WINDOWS_MAKE.is_file():
                raise FileNotFoundError(f"GNU Make not found at {WINDOWS_MAKE}")
            return str(WINDOWS_MAKE)
        make = shutil.which("make")
        if make is None:
            raise FileNotFoundError("GNU Make not found on PATH")
        return make

    def _log_path(self, spec: JobSpec) -> Path:
        return self.root / "log" / "acceptance" / f"{spec.name}.log"

    def _status_locked(self, spec: JobSpec) -> dict[str, object]:
        process = self._processes.get(spec.name)
        code = process.poll() if process is not None else None
        if process is None:
            state = "idle"
        elif spec.name in self._stopped:
            state = "stopped"
        elif code is None:
            state = "running"
        else:
            state = "succeeded" if code == 0 else "failed"
        return {
            "name": spec.name,
            "target": spec.target,
            "heavy": spec.heavy,
            "state": state,
            "exit_code": code,
        }

    def list(self) -> list[dict[str, object]]:
        with self._lock:
            return [self._status_locked(spec) for spec in JOB_SPECS.values()]

    def status(self, name: str) -> dict[str, object]:
        spec = self._spec(name)
        with self._lock:
            result = self._status_locked(spec)
        result["log_tail"] = self.tail(name)
        return result

    def tail(self, name: str) -> str:
        path = self._log_path(self._spec(name))
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - self.max_log_bytes))
                return stream.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def start(self, name: str, *, confirm: bool = False) -> dict[str, object]:
        spec = self._spec(name)
        if spec.heavy and not confirm:
            raise ConfirmationRequired(name)
        with self._lock:
            current = self._processes.get(name)
            if current is not None and current.poll() is None:
                raise JobAlreadyRunning(name)
            make = self._make_executable()
            path = self._log_path(spec)
            path.parent.mkdir(parents=True, exist_ok=True)
            options: dict[str, object] = {
                "cwd": self.root,
                "stdout": None,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
                "shell": False,
            }
            if self.platform_name == "nt":
                options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options["start_new_session"] = True
            with path.open("wb") as stream:
                options["stdout"] = stream
                process = subprocess.Popen([make, spec.target], **options)
            self._processes[name] = process
            self._stopped.discard(name)
            return self._status_locked(spec)

    def stop(self, name: str) -> dict[str, object]:
        spec = self._spec(name)
        with self._lock:
            process = self._processes.get(name)
            if process is None or process.poll() is not None or name in self._stopped:
                return self._status_locked(spec)
            if self.platform_name == "nt":
                result = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                )
                if result.returncode != 0 and process.poll() is None:
                    raise RuntimeError("Unable to stop job process tree")
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("Job process tree did not exit") from None
            else:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            self._stopped.add(name)
            return self._status_locked(spec)

    def stop_all(self) -> None:
        for name in JOB_SPECS:
            self.stop(name)
