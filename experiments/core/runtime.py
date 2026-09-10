"""Small, dependency-free helpers shared by the AE entry points."""

from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "experiments" / "configs"


def absolute(path):
    return Path(path).expanduser().resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_stats(paths):
    return {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths}


def run_logged(command, log_path, *, env=None):
    command = [str(x) for x in command]
    print(f"Running: {shlex.join(command)}\nLog: {log_path}", flush=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another experiment is using {path}; run sequentially") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ensure_parser(log_path):
    parser = ROOT / "preprocess" / "parser"
    inputs = list((parser / "src").glob("*.cpp")) + list((parser / "include").glob("*.hpp"))
    inputs.append(parser / "CMakeLists.txt")
    binaries = [parser / "bin" / name for name in ("processor", "data_processor", "monitor")]
    with exclusive_lock(ROOT / ".pids" / "ae-build.lock"):
        newest = max(p.stat().st_mtime_ns for p in inputs)
        if all(p.is_file() and os.access(p, os.X_OK) and p.stat().st_mtime_ns >= newest for p in binaries):
            return
        build = parser / "build"
        run_logged(["cmake", "-S", parser, "-B", build], log_path)
        run_logged(["cmake", "--build", build, "--parallel", "2", "--target", "processor", "data_processor", "monitor"], log_path)
        (parser / "bin").mkdir(exist_ok=True)
        for binary in binaries:
            shutil.copy2(build / "bin" / binary.name, binary)
