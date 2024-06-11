import subprocess
import sys
from pathlib import Path


def check_git(cmd: str, cwd: Path) -> str:
    return subprocess.check_output(f"git {cmd}".split(), cwd=cwd).decode(sys.stdout.encoding).strip()


def find_repository_root() -> Path:
    cwd = Path.cwd()
    assert "Mirandatz/morefun" in check_git("remote -v", cwd=cwd), "Call me only from a clone of the morefun repo!"
    return Path(check_git("rev-parse --show-toplevel", cwd=cwd))



