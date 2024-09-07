import re
from pathlib import Path
from typing import Optional


def main() -> None:
    persistent_dir = Path("/workspaces/morefun/persistent")
    output_dir = persistent_dir / "padded_tokenstreams"

    tokenstream_paths = list(persistent_dir.glob("v2/run_*_tokenstreams/*"))

    for tokenstream_path in tokenstream_paths:
        run_nr = re.search(r"run_(\d+)_tokenstreams", str(tokenstream_path)).group(1)
        print(run_nr)


if __name__ == "__main__":
    main()
