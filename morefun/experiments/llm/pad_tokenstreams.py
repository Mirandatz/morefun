from pathlib import Path


def main() -> None:
    persistent_dir = Path("/workspaces/morefun/persistent")
    output_dir = persistent_dir / "padded_tokenstreams"

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenstreams_dir = persistent_dir / "tokenstreams"

    tokenstreams = {p: p.read_text() for p in tokenstreams_dir.iterdir()}
    biggest_tokenstream = max(tokenstreams.values(), key=len)
    biggest_tokenstream_len = len(biggest_tokenstream)

    # pad all tokenstreams
    for path, tokenstream in tokenstreams.items():
        padding_len = biggest_tokenstream_len - len(tokenstream)
        padding = "¿" * padding_len
        padded_tokenstream = padding + tokenstream
        path = output_dir / path.name
        path.write_text(padded_tokenstream)


if __name__ == "__main__":
    main()
