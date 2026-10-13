#!/usr/bin/env python3
"""Install byte-identical benchmark adapters into pinned official repositories."""

import argparse
from pathlib import Path
import shutil
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORIES = {
    "AD-CLIP": ("https://github.com/mainaksingha01/AD-CLIP.git",
                "06a30ce32ab7a0dcc29a5ac72e66c4e868f70c89"),
    "DAPrompt": ("https://github.com/LeapLabTHU/DAPrompt.git",
                 "f24b0c16c06b585104ba97615f89cd5df4def872"),
}


def install(name, root):
    url, revision = REPOSITORIES[name]
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(root)], check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=root, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != revision:
        raise ValueError(f"{root}: expected {revision}, found {actual}; existing clones are not reset")
    subprocess.run(["git", "diff", "--exit-code", revision, "--",
                    "trainers", "clip", "configs", "datasets"], cwd=root, check=True)
    sources = PROJECT / "external" / "adapters" / name
    files = sorted(path for path in sources.rglob("*") if path.is_file())
    # Check every destination before copying; never overwrite local changes.
    for source in files:
        destination = root / source.relative_to(sources)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Adapter differs: {destination}")
    for source in files:
        destination = root / source.relative_to(sources)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    print(f"Ready: {root} at {revision}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace", type=Path, default=PROJECT.parent)
    args = cli.parse_args()
    for name in REPOSITORIES:
        install(name, args.workspace.resolve() / name)
    print("Run train_vlm_competitors.py from the PCTE checkout with explicit repository paths.")


if __name__ == "__main__":
    main()
