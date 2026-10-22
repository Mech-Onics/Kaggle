"""Safely declutter the EXISTING C:\\AI Workspace\\Kaggle repository.

Preview: python cleanup_existing_repo.py
Execute: python cleanup_existing_repo.py --apply

Moves known superseded experiments to a timestamped folder BESIDE the repo,
never deletes files, does not create a Git repo, and never touches working
training/test datasets or the best/last geometry-adversarial checkpoints.
"""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys

CORE_NOTEBOOKS = {"Geometry_Adversarial_Training.ipynb", "Test_Submission.ipynb"}
LEGACY_NAMES = {
    "deep_learning.ipynb",
    "Image_Audit.ipynb",
    "Rebuilt_Pipeline_PreTraining.ipynb",
    "Robust_Recognition_PreTraining.ipynb",
    "staged_dataset.py",
    "model_comparison.csv",
    "staged_checkpoints",
    "__pycache__",
}
LEGACY_GLOBS = ("final_submission*.csv", "submission_variable*.csv")
ESSENTIALS = {
    Path("outputs/preflight_audit/group_assignments.csv"),
    Path("outputs/robust_recognition_pipeline/best_geometry_adversarial.pt"),
    Path("outputs/robust_recognition_pipeline/last_geometry_adversarial.pt"),
    Path("outputs/robust_recognition_pipeline/geometry_adversarial_history.csv"),
    Path("outputs/robust_recognition_pipeline/train_val_split.csv"),
    Path("outputs/robust_recognition_pipeline/submission_geometry_adversarial.csv"),
    Path("outputs/robust_recognition_pipeline/submission_geometry_adversarial_flip_tta.csv"),
}


def run():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Move legacy files; otherwise show a preview only")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Existing Kaggle root folder")
    ap.add_argument("--archive", type=Path, default=None, help="Archive destination OUTSIDE the repo")
    args = ap.parse_args()
    root = args.root.resolve()
    if not all((root / x).is_file() for x in CORE_NOTEBOOKS):
        sys.exit("SAFETY STOP: Run inside existing Kaggle root; both active notebooks must exist.")
    if not (root / "train_labels.csv").is_file():
        sys.exit("SAFETY STOP: train_labels.csv missing from project root.")

    archive = args.archive.resolve() if args.archive else root.parent / (
        root.name + "_legacy_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if root == archive or root in archive.parents:
        sys.exit("SAFETY STOP: Archive must be outside the existing repository.")
    if archive.exists():
        sys.exit("SAFETY STOP: Archive target already exists; choose a new --archive path.")

    moves: list[tuple[Path, Path]] = []
    targets: set[Path] = set()
    for name in LEGACY_NAMES:
        p = root / name
        if p.exists() and p not in targets:
            moves.append((p, archive / name))
            targets.add(p)
    for pattern in LEGACY_GLOBS:
        for p in root.glob(pattern):
            if p.is_file() and p not in targets:
                moves.append((p, archive / p.name))
                targets.add(p)

    # Retain exactly the preflight group map needed by the active notebook;
    # archive review lists and diagnostic images that no longer drive training.
    audit = root / "outputs" / "preflight_audit"
    if audit.is_dir():
        for p in audit.iterdir():
            if p.name == "group_assignments.csv":
                continue
            if p not in targets:
                moves.append((p, archive / "outputs" / "preflight_audit" / p.name))
                targets.add(p)

    print("Existing repository:", root)
    print("Archive OUTSIDE repository:", archive)
    print("Mode:", "MOVE (no deletion)" if args.apply else "DRY RUN (no changes)")
    print("\nWill archive:")
    if not moves:
        print("  (nothing matched; repository is already clean)")
    for src, dst in sorted(moves, key=lambda pair: str(pair[0])):
        print(" -", src.relative_to(root), "->", dst.relative_to(archive))
    print("\nWill preserve the following core files and folders:")
    for name in sorted(CORE_NOTEBOOKS | {"README.md", ".gitignore", "train_labels.csv", "sample_submission.csv", "train", "test"}):
        print(" +", name)
    for p in sorted(ESSENTIALS):
        print(" +", p.as_posix(), "(exists)" if (root / p).exists() else "(not found; NOT required for cleanup)")

    try:
        status = subprocess.run(["git", "status", "--short"], cwd=root, text=True,
                                capture_output=True, check=False)
        if status.returncode == 0:
            print("\nExisting Git changes (review before committing):")
            print(status.stdout.strip() or "  Working tree clean")
    except FileNotFoundError:
        print("\nGit not available; local file cleanup is still possible.")

    if not args.apply:
        print("\nNo files changed. Run: python cleanup_existing_repo.py --apply")
        return

    # Keep all existing .gitignore entries. Only append this project's new rules.
    additions = root / "gitignore_additions.txt"
    existing_ignore = root / ".gitignore"
    if additions.is_file():
        section_marker = "# ENGE707 active pipeline: local-only data and outputs"
        current = existing_ignore.read_text(encoding="utf-8") if existing_ignore.exists() else ""
        if section_marker not in current:
            new_rules = additions.read_text(encoding="utf-8").rstrip()
            with existing_ignore.open("a", encoding="utf-8") as fd:
                if current and not current.endswith("\n"):
                    fd.write("\n")
                fd.write("\n" + new_rules + "\n")
            print("Updated .gitignore by appending rules; retained all existing rules.")

    # Build archive only after checks pass, and never overwrite existing files.
    if moves:
        archive.mkdir(parents=True, exist_ok=False)
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                sys.exit(f"SAFETY STOP: Destination exists: {dst}")
            shutil.move(str(src), str(dst))
            print("Moved:", src.relative_to(root))
    print("\nCompleted. No files deleted. Archive:", archive if moves else "(none)")
    print("Check `git status` and notebooks, then commit selected changes to your SAME repository.")


if __name__ == "__main__":
    run()
