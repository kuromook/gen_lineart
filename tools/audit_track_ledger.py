#!/usr/bin/env python3
"""Reconcile the common foundation's track ledger against what is actually on disk.

Why this exists: between 2026-09-17 and 2026-09-30 this project's foundation fell
thirteen days behind without anyone noticing. Four tracks kept committing and sent
nothing; three proposals sat in gitignored `outbox/` directories; two tracks were
invisible to `git worktree list` because one is a separate clone and the other is a
worktree of that clone; and two directories named in `doc/worktree_policy.md` had
stopped existing. Every one of those is mechanically detectable, and all of them
were being caught -- when they were caught at all -- by a human remembering to ask
"has anything come in?".

So this is the one piece of discipline in the set that does not rely on attention.
The companion rules (quote no number without its configuration; anchor every new
measurement against a known cell; the foundation session does not run experiments)
cost nothing to follow and are written in `doc/CURRENT.md`. This one needed code.

What it checks:

  1. every track directory named in CURRENT.md's ledger exists on disk
  2. every `lineart-*` directory on disk appears in the ledger
  3. stale `git worktree list` entries whose directories are gone
  4. how each track is attached -- worktree of this repo, separate clone, or a
     plain directory -- because a separate clone is invisible to a worktree audit
  5. the reporting gap: each track's last commit against the last notice it sent,
     where a positive gap means work exists that the foundation has not been told
     about
  6. uncommitted or unpushed work sitting in a track

Read-only. It never writes, commits, or touches another tree.

Usage:
    ./venv/bin/python tools/audit_track_ledger.py            # report
    ./venv/bin/python tools/audit_track_ledger.py --quiet    # findings only
    echo $?                                                  # 1 if findings
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FOUNDATION = Path(__file__).resolve().parent.parent
SIBLINGS = FOUNDATION.parent
LEDGER = FOUNDATION / "doc" / "CURRENT.md"
INBOX = FOUNDATION / "inbox"

# A track whose last commit is older than this many days behind its last notice is
# not interesting -- it is simply idle. The gap we care about is work done and not
# reported, which is the opposite sign.
QUIET_DAYS = 3


def run(args, cwd=None):
    """Return stdout, or None if the command fails. Never raises."""
    try:
        done = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def ledger_dirs():
    """Track directory -> whether its ledger entry marks it finished.

    "Finished" means closed, paused or dormant: a track that has stopped on
    purpose should not be reported as owing the foundation a notice. The marker is
    read from the ledger bullet itself, so a track's status lives in one place
    rather than being duplicated into this script.
    """
    if not LEDGER.exists():
        return {}
    text = LEDGER.read_text(encoding="utf-8")
    entries = {}
    # Ledger entries are bullets; a bullet runs until the next one starts.
    for bullet in re.split(r"\n- ", text):
        names = re.findall(r"`\.\./(lineart-[A-Za-z0-9._-]+)`", bullet)
        if not names:
            continue
        head = bullet[:900]
        finished = bool(
            re.search(r"\bCLOSED\b|\bclosed\b|\bdormant\b|\bpaused\b", head)
        )
        # The first name in a bullet is its subject; later ones are cross-references.
        entries.setdefault(names[0], finished)
        for other in names[1:]:
            entries.setdefault(other, entries.get(other, False))
    return entries


def disk_dirs():
    return {
        p.name
        for p in SIBLINGS.iterdir()
        if p.is_dir() and p.name.startswith("lineart-")
    }


def registered_worktrees():
    """Directory name -> path, from `git worktree list` in the foundation."""
    out = run(["git", "worktree", "list", "--porcelain"], cwd=FOUNDATION)
    found = {}
    if not out:
        return found
    for line in out.splitlines():
        if line.startswith("worktree "):
            path = Path(line[len("worktree ") :])
            found[path.name] = path
    return found


def attachment(path, worktrees):
    """How this directory relates to the foundation repo."""
    if path.name in worktrees:
        return "worktree"
    git = path / ".git"
    if not git.exists():
        return "plain directory (not a repo)"
    common = run(["git", "rev-parse", "--git-common-dir"], cwd=path)
    if common:
        resolved = (path / common).resolve() if not Path(common).is_absolute() else Path(common)
        foundation_git = (FOUNDATION / ".git").resolve()
        if resolved == foundation_git:
            return "worktree"
        # Its own object store: an independent clone, or a worktree of one.
        parent = resolved.parent
        if parent != path:
            return f"separate clone (git dir under {parent.name}/)"
        return "separate clone"
    return "unknown"


def last_commit(path):
    out = run(["git", "log", "-1", "--format=%cI"], cwd=path)
    if not out:
        return None
    try:
        return datetime.fromisoformat(out)
    except ValueError:
        return None


def last_notice():
    """Directory name -> (date, filename) of the most recent notice it sent.

    Notices carry a `発信: `<dir>`` line; fall back to any mention of the
    directory name so a differently-formatted notice still counts.
    """
    newest = {}
    if not INBOX.is_dir():
        return newest
    for note in sorted(INBOX.glob("*.md")):
        try:
            head = note.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        sender = re.search(r"発信:\s*`(lineart-[A-Za-z0-9._-]+)`", head)
        names = (
            [sender.group(1)]
            if sender
            else re.findall(r"`(lineart-[A-Za-z0-9._-]+)`", head[:600])
        )
        stamp = re.search(r"(\d{8})", note.name)
        when = None
        if stamp:
            try:
                when = datetime.strptime(stamp.group(1), "%Y%m%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                when = None
        if when is None:
            when = datetime.fromtimestamp(note.stat().st_mtime, tz=timezone.utc)
        for name in names:
            if name not in newest or when > newest[name][0]:
                newest[name] = (when, note.name)
    return newest


def dirty(path):
    """(uncommitted file count, unpushed commit count); None where not applicable."""
    status = run(["git", "status", "--porcelain"], cwd=path)
    uncommitted = len([l for l in status.splitlines() if l.strip()]) if status is not None else None
    ahead = run(["git", "log", "--oneline", "@{u}..HEAD"], cwd=path)
    unpushed = len([l for l in ahead.splitlines() if l.strip()]) if ahead is not None else None
    return uncommitted, unpushed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--quiet", action="store_true", help="print findings only, skip the table"
    )
    parser.add_argument(
        "--quiet-days",
        type=int,
        default=QUIET_DAYS,
        help=f"days of unreported commits before it counts as a gap (default {QUIET_DAYS})",
    )
    args = parser.parse_args()

    ledger_status = ledger_dirs()
    ledger = set(ledger_status)
    disk = disk_dirs()
    worktrees = registered_worktrees()
    notices = last_notice()
    now = datetime.now(timezone.utc)

    findings = []

    # 1 / 2: the ledger against the filesystem.
    for name in sorted(ledger - disk):
        findings.append(
            f"ledger names `{name}` but no such directory exists -- "
            f"anyone following CURRENT.md will hit a dead end"
        )
    for name in sorted(disk - ledger):
        findings.append(
            f"`{name}` exists on disk but is in no ledger entry -- "
            f"a track the foundation does not know about"
        )

    # 3: worktree registrations pointing at nothing.
    for name, path in sorted(worktrees.items()):
        if name != FOUNDATION.name and not path.exists():
            findings.append(
                f"`git worktree list` still registers `{name}` but its directory is gone "
                f"-- run `git worktree prune`"
            )

    rows = []
    for name in sorted(disk):
        path = SIBLINGS / name
        how = attachment(path, worktrees)
        commit = last_commit(path)
        notice = notices.get(name)
        uncommitted, unpushed = dirty(path)

        gap = None
        if commit and notice:
            gap = (commit - notice[0]).days
        elif commit and not notice:
            gap = (now - commit).days

        rows.append((name, how, commit, notice, gap, uncommitted, unpushed))

        # 4: separate clones are invisible to a worktree audit, so they must be
        # named explicitly somewhere or they silently disappear.
        if how.startswith("separate clone") and name in ledger:
            pass  # in the ledger, so it is discoverable; nothing to report
        elif how.startswith("separate clone"):
            findings.append(
                f"`{name}` is a {how} -- it will never appear in `git worktree list` "
                f"from the foundation, so it must be named in the ledger or it is invisible"
            )

        # 5: the reporting gap -- but a track that stopped on purpose owes nothing.
        finished = ledger_status.get(name, False)
        if not finished and gap is not None and gap > args.quiet_days:
            if notice:
                findings.append(
                    f"`{name}` has committed {gap}d past its last notice "
                    f"({notice[1]}) -- unreported results are likely"
                )
            else:
                findings.append(
                    f"`{name}` has never sent a notice and last committed {gap}d ago "
                    f"-- the foundation has no record of what it found"
                )

        # 6: work that exists only on this machine.
        if uncommitted:
            findings.append(
                f"`{name}` has {uncommitted} uncommitted path(s)"
                + (" -- expected if a session is working there now" if gap == 0 else "")
            )
        if unpushed:
            findings.append(f"`{name}` has {unpushed} unpushed commit(s)")

    if not args.quiet:
        print(f"Track ledger audit -- {now.astimezone().strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"foundation: {FOUNDATION}")
        print()
        head = f"{'directory':34} {'attached as':24} {'last commit':12} {'last notice':12} {'gap':>5}"
        print(head)
        print("-" * len(head))
        for name, how, commit, notice, gap, _unc, _unp in rows:
            mark = " (finished)" if ledger_status.get(name) else ""
            print(
                f"{name + mark:34} {how:24} "
                f"{commit.strftime('%m-%d') if commit else '--':12} "
                f"{notice[0].strftime('%m-%d') if notice else 'never':12} "
                f"{(str(gap) + 'd') if gap is not None else '--':>5}"
            )
        print()

    if findings:
        print(f"{len(findings)} finding(s):")
        for f in findings:
            print(f"  - {f}")
        return 1

    print("no findings: ledger, filesystem and notices agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
