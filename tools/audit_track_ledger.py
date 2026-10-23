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

  1. every track directory the ledger *claims* exists is on disk -- a bullet can
     also name a path in order to say it is not there, and that is not a claim
  2. every `lineart-*` directory on disk appears in the ledger
  3. stale `git worktree list` entries whose directories are gone
  4. how each track is attached -- worktree of this repo, separate clone, or a
     plain directory -- because a separate clone is invisible to a worktree audit
  5. the reporting gap: each track's last commit against the last notice it sent,
     where a positive gap means work exists that the foundation has not been told
     about
  6. uncommitted or unpushed work sitting in a track
  7. every commit hash the ledger quotes still resolves. A history rewrite, a
     dropped branch or a typo leaves a number that looks like provenance and
     leads nowhere -- found 2026-10-24, when the rewrite that removed the
     published artwork left four dead references behind, three of them to the
     very commits it destroyed. Known-dead hashes are declared in
     `DEAD_HASHES_OK` with the reason
  8. a frozen `doc/CURRENT.md` left in a track tree. Every track cut from the
     foundation inherits a copy, that copy stops updating at the branch point,
     and its own opening paragraph says "this file is the first document to
     read" -- so a fresh session obeying it is handed a dead world. Track J
     found its copy 14 days and 3,682 lines behind on 2026-10-23; the oldest on
     disk was 82 days behind. The name promises currency, and a promise is not
     a check

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

# Prose that names a track directory in order to say it is *not* there. Matched
# against the sentence the name sits in, never the whole bullet, so that one
# released reservation does not suppress the claims around it.
NON_CLAIM = re.compile(
    r"\bnever\b|\bno such\b|\bdoes not exist\b|\bwas not created\b"
    r"|\bno\b[^.]{0,160}?\bever (?:created|existed)\b"
    r"|\breservation is released\b",
    re.IGNORECASE,
)


def run(args, cwd=None):
    """Return stdout, or None if the command fails. Never raises."""
    try:
        done = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def claimed_dirs(bullet):
    """Directory names this bullet asserts exist, in order of appearance.

    A bullet can name a path in order to say it is not there -- "the letter L
    reservation is released (no `../lineart-stroke-bridging` worktree was ever
    created)". Counting that as a claim sends the reader to CURRENT.md to correct
    a record that was already right, which is the expensive direction for the one
    check that does not depend on anyone's attention. So the negation is read out
    of the ledger prose, in the same spirit as the finished marker: the status
    stays in one place. Markdown wraps mid-sentence, so whitespace is normalised
    before the sentence around each name is cut out.
    """
    flat = re.sub(r"\s+", " ", bullet)
    names = []
    for m in re.finditer(r"`\.\./(lineart-[A-Za-z0-9._-]+)`", flat):
        left = max(flat.rfind(". ", 0, m.start()), flat.rfind(" - ", 0, m.start()))
        ends = [e for e in (flat.find(". ", m.end()), flat.find(" - ", m.end())) if e != -1]
        sentence = flat[left + 1 : min(ends) if ends else len(flat)]
        if NON_CLAIM.search(sentence):
            continue
        names.append(m.group(1))
    return names


STATUS_STOPPED = re.compile(r"\bclosed\b|\bdormant\b|\bpaused\b", re.IGNORECASE)
STATUS_ACTIVE = re.compile(r"\bactive\b", re.IGNORECASE)


def entry_is_finished(bullet):
    """Whether a ledger entry declares its own track stopped.

    Only the entry's opening is consulted -- up to the first sentence end after
    the subject, which is where this ledger states status -- and an explicit
    "active" wins over a stopped word in the same opening ("active, idle" is a
    running track). Prose further down may say "closed doors" or "near-closed
    registrations" about something other than the track.
    """
    flat = re.sub(r"\s+", " ", bullet)
    # The opening is everything up to the end of the sentence that names the
    # status, capped so a long first sentence cannot swallow the whole entry.
    opening = flat[:260]
    if STATUS_ACTIVE.search(opening):
        return False
    return bool(STATUS_STOPPED.search(opening))


def ledger_dirs():
    """Track directory -> whether its ledger entry marks it finished.

    "Finished" means closed, paused or dormant: a track that has stopped on
    purpose should not be reported as owing the foundation a notice. The marker is
    read from the ledger bullet itself, so a track's status lives in one place
    rather than being duplicated into this script.

    The status word is taken from the entry's **opening**, not from anywhere in
    its first 900 characters, and an explicit "active" there settles it. Track L's
    entry forced this on 2026-10-23: it opens "active, opened 2026-10-23" and
    goes on to describe "eight closed doors", and the wider window read that
    "closed" as the track's own status and printed `(finished)`. Third instance
    of this script matching a word used for something else -- after the denied
    directory name and the notice that named its branch -- so the general form
    holds: **a check that scans prose for a word will match the word used for
    something else, and the fix is to narrow where it is allowed to look.**
    """
    if not LEDGER.exists():
        return {}
    text = LEDGER.read_text(encoding="utf-8")
    entries = {}
    # Ledger entries are bullets; a bullet runs until the next one starts.
    for bullet in re.split(r"\n- ", text):
        names = claimed_dirs(bullet)
        if not names:
            continue
        finished = entry_is_finished(bullet)
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


# Files the foundation maintains inside every track tree. A commit touching only
# these is housekeeping done *from* the foundation, not work done *by* the track.
FOUNDATION_FILES = ("doc/CURRENT.md", "CLAUDE.md")


def last_commit(path, own_work_only=False):
    """When this tree last committed.

    `own_work_only` skips commits that touch nothing but the files the
    foundation maintains in every tree (`FOUNDATION_FILES`). The reporting gap
    asks "has this track done work it has not told the foundation about", and a
    sweep the foundation itself ran is not that. Without this, the 2026-10-23
    frozen-ledger sweep made the audit report *its own commits* as a track's
    unreported results -- a check contradicted by the housekeeping done to fix
    another of its findings.
    """
    if not own_work_only:
        out = run(["git", "log", "-1", "--format=%cI"], cwd=path)
        return _stamp(out)
    log = run(
        ["git", "log", "--format=%cI%x00%H", "-n", "40"], cwd=path
    )
    if not log:
        return None
    for line in log.splitlines():
        stamp, _, sha = line.partition("\x00")
        if not sha:
            continue
        files = run(
            ["git", "show", "--pretty=", "--name-only", sha], cwd=path
        )
        touched = [f for f in (files or "").splitlines() if f.strip()]
        if touched and all(f in FOUNDATION_FILES for f in touched):
            continue  # foundation housekeeping; keep looking
        return _stamp(stamp)
    return None


def _stamp(out):
    if not out:
        return None
    try:
        return datetime.fromisoformat(out.strip())
    except ValueError:
        return None


# The sentence a frozen ledger copy uses to put itself first. The hazard is not
# that the copy exists -- it is that the copy still makes this claim, which a
# briefing's warning can only mitigate, since the warning has to be read first.
# Hashes the ledger quotes that are known not to resolve, with why. Anything else
# that fails to resolve is a dead reference: a number that looks like provenance
# and leads nowhere.
DEAD_HASHES_OK = {
    "f38a601": "destroyed by the 2026-10-24 history rewrite; named by the entry about it",
    "51623d3": "same rewrite -- one of the two commits that carried the data",
    "7d7087d": "same rewrite -- the other one",
    "53b8406cfbb4f4d1": "a content fingerprint, not a commit",
    "30c5104": "Track L's briefing before the rewrite renumbered it to 7d9f080; "
               "quoted to document the renumbering",
}

HASH_IN_PROSE = re.compile(r"`([0-9a-f]{7,40})`")


def dead_hashes():
    """Hashes quoted in the ledger that do not resolve to a commit here."""
    if not LEDGER.exists():
        return []
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    out = []
    for h in sorted(set(HASH_IN_PROSE.findall(text))):
        if h in DEAD_HASHES_OK:
            continue
        if run(["git", "cat-file", "-e", h + "^{commit}"], cwd=FOUNDATION) is None:
            out.append(h)
    return out


FIRST_DOCUMENT_CLAIM = re.compile(r"first document to read", re.IGNORECASE)


def frozen_ledger(path):
    """(line count, days behind) for a track's own `doc/CURRENT.md`, or None.

    None when the tree has no copy, when the copy is the foundation's own file,
    or when the copy **no longer claims to be the first document to read** --
    which is the property that makes a stale copy dangerous and the only thing
    a fix has to remove. A tree that replaced its copy with a pointer, or
    merged the foundation's conditional wording, reports nothing. The age is
    the gap between the two files' last commits touching that path.
    """
    local = path / "doc" / "CURRENT.md"
    if path == FOUNDATION or not local.is_file():
        return None
    try:
        text = local.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not FIRST_DOCUMENT_CLAIM.search(text):
        return None
    lines = text.count("\n") + 1
    theirs = run(["git", "log", "-1", "--format=%cI", "--", "doc/CURRENT.md"], cwd=path)
    ours = run(["git", "log", "-1", "--format=%cI", "--", "doc/CURRENT.md"], cwd=FOUNDATION)
    days = None
    if theirs and ours:
        try:
            days = (
                datetime.fromisoformat(ours.strip()) - datetime.fromisoformat(theirs.strip())
            ).days
        except ValueError:
            days = None
    return lines, days


def notice_senders(head):
    """Directory names a notice identifies itself with, best source first.

    A `発信: `<dir>`` line is authoritative; otherwise any mention of a
    directory name in the opening; otherwise the branch the notice names, since
    a track on branch `<topic>` lives in `lineart-<topic>` (CLAUDE.md §5).
    """
    sender = re.search(r"発信:\s*`(lineart-[A-Za-z0-9._-]+)`", head)
    if sender:
        return [sender.group(1)]
    names = re.findall(r"`(lineart-[A-Za-z0-9._-]+)`", head[:600])
    if names:
        return names
    return [
        "lineart-" + b
        for b in re.findall(r"branch\s+`([a-z0-9][a-z0-9._-]*)`", head[:600])
    ]


def last_notice():
    """Directory name -> (date, filename) of the most recent notice it sent.

    Notices carry a `発信: `<dir>`` line; fall back to any mention of the
    directory name, then to the branch name a notice gives instead (a track on
    branch `<topic>` lives in `lineart-<topic>`, per CLAUDE.md §5), so a
    differently-formatted notice still counts.
    """
    newest = {}
    if not INBOX.is_dir():
        return newest
    for note in sorted(INBOX.glob("*.md")):
        try:
            head = note.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        names = notice_senders(head)
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


def self_test():
    """Assert the claim reader on the cases that produced it. Exits non-zero on failure."""
    claims = "A track lives in `../lineart-face-words` and is current."
    released = (
        "Track K's second axis closed itself: the letter L reservation is "
        "released (no `stroke-bridging` branch and no "
        "`../lineart-stroke-bridging` worktree was ever created). Commit `78856b2`."
    )
    assert claimed_dirs(claims) == ["lineart-face-words"], claimed_dirs(claims)
    assert claimed_dirs(released) == [], claimed_dirs(released)
    # A negation must not reach past its own sentence.
    both = released + " Work continues in `../lineart-solid-fill`."
    assert claimed_dirs(both) == ["lineart-solid-fill"], claimed_dirs(both)
    # Wrapped markdown: the sentence survives the line break.
    wrapped = released.replace("and no ", "and no\n  ")
    assert claimed_dirs(wrapped) == [], claimed_dirs(wrapped)
    # A notice that names its branch and not its directory still has a sender.
    by_branch = "宛: 基盤 / 2026-10-23 14:20 / branch `solid-fill`(push 済み)"
    assert notice_senders(by_branch) == ["lineart-solid-fill"], notice_senders(by_branch)
    by_dir = "発信: `lineart-face-words` / branch `face-words`"
    assert notice_senders(by_dir) == ["lineart-face-words"], notice_senders(by_dir)
    # Status comes from the entry's opening, and "closed doors" is not a status.
    track_l = (
        "`../lineart-fill-region` (branch `fill-region`, **Track L**) -- **active, "
        "opened 2026-10-23 by user approval**, worktree of this repo. Proposal: "
        "eight closed doors with their measurements, five candidates."
    )
    assert not entry_is_finished(track_l)
    assert entry_is_finished(
        "`../lineart-stroke-topology` (branch `stroke-topology`) -- **CLOSED "
        "2026-10-19**, tree clean and pushed."
    )
    assert not entry_is_finished(
        "`../lineart-stroke-selection` (branch `stroke-selection`) -- active, idle "
        "since 2026-10-04."
    )
    # A frozen ledger is reported for its claim, not for existing.
    assert FIRST_DOCUMENT_CLAIM.search("This file is the first document to read.")
    assert not FIRST_DOCUMENT_CLAIM.search(
        "Moved: the live ledger is ../lineart/doc/CURRENT.md"
    )
    print("self-test: 11 cases pass")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--self-test", action="store_true", help="check the ledger claim reader and exit"
    )
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

    if args.self_test:
        return self_test()

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

    # 7: provenance that leads nowhere.
    for h in dead_hashes():
        findings.append(
            f"the ledger quotes commit `{h}` and it does not resolve -- a number "
            f"that looks like provenance and leads nowhere. Fix the reference, or "
            f"declare it in `DEAD_HASHES_OK` with the reason"
        )

    # 3: worktree registrations pointing at nothing.
    for name, path in sorted(worktrees.items()):
        if name != FOUNDATION.name and not path.exists():
            findings.append(
                f"`git worktree list` still registers `{name}` but its directory is gone "
                f"-- run `git worktree prune`"
            )

    ledger_lines = sum(1 for _ in LEDGER.open(encoding="utf-8", errors="replace"))

    rows = []
    for name in sorted(disk):
        path = SIBLINGS / name
        how = attachment(path, worktrees)
        commit = last_commit(path)
        own = last_commit(path, own_work_only=True)
        notice = notices.get(name)
        uncommitted, unpushed = dirty(path)

        # The gap is about the track's own work, so foundation housekeeping in
        # the tree does not open one.
        gap = None
        if own and notice:
            gap = (own - notice[0]).days
        elif own and not notice:
            gap = (now - own).days

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

        # 7: a frozen ledger copy that still calls itself the first thing to read.
        frozen = frozen_ledger(path)
        if frozen:
            flines, fdays = frozen
            behind = f"{fdays}d and " if fdays and fdays > 0 else ""
            if fdays is None or fdays > 0:
                findings.append(
                    f"`{name}` carries its own `doc/CURRENT.md` ({flines} lines), "
                    f"{behind}{max(ledger_lines - flines, 0)} lines behind the "
                    f"foundation's -- and that copy calls itself the first document "
                    f"to read. Replace that copy with a pointer to "
                    f"`../lineart/doc/CURRENT.md`; a warning in the briefing is "
                    f"weaker, because it has to be read first"
                )

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
