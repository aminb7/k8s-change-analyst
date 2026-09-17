"""Assert the verdict of the most recent change of a given kind in the newest report."""
import json
import sys
from pathlib import Path


def main(kind: str, expected: str, reports_dir: str = "reports") -> int:
    files = sorted(Path(reports_dir).glob("*.json"))
    if not files:
        print(f"FAIL: no reports in {reports_dir}")
        return 1
    report = json.loads(files[-1].read_text())
    candidates = [r for r in report["results"] if r["change"]["kind"] == kind]
    if not candidates:
        print(f"FAIL: no {kind} change analyzed in {files[-1].name}")
        return 1
    latest = max(candidates, key=lambda r: r["change"]["changed_at"])
    got = latest["verdict"]["verdict"]
    status = "PASS" if got == expected else "FAIL"
    print(f"{status}: {latest['change']['id']} expected={expected} got={got} "
          f"(confidence {latest['verdict']['confidence']})")
    return 0 if got == expected else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
