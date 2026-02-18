#!/usr/bin/env python3
"""
v4 문서-코드-테스트 커버리지 게이트.

규칙:
- docs/v4_rule_manifest.yaml (JSON-compatible YAML) 기준
- 각 rule의 code_refs pattern이 지정 파일에 존재해야 함
- 각 rule의 tests 항목이 tests/test_v4_rules.py 내 RULE_ID marker로 존재해야 함
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "v4_rule_manifest.yaml"
TEST_FILE = ROOT / "tests" / "test_v4_rules.py"


def load_manifest(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[FAIL] manifest parse error: {exc}")


def main() -> int:
    if not MANIFEST.exists():
        print(f"[FAIL] missing manifest: {MANIFEST}")
        return 1
    if not TEST_FILE.exists():
        print(f"[FAIL] missing test file: {TEST_FILE}")
        return 1

    manifest = load_manifest(MANIFEST)
    rules = manifest.get("rules", [])
    if not rules:
        print("[FAIL] manifest rules is empty")
        return 1

    test_src = TEST_FILE.read_text(encoding="utf-8")
    test_markers = set(re.findall(r"RULE_ID:\s*([A-Z0-9_]+)", test_src))

    failed = False
    checked_code = 0

    for rule in rules:
        rid = rule.get("id", "UNKNOWN")
        code_refs = rule.get("code_refs", [])
        tests = rule.get("tests", [])

        for ref in code_refs:
            path = ROOT / ref["path"]
            pattern = ref["pattern"]
            if not path.exists():
                print(f"[FAIL] {rid}: missing file {ref['path']}")
                failed = True
                continue
            src = path.read_text(encoding="utf-8")
            checked_code += 1
            if pattern not in src:
                print(f"[FAIL] {rid}: pattern not found in {ref['path']}: {pattern}")
                failed = True

        if tests and rid not in test_markers:
            print(f"[FAIL] {rid}: missing RULE_ID marker in tests ({TEST_FILE})")
            failed = True

    if failed:
        print("[RESULT] v4 coverage gate: FAILED")
        return 1

    print(f"[RESULT] v4 coverage gate: PASSED ({len(rules)} rules, {checked_code} code refs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
