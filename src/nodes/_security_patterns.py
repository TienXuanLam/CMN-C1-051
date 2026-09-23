"""AgentCore Platform v1.0"""

# Shared security patterns for CMN-C1-051.
# Single source of truth — used by pre_process_node (S-1/S-2 gate)
# and post_process_node (S-3 gate). Patch here only.
#
# Finding (2026-08-19, High): the hand-rolled PII/credential regexes previously
# here duplicated a REAL substrate capability (the framework's substrate-reuse
# rule) -- shared.security.detect_pii()/detect_credentials() already ship this.
# The hand-rolled email/PII patterns also used unbounded quantifiers
# ([a-zA-Z0-9._%+\-]+, \S+) -- a ReDoS risk in violation of the framework's
# PII-regex bounding rule, confirmed live: an
# adversarial "a"*50000 + "@" + "b"*50000 + "." payload (well within this
# template's own max_input_chars=1,000,000 limit) took ~23s to scan. The
# hand-rolled CREDENTIAL_PATTERNS also missed value-shapes the framework
# detector covers (JWT, AWS access key, DB connection strings), confirmed live
# against representative fixtures.
#
# scan_for_credentials()/scan_for_pii() below wrap the framework detectors;
# INJECTION_PATTERNS remains per-template (prompt injection has no framework
# substrate equivalent), with every quantifier bounded.

import re

from shared.security.credential_detector import detect_credentials
from shared.security.pii_detector import detect_pii

# ── S-2 / S-3: Credentials ───────────────────────────────────────────────────


def scan_for_credentials(text: str) -> list[str]:
    """Return the credential type names found in text, or [] if none."""
    return [f["type"] for f in detect_credentials(text)]


# ── S-2: PII ─────────────────────────────────────────────────────────────────


def scan_for_pii(text: str) -> list[str]:
    """Return the PII type names found in text, or [] if none."""
    return [f["type"] for f in detect_pii(text)]


# ── S-1: Injection ────────────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|context)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"(system\s*:|\[INST\]|<\|im_start\|>)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]
