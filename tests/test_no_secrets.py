"""Repo ichida ochiq sir va commit qilingan secrets fayli yo'qligini tekshiradi.

SEC-002 regressiya testi. Muammo tuzatilgunga qadar YIQILADI.
"""
import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_PATTERNS = [
    r"""\bAIza[0-9A-Za-z_\-]{30,}\b""",   # Google API key
    r"""\bsk-[A-Za-z0-9]{20,}\b""",
    r"""\bgsk_[A-Za-z0-9]{20,}\b""",
    r"""\b\d{8,10}:[A-Za-z0-9_-]{30,}\b""",
    r"""\bAKIA[0-9A-Z]{16}\b""",
]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "tests"}
SCAN_EXT = {".py", ".js", ".ts", ".env", ".yml", ".yaml", ".toml", ".json"}


def iter_files():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SCAN_EXT or name == ".env":
                yield os.path.join(root, name)


class NoExposedSecretsTest(unittest.TestCase):
    def test_no_streamlit_secrets_committed(self):
        """`.streamlit/secrets.toml` repoda bo'lmasligi kerak (namuna qoladi)."""
        p = os.path.join(REPO_ROOT, ".streamlit", "secrets.toml")
        self.assertFalse(
            os.path.exists(p),
            ".streamlit/secrets.toml repoda mavjud — uni olib tashlang, "
            "faqat secrets.toml.example qoldiring",
        )

    def test_no_hardcoded_secrets_in_source(self):
        findings = []
        for path in iter_files():
            base = os.path.basename(path)
            if base.endswith(".example"):
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            for pattern in SECRET_PATTERNS:
                for m in re.finditer(pattern, text):
                    line_no = text[: m.start()].count("\n") + 1
                    rel = os.path.relpath(path, REPO_ROOT)
                    findings.append(f"{rel}:{line_no} [{m.group(0)[:10]}...]")
        self.assertEqual(findings, [], "Hardcoded sirlar topildi:\n" + "\n".join(findings))


if __name__ == "__main__":
    unittest.main()
