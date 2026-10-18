"""Repository-level documentation and anonymity regression tests."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
README_PATHS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "uct_engine" / "README.md",
)


class TestRepositoryHygiene(unittest.TestCase):
    def test_readme_is_project_focused_and_submission_neutral(self) -> None:
        readme = README_PATHS[0].read_text(encoding="utf-8")
        lowered = readme.lower()

        required_sections = (
            "## Why AgentUCT",
            "## How the method works",
            "## Quick start",
            "## Integration contract",
            "## Scope",
            "## Testing",
        )
        for section in required_sections:
            self.assertIn(section, readme)

        forbidden_phrases = (
            "anonymous submission",
            "anonymous review",
            "double-blind",
            "conference review system",
            "paper id",
            "submission id",
            "submitted technical supplement",
            "agentgit",
        )
        for phrase in forbidden_phrases:
            self.assertNotIn(phrase, lowered)

    def test_documentation_contains_no_email_addresses(self) -> None:
        email_pattern = re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        )
        for path in README_PATHS:
            self.assertIsNone(email_pattern.search(path.read_text(encoding="utf-8")))

    def test_identity_bearing_manuscript_is_not_in_the_code_repository(self) -> None:
        self.assertFalse((REPOSITORY_ROOT / "main .tex").exists())


if __name__ == "__main__":
    unittest.main()
