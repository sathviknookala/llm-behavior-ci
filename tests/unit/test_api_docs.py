from __future__ import annotations

import unittest

from llm_behavior_ci.runtime.api_docs import (
    ApiDocsCorruptionError,
    UnknownApiDocsVersion,
    resolve_api_documentation,
)

_DOCS = (
    "calendar.create_event: name, start_time, end_time\n"
    "calendar.list_events: start_time, end_time\n"
    "supervisor.show_account_passwords: none\n"
    "supervisor.complete_task: status\n"
)


class ResolveApiDocumentationTests(unittest.TestCase):
    def test_both_unset_returns_the_source_text_unchanged(self) -> None:
        self.assertEqual(
            resolve_api_documentation(
                _DOCS,
                api_docs_version=None,
                api_docs_app=None,
            ),
            _DOCS,
        )

    def test_corrupts_only_the_named_apps_lines(self) -> None:
        corrupted = resolve_api_documentation(
            _DOCS,
            api_docs_version="api-docs-corrupt-v1",
            api_docs_app="supervisor",
        )
        self.assertIn("calendar.create_event: name, start_time, end_time", corrupted)
        self.assertIn("calendar.list_events: start_time, end_time", corrupted)
        self.assertIn("supervisor.show_account_passwords: [documentation removed]", corrupted)
        self.assertIn("supervisor.complete_task: [documentation removed]", corrupted)
        self.assertNotIn("none", corrupted)
        self.assertNotIn("status", corrupted)
        self.assertNotEqual(corrupted, _DOCS)

    def test_corruption_is_case_insensitive_on_the_app_name(self) -> None:
        corrupted = resolve_api_documentation(
            _DOCS,
            api_docs_version="api-docs-corrupt-v1",
            api_docs_app="Supervisor",
        )
        self.assertIn("supervisor.show_account_passwords: [documentation removed]", corrupted)

    def test_unknown_app_leaves_every_line_unchanged(self) -> None:
        corrupted = resolve_api_documentation(
            _DOCS,
            api_docs_version="api-docs-corrupt-v1",
            api_docs_app="venmo",
        )
        self.assertEqual(corrupted, _DOCS)

    def test_deterministic_across_calls(self) -> None:
        first = resolve_api_documentation(
            _DOCS,
            api_docs_version="api-docs-corrupt-v1",
            api_docs_app="calendar",
        )
        second = resolve_api_documentation(
            _DOCS,
            api_docs_version="api-docs-corrupt-v1",
            api_docs_app="calendar",
        )
        self.assertEqual(first, second)

    def test_unknown_version_is_rejected(self) -> None:
        with self.assertRaises(UnknownApiDocsVersion):
            resolve_api_documentation(
                _DOCS,
                api_docs_version="api-docs-corrupt-v99",
                api_docs_app="calendar",
            )

    def test_setting_only_one_field_is_rejected(self) -> None:
        with self.assertRaises(ApiDocsCorruptionError):
            resolve_api_documentation(
                _DOCS,
                api_docs_version="api-docs-corrupt-v1",
                api_docs_app=None,
            )
        with self.assertRaises(ApiDocsCorruptionError):
            resolve_api_documentation(
                _DOCS,
                api_docs_version=None,
                api_docs_app="calendar",
            )

    def test_no_colon_line_keeps_only_its_header(self) -> None:
        corrupted = resolve_api_documentation(
            "calendar.no_colon_here\n",
            api_docs_version="api-docs-corrupt-v1",
            api_docs_app="calendar",
        )
        self.assertEqual(corrupted, "calendar.no_colon_here: [documentation removed]\n")


if __name__ == "__main__":
    unittest.main()
