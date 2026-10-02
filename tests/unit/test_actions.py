import unittest

from llm_behavior_ci.runtime.actions import ActionRejected, parse_model_output
from llm_behavior_ci.runtime.prompts import render_system_text


class ActionParserTests(unittest.TestCase):
    def test_native_calendar_call(self) -> None:
        text = 'apis.calendar.show_calendar(date="2026-01-01")'
        action, app_name, api_name = parse_model_output(text)
        self.assertEqual(action, text)
        self.assertEqual(app_name, "calendar")
        self.assertEqual(api_name, "show_calendar")

    def test_supervisor_complete_task(self) -> None:
        text = "apis.supervisor.complete_task()"
        self.assertEqual(
            parse_model_output(text),
            (text, "supervisor", "complete_task"),
        )

    def test_supervisor_complete_task_with_answer(self) -> None:
        text = 'apis.supervisor.complete_task(answer="done")'
        self.assertEqual(
            parse_model_output(text),
            (text, "supervisor", "complete_task"),
        )

    def test_rejects_prose_plus_call(self) -> None:
        with self.assertRaises(ActionRejected):
            parse_model_output(
                'please run\napis.calendar.show_calendar(date="2026-01-01")'
            )

    def test_rejects_markdown_fence(self) -> None:
        with self.assertRaises(ActionRejected):
            parse_model_output(
                '```python\napis.calendar.show_calendar(date="2026-01-01")\n```'
            )

    def test_rejects_two_calls(self) -> None:
        with self.assertRaises(ActionRejected):
            parse_model_output(
                "apis.calendar.show_calendar()\napis.supervisor.complete_task()"
            )

    def test_rejects_nested_call(self) -> None:
        with self.assertRaises(ActionRejected):
            parse_model_output(
                'apis.calendar.show_calendar(date=apis.supervisor.complete_task())'
            )

    def test_rejects_assignment(self) -> None:
        with self.assertRaises(ActionRejected):
            parse_model_output("x = apis.calendar.show()")

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ActionRejected):
            parse_model_output("")

    def test_rejects_attribute_without_call(self) -> None:
        with self.assertRaises(ActionRejected):
            parse_model_output("apis.calendar")

    def test_stop_and_stop_with_trailer(self) -> None:
        self.assertEqual(parse_model_output("STOP"), (None, None, None))
        self.assertEqual(parse_model_output("STOP\nmore"), (None, None, None))

    def test_legacy_call(self) -> None:
        self.assertEqual(
            parse_model_output("CALL calendar lookup\napp.lookup()"),
            ("app.lookup()", "calendar", "lookup"),
        )

    def test_prompt_v2_execute_asks_for_one_native_call(self) -> None:
        text = render_system_text(
            prompt_version="prompt-v2",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertIn("apis.<app>.<api>(...)", text)
        self.assertIn("apis.supervisor.complete_task(...)", text)
        self.assertIn("Pass every argument by keyword.", text)
        self.assertIn("Do not repeat a call that just failed.", text)
        self.assertIn(
            "When an app requires authentication, obtain credentials through "
            "the documented AppWorld and supervisor APIs.",
            text,
        )
        self.assertIn(
            "Do not guess usernames, passwords, access tokens, IDs, or other "
            "credentials.",
            text,
        )
        self.assertIn(
            "Reuse credential and token values returned by earlier API calls "
            "when a later call requires them.",
            text,
        )
        self.assertNotIn("CALL <app> <api>", text)
        self.assertNotIn("STOP", text)
        legacy = render_system_text(
            prompt_version="prompt-v1",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertIn("CALL <app> <api>", legacy)
        self.assertNotIn("Do not guess usernames", legacy)

    def test_prompt_v3_execute_text_is_unchanged(self) -> None:
        text = render_system_text(
            prompt_version="prompt-v3",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertIn(
            "Before calling any app API that requires authentication, first obtain "
            "the existing user's credentials using the documented supervisor "
            "credential APIs, then log into that app using exactly the returned "
            "credentials.",
            text,
        )
        self.assertIn("apis.<app>.<api>(...)", text)
        self.assertNotIn("supervisor.login", text)
        self.assertNotIn("complete_task() only as the final action", text)
        legacy = render_system_text(
            prompt_version="prompt-v3",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="tool_calling",
            mode="execute",
        )
        self.assertIn("CALL <app> <api>", legacy)

    def test_prompt_v4_execute_keeps_native_format_and_invariants(self) -> None:
        text = render_system_text(
            prompt_version="prompt-v4",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertIn("apis.<app>.<api>(...)", text)
        self.assertIn("apis.supervisor.complete_task(...)", text)
        self.assertIn("Pass every argument by keyword.", text)
        self.assertIn("Do not repeat a call that just failed.", text)
        self.assertIn(
            "Before calling any app API that requires authentication, retrieve "
            "the existing user's credentials using the documented supervisor "
            "credential APIs.",
            text,
        )
        self.assertIn(
            "Use the returned username and password exactly to log into that app.",
            text,
        )
        self.assertIn(
            "Reuse the returned access token for later authenticated calls.",
            text,
        )
        self.assertIn(
            "Never fabricate usernames, passwords, tokens, IDs, or "
            "authentication state.",
            text,
        )
        self.assertIn("such as supervisor.login", text)
        self.assertIn(
            "Do not create a new account unless the user task explicitly "
            "requests account creation.",
            text,
        )
        self.assertIn(
            "return to the credential-retrieval and login flow",
            text,
        )
        self.assertIn(
            "Call complete_task() only as the final action, after the requested "
            "work has actually been performed.",
            text,
        )
        self.assertIn(
            "Never call complete_task() merely because authentication or "
            "another tool call failed.",
            text,
        )
        self.assertIn(
            "do not repeat the same failed call unchanged.",
            text,
        )
        self.assertIn(
            "do not repeatedly issue the identical successful read",
            text,
        )
        self.assertIn(
            "Continue to the next unresolved subgoal instead of looping",
            text,
        )
        self.assertNotIn("CALL <app> <api>", text)
        self.assertNotIn("STOP", text)
        with self.assertRaisesRegex(
            ValueError,
            "prompt-v4 does not support tool_calling action interface",
        ):
            render_system_text(
                prompt_version="prompt-v4",
                plan_format_version="plan-v1",
                thinking_enabled=False,
                action_interface="tool_calling",
                mode="execute",
            )

    def test_runtime_auth_prompt_is_native_and_authentication_neutral(self) -> None:
        text = render_system_text(
            prompt_version="prompt-runtime-auth-v1",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertIn(
            "Authentication and session credentials are managed by the runtime.",
            text,
        )
        self.assertIn("Use tool results to progress toward the requested task.", text)
        self.assertIn("apis.<app>.<api>(...)", text)
        self.assertIn("apis.supervisor.complete_task(...)", text)
        self.assertNotIn("credential APIs", text)
        self.assertNotIn("log into", text)
        self.assertNotIn("access token", text.lower())
        self.assertNotIn("signup", text.lower())
        self.assertNotIn("password", text.lower())
        self.assertNotIn("CALL <app> <api>", text)
        previous = render_system_text(
            prompt_version="prompt-v3",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertIn("credential APIs", previous)
        with self.assertRaisesRegex(
            ValueError,
            "prompt-runtime-auth-v1 does not support tool_calling action interface",
        ):
            render_system_text(
                prompt_version="prompt-runtime-auth-v1",
                plan_format_version="plan-v1",
                thinking_enabled=False,
                action_interface="tool_calling",
                mode="execute",
            )


if __name__ == "__main__":
    unittest.main()
