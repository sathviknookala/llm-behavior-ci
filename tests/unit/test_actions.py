import unittest

from llm_behavior_ci.runtime.actions import ActionRejected, parse_model_output


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


if __name__ == "__main__":
    unittest.main()
