"""Tests for cai.cli helpers - the headless run driver and flag validation."""
import io
import os
import sys
import json
import argparse
import threading

import pytest

from cai import cli
from cai.events import Event, EventType


class FakeDriveRun:
    """the slice of a Run that _drive consumes: iterable Events, then text."""

    def __init__(self, events, text="answer"):
        self._events = events
        self.text = text
        self.stream = True
        self.interrupt = threading.Event()

    def __iter__(self):
        return iter(self._events)


def test_drive_streams_reasoning_by_default(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_STDOUT_TTY", True)
    events = [Event(type=EventType.REASONING, text="pondering"),
              Event(type=EventType.CONTENT, text="answer")]
    assert cli._drive(FakeDriveRun(events)) == 0
    out = capsys.readouterr().out
    assert "pondering" in out
    assert "answer" in out


def test_drive_respects_show_reasoning_off(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_STDOUT_TTY", True)
    events = [Event(type=EventType.REASONING, text="pondering"),
              Event(type=EventType.CONTENT, text="answer")]
    assert cli._drive(FakeDriveRun(events), show_reasoning=False) == 0
    out = capsys.readouterr().out
    assert "pondering" not in out
    assert "answer" in out


def test_version_prints_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"cai {cli.cai_version()}"
    assert cli.cai_version() != ""


def test_line_by_line_needs_a_prompt():
    with pytest.raises(SystemExit):
        cli.main(["--line-by-line"])


def test_line_by_line_rejects_watch():
    with pytest.raises(SystemExit):
        cli.main(["--line-by-line", "--watch", "-p", "x"])


def test_line_by_line_rejects_interactive():
    with pytest.raises(SystemExit):
        cli.main(["--line-by-line", "-i", "-p", "x"])


def test_cores_must_be_positive():
    with pytest.raises(SystemExit):
        cli.main(["--line-by-line", "-p", "x", "--cores", "0"])


def test_diag_tool_call_renders_long_args_as_a_block(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_STDERR_TTY", True)
    cli._diag_tool_call("fs__edit_file", {"file_path": "a.py", "old_text": "x\ny", "new_text": "z"})
    err = capsys.readouterr().err
    from cai.screen.ansi import ansi_strip
    assert ansi_strip(err).splitlines() == ["  -> fs__edit_file(file_path=a.py, new_text=z)",
                                            "    old_text:",
                                            "      x",
                                            "      y"]
    monkeypatch.setattr(cli, "_STDERR_TTY", False)
    cli._diag_tool_call("fs__edit_file", {"old_text": "x\ny"})
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------
# --system-one
# --------------------------------------------------------------------------

import cai.api as api
from cai import config


class FakeSystemOneResponse:
    def __init__(self, body, status_code=200):
        self.status_code = status_code
        self.text = "error-body"
        self._body = body

    def json(self):
        return self._body

    def close(self):
        return


class SystemOnePost:
    """a fake requests.post for the System One endpoint: records every
    request and answers each with `reply(state)` - the answer for that
    state - so line-by-line tests can map lines to distinct answers."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, url, **kwargs):
        record = {}
        record['url'] = url
        record['kwargs'] = kwargs
        self.calls.append(record)
        body = {}
        body['model'] = "jev-latest"
        body['answers'] = {"answer": self.reply(kwargs['json']['state'])}
        body['usage'] = {"input_tokens": 1, "output_tokens": 1}
        return FakeSystemOneResponse(body)

    @property
    def data(self):
        return self.calls[-1]['kwargs']['json']


def _write_config(tmp_path, monkeypatch, data):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    os.makedirs(config.config_dir(), exist_ok=True)
    with open(config.config_path(), "w") as f:
        json.dump(data, f)
    with open(config.api_key_path(), "w") as f:
        f.write("sk-test\n")


def _install_system_one(tmp_path, monkeypatch, reply, stdin="the state\n", model="jev-latest"):
    data = {}
    data['base_url'] = "https://example.test/v1"
    data['model'] = "chat-model"
    if model is not None:
        data['system_one_model'] = model
    _write_config(tmp_path, monkeypatch, data)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    post = SystemOnePost(reply)
    monkeypatch.setattr(api.requests, "post", post)
    return post


def _flags(**values):
    args = argparse.Namespace(choice=None, score=None)
    for key, value in values.items():
        setattr(args, key, value)
    return args


def test_system_one_needs_a_question():
    with pytest.raises(SystemExit):
        cli.main(["--system-one"])


def test_system_one_rejects_interactive():
    with pytest.raises(SystemExit):
        cli.main(["--system-one", "-i", "-p", "x"])


def test_system_one_rejects_watch():
    with pytest.raises(SystemExit):
        cli.main(["--system-one", "--watch", "-p", "x"])


def test_choice_and_score_are_exclusive():
    with pytest.raises(SystemExit):
        cli.main(["--system-one", "--choice", "a", "b", "--score", "x", "y", "-p", "q"])


def test_score_needs_two_levels():
    with pytest.raises(SystemExit):
        cli.main(["--system-one", "--score", "only", "-p", "q"])


def test_choice_without_system_one_is_an_error():
    with pytest.raises(SystemExit):
        cli.main(["--choice", "a", "b", "-p", "q"])


def test_question_noul_by_default():
    question = cli._system_one_question(_flags(), "is it bad?")
    assert question == {"type": "noul", "instructions": "is it bad?"}


def test_question_choice_with_descriptions():
    args = _flags(choice=["billing=payments, refunds", "technical=bugs", "sales"])
    question = cli._system_one_question(args, "which team?")
    assert question['type'] == "choice"
    assert question['instructions'] == "which team?"
    assert question['criteria'] == {"billing": "payments, refunds",
                                    "technical": "bugs",
                                    "sales": None}


def test_question_score_keeps_level_order():
    args = _flags(score=["calm", "frustrated", "very angry"])
    question = cli._system_one_question(args, "how angry?")
    assert question == {"type": "score",
                        "instructions": "how angry?",
                        "criteria": ["calm", "frustrated", "very angry"]}


def test_answer_line_shapes():
    assert cli._system_one_answer_line({"type": "noul", "noul": 0.92}) == "0.92"
    choice = {"type": "choice", "choice": "technical", "confidence": 0.82,
              "probabilities": {"technical": 0.85, "billing": 0.15}}
    assert cli._system_one_answer_line(choice) == "technical\t0.82"
    score = {"type": "score", "score": 1.6, "confidence": 0.78,
             "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65}}
    assert cli._system_one_answer_line(score) == "1.6\t0.78"


def test_system_one_one_shot_over_stdin(tmp_path, monkeypatch, capsys):
    def reply(state):
        return {"type": "noul", "noul": 0.92}
    post = _install_system_one(tmp_path, monkeypatch, reply, stdin="rm -rf /\n")
    assert cli.main(["--system-one", "--", "is", "this", "dangerous?"]) == 0
    assert capsys.readouterr().out == "0.92\n"
    assert post.calls[-1]['url'] == "https://example.test/v1/systemone"
    assert post.data == {"model": "jev-latest",
                         "state": "rm -rf /\n",
                         "questions": {"answer": {"type": "noul",
                                                  "instructions": "is this dangerous?"}}}
    assert post.calls[-1]['kwargs']['headers']['Authorization'] == "Bearer sk-test"


def test_system_one_one_shot_over_file(tmp_path, monkeypatch, capsys):
    path = tmp_path / "ticket.txt"
    path.write_text("I was charged twice")
    def reply(state):
        return {"type": "choice", "choice": "billing", "confidence": 0.97,
                "probabilities": {"billing": 0.98, "technical": 0.02}}
    post = _install_system_one(tmp_path, monkeypatch, reply)
    code = cli.main(["--system-one", "--file", str(path),
                     "--choice", "billing=payments", "technical", "--", "which team?"])
    assert code == 0
    assert capsys.readouterr().out == "billing\t0.97\n"
    assert post.data['state'] == "I was charged twice"
    assert post.data['questions']['answer']['criteria'] == {"billing": "payments",
                                                            "technical": None}


def test_system_one_model_flag_overrides_config(tmp_path, monkeypatch, capsys):
    def reply(state):
        return {"type": "noul", "noul": 0.5}
    post = _install_system_one(tmp_path, monkeypatch, reply)
    assert cli.main(["--system-one", "--model", "circuit-8b", "-p", "q"]) == 0
    assert post.data['model'] == "circuit-8b"


def test_system_one_flags_target_the_system_one_stack(tmp_path, monkeypatch, capsys):
    def reply(state):
        return {"type": "noul", "noul": 0.5}
    post = _install_system_one(tmp_path, monkeypatch, reply)
    with open(config.config_path()) as f:
        data = json.load(f)
    data['system_one_base_url'] = "https://jev.test/v1"
    with open(config.config_path(), "w") as f:
        json.dump(data, f)
    code = cli.main(["--system-one", "--base-url", "https://flag.test/v1",
                     "--api-key", "sk-flag", "-p", "q"])
    assert code == 0
    assert post.calls[-1]['url'] == "https://flag.test/v1/systemone"
    assert post.calls[-1]['kwargs']['headers']['Authorization'] == "Bearer sk-flag"


def test_system_one_model_must_come_from_config(tmp_path, monkeypatch, capsys):
    def reply(state):
        return {"type": "noul", "noul": 0.5}
    _install_system_one(tmp_path, monkeypatch, reply, model=None)
    with pytest.raises(SystemExit):
        cli.main(["--system-one", "-p", "q"])
    assert "system_one_model" in capsys.readouterr().err


def test_system_one_api_error_is_reported(tmp_path, monkeypatch, capsys):
    def reply(state):
        return {"type": "noul", "noul": 0.5}
    _install_system_one(tmp_path, monkeypatch, reply)

    def failing_post(url, **kwargs):
        return FakeSystemOneResponse({}, status_code=422)
    monkeypatch.setattr(api.requests, "post", failing_post)
    assert cli.main(["--system-one", "-p", "q"]) == 1
    assert "422" in capsys.readouterr().err


def test_system_one_line_by_line_keeps_input_order(tmp_path, monkeypatch, capsys):
    def reply(state):
        scores = {"open('x')": 0.97, "import os": 0.91, "print(1)": 0.03}
        return {"type": "noul", "noul": scores[state]}
    post = _install_system_one(tmp_path, monkeypatch, reply,
                               stdin="open('x')\n\nimport os\nprint(1)\n")
    code = cli.main(["--line-by-line", "--cores", "3", "--system-one",
                     "--", "does this touch the file system?"])
    assert code == 0
    assert capsys.readouterr().out == ("open('x')\t0.97\n"
                                       "import os\t0.91\n"
                                       "print(1)\t0.03\n")
    assert len(post.calls) == 3
    states = []
    for call in post.calls:
        states.append(call['kwargs']['json']['state'])
    assert sorted(states) == ["import os", "open('x')", "print(1)"]


def test_system_one_run_contract():
    def ask(state):
        return state.upper()
    run = cli.SystemOneRun(ask, "abc")
    assert run.text == ""
    assert list(run) == []
    assert run.text == "ABC"
    run.close()
