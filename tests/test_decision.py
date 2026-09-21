"""cai.decide - the SDK's System One verb: config defaults resolved once, the
wire shapes passed through untouched."""
import json
import os

import pytest

import cai
import cai.api as api
from cai import config
from cai import decision
from cai.api import SystemOneApi


class FakeResponse:
    def __init__(self, body):
        self.status_code = 200
        self.text = ""
        self._body = body

    def json(self):
        return self._body

    def close(self):
        return


class Post:
    """a fake requests.post answering every request with the same body and
    recording the last one."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, url, **kwargs):
        record = {}
        record['url'] = url
        record['kwargs'] = kwargs
        self.calls.append(record)
        body = {}
        body['model'] = "jev-latest"
        body['answers'] = self.answers
        body['usage'] = {"input_tokens": 3, "output_tokens": 1}
        return FakeResponse(body)

    @property
    def data(self):
        return self.calls[-1]['kwargs']['json']


def questions():
    q = {}
    q['risky'] = {"type": "noul", "instructions": "Would this damage user data?"}
    return q


def answers():
    a = {}
    a['risky'] = {"type": "noul", "noul": 0.92}
    return a


def install_post(monkeypatch):
    post = Post(answers())
    monkeypatch.setattr(api.requests, "post", post)
    return post


def write_config(tmp_path, monkeypatch, model="jev-latest"):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    os.makedirs(config.config_dir(), exist_ok=True)
    data = {}
    data['base_url'] = "https://example.test/v1"
    data['model'] = "chat-model"
    if model is not None:
        data['system_one_model'] = model
    with open(config.config_path(), "w") as f:
        json.dump(data, f)
    with open(config.api_key_path(), "w") as f:
        f.write("sk-file\n")


def test_given_api_and_model_never_touch_the_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "nowhere"))
    post = install_post(monkeypatch)
    given = SystemOneApi("https://given.test/v1", "sk-given")
    got, usage = decision.decide("rm -rf /", questions(), model="m", api=given)
    assert got == answers()
    assert usage == {"input_tokens": 3, "output_tokens": 1}
    assert post.calls[-1]['url'] == "https://given.test/v1/systemone"
    assert post.data == {"model": "m", "state": "rm -rf /", "questions": questions()}


def test_model_defaults_to_system_one_model(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    post = install_post(monkeypatch)
    decision.decide("s", questions(), api=SystemOneApi("https://given.test/v1", "k"))
    assert post.data['model'] == "jev-latest"


def test_api_defaults_to_config_endpoint_and_key(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    post = install_post(monkeypatch)
    decision.decide("s", questions(), model="m")
    assert post.calls[-1]['url'] == "https://example.test/v1/systemone"
    assert post.calls[-1]['kwargs']['headers']['Authorization'] == "Bearer sk-file"


def test_default_api_honors_runtime_overrides(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    post = install_post(monkeypatch)
    config.set_override("base_url", "https://override.test/v1")
    config.set_override("api_key", "sk-override")
    decision.decide("s", questions(), model="m")
    assert post.calls[-1]['url'] == "https://override.test/v1/systemone"
    assert post.calls[-1]['kwargs']['headers']['Authorization'] == "Bearer sk-override"


def test_missing_model_is_a_config_error_before_any_request(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, model=None)
    post = install_post(monkeypatch)
    with pytest.raises(ValueError) as info:
        decision.decide("s", questions())
    assert "system_one_model" in str(info.value)
    assert post.calls == []


def test_sdk_exports():
    assert cai.decide is decision.decide
    assert cai.SystemOneApi is SystemOneApi
    assert "decide" in cai.__all__
    assert "SystemOneApi" in cai.__all__
