"""decision: the System One decision call - the SDK's typed-answer verb.

decide(state, questions) is to SystemOneApi what call_llm is to OpenAiApi: the
stateless layer-1 entry a hook, a command or a plain script calls, with the
config defaults (model, endpoint, key) resolved here instead of by every
caller. no loop, no stream, no tools - one POST, typed answers back.

    answers, usage = cai.decide(
        "rm -rf ~/projects",
        {"risky": {"type": "noul",
                   "instructions": "Would this shell command destroy user data?"}})
    if answers["risky"]["noul"] > 0.8:
        ...

the question and answer shapes are the wire ones, documented on
SystemOneApi.system_one."""
from cai import config
from cai.api import SystemOneApi


def default_model():
    """the `system_one_model` config value. there is no fallback: a chat model
    cannot answer a System One request, so a missing key is a ValueError that
    names it."""
    model = config.load_optional("system_one_model")
    if model: return model
    raise ValueError("cai.decide needs a model: set `system_one_model` in "
                     f"{config.config_path()}")


def default_api():
    """a SystemOneApi on the configured base_url and api key - the same
    resolution Agent uses for its OpenAiApi (config file, cai.settings shadow,
    --base-url/--api-key overrides)."""
    cfg = config.load_config()
    return SystemOneApi(cfg.base_url,
                        config.load_api_key(),
                        ssl_verify=config.load_optional("ssl_verify", True))


def decide(state, questions, model=None, api=None):
    """ask `questions` (a name -> wire question map) about `state` (a string,
    or any JSON object/array) and return (answers, usage) as SystemOneApi
    returns them. a model or api the caller does not supply comes from config;
    given both, the call never touches the disk."""
    if model is None: model = default_model()
    if api is None: api = default_api()
    return api.system_one(state, questions, model)
