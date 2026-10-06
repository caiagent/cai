"""decision: the System One decision call - the SDK's typed-answer verb.

decide(state, questions) is to SystemOneApi what call_llm is to OpenAiApi: the
stateless layer-1 entry a hook, a command or a plain script calls, with the
config defaults (model, endpoint, key - its own stack, see
default_api) resolved here instead of by every
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


def default_api(base_url=None, api_key=None):
    """a SystemOneApi on the System One stack: `system_one_base_url` and the
    system_one_api_key file, each falling back to the chat `base_url` / api_key
    (with their cai.settings shadow and --base-url/--api-key overrides). a
    base_url or api_key the caller passes wins over both."""
    if base_url is None: base_url = config.load_optional("system_one_base_url")
    if base_url is None: base_url = config.load_config().base_url
    if api_key is None: api_key = config.load_system_one_api_key()
    return SystemOneApi(base_url,
                        api_key,
                        ssl_verify=config.load_optional("ssl_verify", True))


def decide(state, questions, model=None, api=None):
    """ask `questions` (a name -> wire question map) about `state` (a string,
    or any JSON object/array) and return (answers, usage) as SystemOneApi
    returns them. a model or api the caller does not supply comes from config;
    given both, the call never touches the disk."""
    if model is None: model = default_model()
    if api is None: api = default_api()
    return api.system_one(state, questions, model)
