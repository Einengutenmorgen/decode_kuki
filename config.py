"""Settings for the KuKi L4 decoding experiment. Standalone: nothing is imported from the S1-S5 pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("KUKI_DATA", ROOT / "data"))   # kuki_{ru,tr}_aggregate.jsonl
DS_DIR, JUDGE_DIR = ROOT / "datasets", ROOT / "judgments"
RES_DIR = Path(os.environ.get("KUKI_RESULTS", ROOT / "results"))
SEED = 13
PROMPT_VERSION = "v2"   # bump whenever prompts.py changes; results files carry it, so versions never mix
DEV_SIZE = {"ru": 40, "tr": 10}                                # cases per language, drawn phrase-grouped

# Label Studio user id -> annotator letter (dataset description, section 3).
ANNOTATOR_IDS = {"ru": {8: "A", 4: "B", 10: "C"}, "tr": {11: "A", 7: "B", 6: "C"}}
L4_LAYER = "l4_coded"
LANG_NAME = {"ru": "Russian", "tr": "Turkish"}
# RU exports separate paragraphs with single newlines, TR with blank lines.
PARA_SEP = {"ru": "\n", "tr": "\n\n"}
# source -> (name shown in the prompt, orientation pole). VERIFY poles (same mapping as the S1-S5 pipeline).
OUTLETS = {
    "ria_novosti": ("RIA Novosti", "gov_close"), "ng": ("Nezavisimaya Gazeta", "gov_close"),
    "theinsider": ("The Insider", "gov_distant"), "holod": ("Holod", "gov_distant"),
    "sabah": ("Sabah", "gov_close"), "cumhuriyet": ("Cumhuriyet", "gov_distant"),
    "yenicag": ("Yeniçağ", "gov_distant"), "odatv": ("Oda TV", "gov_distant"),
}

# Conditions: one factor at a time around DEFAULT. The "article" context contains the title as stored.
DEFAULT = dict(context="paragraph", title=False, outlet="none", prompt_lang="en", naming="named")
def cond(**kw): return {**DEFAULT, **kw}
CONDITIONS = [
    cond(),                                                     # default
    cond(context="phrase"), cond(context="phrase", title=True), cond(context="sentence"),
    cond(title=True), cond(context="article"),                  # axis 1: context
    cond(outlet="true"), cond(outlet="swapped"),                # axis 2: outlet
    cond(naming="unnamed"), cond(context="article", naming="unnamed"),   # axis 3: detection ablation
]
def condition_id(c):
    return f"ctx-{c['context']}__title-{int(c['title'])}__outlet-{c['outlet']}__{c['prompt_lang']}__{c['naming']}"

# Models served through an OpenAI-compatible vLLM endpoint. extra_body switches thinking off (VERIFY per model).
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
MODELS = {
    "olmo3-7b-instruct": dict(hf_id="allenai/Olmo-3-7B-Instruct", extra_body={}),
    "gemma4-e4b":        dict(hf_id="google/gemma-4-E4B-it", extra_body={}),          # VERIFY thinking switch
    "gemma4-12b":        dict(hf_id="google/gemma-4-12B-it", extra_body={}),          # VERIFY thinking switch
    "gemma4-31b":        dict(hf_id="google/gemma-4-31B-it", extra_body={}),          # VERIFY repo name + switch
    "qwen3.8-27b":       dict(hf_id="Qwen/Qwen3.8-27B", extra_body=NO_THINK),         # VERIFY switch
    "olmo3.1-32b":       dict(hf_id="allenai/Olmo-3.1-32B-Instruct", extra_body={}),  # VERIFY repo name
}