"""Prompt templates. English only for now; a native version adds TEMPLATES["ru"/"tr"] with the same keys."""
FIELDS = {"literal": {"type": "string"}, "insider": {"type": "string"}, "why": {"type": "string"}}
NAMED_SCHEMA = {"type": "object", "properties": FIELDS, "required": list(FIELDS), "additionalProperties": False}
UNNAMED_SCHEMA = {"type": "object", "required": ["items"], "additionalProperties": False, "properties": {"items": {
    "type": "array", "maxItems": 5, "items": {"type": "object", "required": ["phrase", *FIELDS],
                                              "additionalProperties": False,
                                              "properties": {"phrase": {"type": "string"}, **FIELDS}}}}}

TEMPLATES = {"en": {
    "system": "You are an expert in {language} political media and its cultural and historical background.",
    "intro": ("Coded language is wording that carries a specific meaning for in-group readers but whose surface "
              "translation does not reveal it: euphemisms, political allusions, loaded nicknames, metaphors with "
              "insider meaning, or names and dates that point to a specific person, group or event."),
    "text": "Text (original {language}):\n{context}",
    "no_text": "The expression comes from a {language} opinion article.",
    "named": ('The coded expression is: "{phrase}"{marker}.\nGive the following, in English:\n'
              '- "literal": a word-for-word translation of the expression itself (not a summary of the text)\n'
              '- "insider": what it actually signals to an in-group reader, in one or two sentences\n'
              '- "why": one line on the cultural or political knowledge an outsider would need but lack\n'
              'Answer with one JSON object with the keys "literal", "insider", "why".'),
    "marker": " (marked with [[ ]] in the text)",
    "unnamed": ('Find up to 5 coded expressions in the text. For each, give the expression exactly as it appears '
                'in the text and, in English, "literal", "insider" and "why" as defined: literal = a word-for-word translation of the '
                'expression itself (not a summary of the text); insider = what it signals to an in-group reader, in '
                'one or two sentences; why = one line on the knowledge an outsider would lack.\nAnswer with one JSON object '
                '{{"items": [{{"phrase": ..., "literal": ..., "insider": ..., "why": ...}}]}}.'),
}}

def render(lang_code, *, language, phrase, context, outlet, title, naming):
    """-> (messages, schema). `context` is the text excerpt (marked for named, plain for unnamed) or None."""
    t = TEMPLATES[lang_code]; parts = [t["intro"]]
    meta = ([f"Source: {outlet}"] if outlet else []) + ([f"Title: {title}"] if title else [])
    if meta: parts.append("\n".join(meta))
    parts.append(t["text"].format(language=language, context=context) if context else t["no_text"].format(language=language))
    if naming == "named":
        parts.append(t["named"].format(phrase=phrase, marker=t["marker"] if context else ""))
    else:
        parts.append(t["unnamed"])
    messages = [{"role": "system", "content": t["system"].format(language=language)},
                {"role": "user", "content": "\n\n".join(parts)}]
    return messages, (NAMED_SCHEMA if naming == "named" else UNNAMED_SCHEMA)