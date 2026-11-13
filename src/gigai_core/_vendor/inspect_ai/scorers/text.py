# Lifted from inspect_ai src/inspect_ai/_util/text.py @ 9567b431 (MIT, Copyright (c) 2024 UK AI Security Institute)
import re
import string


def strip_punctuation(s: str) -> str:
    return s.strip(string.whitespace + string.punctuation)


def strip_numeric_punctuation(s: str) -> str:
    # un-escape LaTeX-escaped currency/formatting symbols (e.g. "\$20" -> "$20").
    # Models frequently escape these inside LaTeX math mode, since a bare "$"
    # would otherwise open/close a math expression. Without this, "\$20" is
    # left as "\20" below (the backslash isn't a stripped character), which
    # fails to parse as a number and is silently rejected as a non-numeric
    # token by callers such as match(numeric=True).
    s = re.sub(r"\\([$,£,€*_])", r"\1", s)

    # strip $, €, £, and ,
    # *,_ to string formatting characters sometimes added by LLMs
    stripped = re.sub(r"[$,£,€,*,_]", "", s)

    # strip . if it's followed by a space, the end of the string,
    # or a non-digit character
    stripped = re.sub(r"\.(?=\s|$|\D)", "", stripped)
    return stripped
