"""``JsonValue`` for data that really is open-ended, and the closed string sets."""

from __future__ import annotations

from typing import Literal

from pydantic import JsonValue as JsonValue

ParseStatus = Literal["ok", "invalid", "not_json", "not_parsed"]
Outcome = Literal["succeeded", "failed"]
Direction = Literal["two-sided", "greater", "less"]
