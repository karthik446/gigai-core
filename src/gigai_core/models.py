"""The one base class of every data record in core."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CoreModel(BaseModel):
    """Immutable, strict and closed.

    ``frozen``: a record never changes after it is built. ``strict``: no coercion (``"1"`` is not an int, a string is
    not a datetime). ``extra="forbid"``: an unknown key in a file is an error, so a typo or a newer schema is never
    silently dropped. ``allow_inf_nan=False``: every float is finite, because NaN is written as ``null`` and then
    fails to read back.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
