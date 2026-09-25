"""Regression test for the plan's FE8 finding: a front-end error code with
no matching `err.<CODE>` locale entry falls all the way through i18n.mjs's
en -> ja -> key fallback chain to the raw key string ("err.E_HTTP"), not a
readable message -- this happened for E_HTTP (api.mjs's own code for a
non-JSON HTTP error response) because it isn't one of errors.ALL_CODES,
so nothing enforced its presence in the dictionaries the way the codes
below already are.

Kept as a plain read-and-assert over the shipped JSON files (not a browser
test) since this is a static completeness property of the three locale
files, not something that needs a real page.
"""
import json
from pathlib import Path

import pytest

from server import errors

LOCALES_DIR = Path(__file__).resolve().parents[1] / "static" / "locales"
LANGS = ("ja", "en", "zh")

#: Codes the front end can produce that AREN'T in errors.ALL_CODES (that
#: set is server-emitted codes only) -- api.mjs's own client-side codes.
EXTRA_FRONTEND_CODES = frozenset({"E_HTTP", "NETWORK", "UNKNOWN"})


@pytest.fixture(scope="module")
def dicts():
    return {lang: json.loads((LOCALES_DIR / f"{lang}.json").read_text()) for lang in LANGS}


@pytest.mark.parametrize("lang", LANGS)
def test_every_server_error_code_has_a_translation(dicts, lang):
    missing = {code for code in errors.ALL_CODES if f"err.{code}" not in dicts[lang]}
    assert not missing, f"{lang}.json is missing err.<CODE> for: {sorted(missing)}"


@pytest.mark.parametrize("lang", LANGS)
def test_frontend_only_codes_have_a_translation(dicts, lang):
    missing = {code for code in EXTRA_FRONTEND_CODES if f"err.{code}" not in dicts[lang]}
    assert not missing, f"{lang}.json is missing err.<CODE> for: {sorted(missing)}"


def test_all_three_locale_files_carry_the_same_key_set(dicts):
    key_sets = {lang: set(d.keys()) for lang, d in dicts.items()}
    ja_keys = key_sets["ja"]
    for lang in LANGS:
        missing = ja_keys - key_sets[lang]
        extra = key_sets[lang] - ja_keys
        assert not missing, f"{lang}.json is missing keys present in ja.json: {sorted(missing)}"
        assert not extra, f"{lang}.json has keys not present in ja.json: {sorted(extra)}"
