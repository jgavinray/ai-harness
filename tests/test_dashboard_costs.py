"""The dashboard's cost table must cover every backend the live config serves.

Invariant: a backend row on /dashboard shows "$–" for input and decode cost
whenever `COST` in static/dashboard.html has no entry for that backend's model
id. A fleet member added to harness.toml without a matching rate line is
therefore invisible in Total Cost and in the Opus-equivalent/ROI lines, which
silently understates both. These tests bind the two files together.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "harness.toml"
DASHBOARD = ROOT / "src" / "harness" / "static" / "dashboard.html"


def _configured_models() -> list[str]:
    """Model ids of the ACTIVE [[backends]] entries (commented-out ones are
    not parsed by tomllib, which is exactly the set the dashboard renders)."""
    cfg = tomllib.loads(CONFIG.read_text())
    return [b["model"] for b in cfg["backends"]]


def _cost_table() -> dict[str, dict[str, float]]:
    """Parse the COST object literal out of the dashboard's inline script."""
    html = DASHBOARD.read_text()
    block = re.search(r"const COST = \{(.*?)\n\};", html, re.S)
    assert block, "COST object literal not found in dashboard.html"
    table: dict[str, dict[str, float]] = {}
    for line in block.group(1).splitlines():
        entry = re.search(r'"([^"]+)":\s*\{(.*?)\}', line)
        if not entry:
            continue
        rates = dict(re.findall(r"(\w+):\s*([0-9.]+)", entry.group(2)))
        table[entry.group(1)] = {k: float(v) for k, v in rates.items()}
    return table


def test_cost_table_covers_every_configured_backend():
    missing = sorted(set(_configured_models()) - set(_cost_table()))
    assert not missing, f"backends with no dashboard cost rate: {missing}"


def test_every_cost_entry_declares_all_four_rates():
    for model, rates in _cost_table().items():
        assert set(rates) == {"in", "out", "in_cached", "out_cached"}, model


def test_nemotron_rates_match_the_owner_quoted_prices():
    rates = _cost_table()["nemotron-3.5-lightning-30b"]
    assert rates == {"in": 0.10, "out": 0.25, "in_cached": 0.05, "out_cached": 0.0}


def test_qwen38_rates_match_the_owner_quoted_prices():
    rates = _cost_table()["qwen3.8-27b"]
    assert rates == {"in": 0.45, "out": 3.20, "in_cached": 0.045, "out_cached": 0.0}


def test_qwen36_rate_retained_for_historical_cost_accounting():
    """qwen27 was re-pointed at qwen3.8-27b 2026-08-14; qwen3.6-27b is no
    longer an active backend but its rate stays so cost figures already
    accrued under that model id remain computable."""
    rates = _cost_table()["qwen3.6-27b"]
    assert rates == {"in": 0.60, "out": 3.60, "in_cached": 0.075, "out_cached": 0.0}
