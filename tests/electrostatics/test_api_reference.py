"""docs/electrostatics_api.md is the generated reference of the public API, current.

Regenerated in memory from the package and compared with the committed page:
a renamed argument, a new public name or a changed docstring summary fails
here until ``python tools/gen_electrostatics_api.py`` is run.
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_the_api_reference_is_generated_and_current():
    spec = importlib.util.spec_from_file_location("gen_api", ROOT / "tools" / "gen_electrostatics_api.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    expected = generator.render()
    current = (ROOT / "docs" / "electrostatics_api.md").read_text()
    assert current == expected, (
        "docs/electrostatics_api.md is out of date: run python tools/gen_electrostatics_api.py"
    )
    import trace_electrostatics

    for name in trace_electrostatics.__all__:
        assert f"### `{name}`" in current, name
