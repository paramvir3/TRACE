"""Session-wide isolation of the global ``torch`` default dtype.

``tests/electrostatics/conftest.py`` must call
``torch.set_default_dtype(torch.float64)`` at IMPORT time: e3nn caches generated
``TensorProduct`` code under the dtype in force when a module is first
constructed, and a later ``.double()`` is a silent no-op (SPEC 3 records 2.54e-08
against 1.94e-15).  There is no way to scope an import-time side effect to a
fixture.

pytest imports every ``conftest.py`` during collection, before any test runs, so
that assignment leaks into the whole session.  Running ``pytest tests`` then
failed seven tests that pass when run on their own:

    test_muon_optimizer.py                  (2)
    test_architecture_v2.py                 (2)
    test_stress_finite_difference.py        (1)
    test_batched_graph.py                   (1)   <- named
        test_float64_stress_works_without_a_global_default_dtype
    test_aot_export.py                      (1)

They fail for a real reason, not a spurious one: ``test_muon_update_...`` compares
a reference that takes the *default* dtype against a call pinned to
``ns_dtype=torch.float32``, so under a leaked float64 default the two genuinely
differ, by 6.8e-07 against a 1e-07 tolerance -- exactly float32 resolution.

A suite whose result depends on which subset you run is worse than a failing
one, because it reports success.  So rather than restore the dtype after the
electrostatics package -- pytest defers package-scope teardown, and relying on
directory collection order is fragile anyway -- every test is given the dtype its
own suite expects, before it runs.  Order-independent by construction:
``pytest tests``, ``pytest tests/electrostatics``, ``pytest -k muon`` and
``pytest -p no:randomly`` all agree.
"""

import pytest
import torch

#: torch's own default, which every suite outside ``tests/electrostatics``
#: is written against.  Not read from torch at import time -- by then the
#: electrostatics conftest may already have changed it.
TORCH_NATIVE_DEFAULT_DTYPE = torch.float32


@pytest.fixture(autouse=True)
def _default_dtype_isolation(request):
    """Set the global default dtype this test's suite expects, before it runs."""
    electrostatics = "electrostatics" in request.node.nodeid
    torch.set_default_dtype(
        torch.float64 if electrostatics else TORCH_NATIVE_DEFAULT_DTYPE
    )
    yield


def pytest_configure(config):
    # Registered here as well as in tests/electrostatics/conftest.py, so that
    # slow tests outside that folder (the user-guide run) are known marks.
    config.addinivalue_line("markers", "slow: long-running regression test")
