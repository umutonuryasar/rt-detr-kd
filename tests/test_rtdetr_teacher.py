"""Tests for the lyuwenyu/RT-DETR teacher adapter.

These cover the load-bearing invariants of ``rtdetr_teacher``:

  - The ``_lyuwenyu_import_context`` swaps and restores ``src.*`` in
    ``sys.modules`` cleanly, with no permanent leakage.
  - Our own ``src.*`` modules remain importable AFTER the context exits.
  - The context refuses to enter when the submodule is missing.

Building the actual lyuwenyu RTDETR model is skipped here because it
requires the submodule checkout (a 100+ MB clone) to be present and CI
runs without it. The build path is exercised by ``tools/verify_teacher_kd.py``
on the developer's machine.
"""

import sys
import pytest

from src.models.rtdetr_teacher import _lyuwenyu_import_context, _LYUWENYU_PYTORCH_DIR


def test_our_src_present_before_context():
    """Sanity: our package is in sys.modules at test start."""
    assert "src" in sys.modules
    assert "src.models.rtdetr_teacher" in sys.modules


def test_context_restores_our_src_modules():
    """Entering and exiting the context must restore sys.modules['src.*']
    to the original references — no leftover lyuwenyu entries, no missing
    entries from ours."""
    if not _LYUWENYU_PYTORCH_DIR.is_dir():
        pytest.skip("lyuwenyu submodule not checked out")

    before = {k: v for k, v in sys.modules.items()
              if k == "src" or k.startswith("src.")}
    before_path = sys.path[:]

    with _lyuwenyu_import_context():
        # Inside the context, our src.* should be GONE from sys.modules
        # (we don't actually import anything here — just check the swap).
        assert "src" not in sys.modules or sys.modules["src"] is not before["src"]
        # The lyuwenyu root must be on sys.path.
        assert str(_LYUWENYU_PYTORCH_DIR) in sys.path

    # After exit: our entries are back, lyuwenyu entries are gone.
    after = {k: v for k, v in sys.modules.items()
             if k == "src" or k.startswith("src.")}
    assert after == before, (
        "src.* sys.modules entries changed after context exit:\n"
        f"  added: {set(after) - set(before)}\n"
        f"  removed: {set(before) - set(after)}\n"
    )
    assert sys.path == before_path, "sys.path not restored"


def test_our_imports_still_work_after_context():
    """After the context exits, importing our own modules must still work."""
    if not _LYUWENYU_PYTORCH_DIR.is_dir():
        pytest.skip("lyuwenyu submodule not checked out")

    with _lyuwenyu_import_context():
        pass

    # Fresh import via importlib to bypass any caching trickery.
    import importlib
    mod = importlib.import_module("src.distillation.kd_loss")
    assert hasattr(mod, "KDLoss"), "Our src.distillation.kd_loss broken after context"


def test_context_raises_when_submodule_missing(monkeypatch):
    """If the submodule directory does not exist, entering the context
    must raise a clear error pointing at the fix."""
    import src.models.rtdetr_teacher as rtm
    bogus = rtm._LYUWENYU_PYTORCH_DIR.parent / "does_not_exist"
    monkeypatch.setattr(rtm, "_LYUWENYU_PYTORCH_DIR", bogus)
    with pytest.raises(RuntimeError, match="submodule"):
        with rtm._lyuwenyu_import_context():
            pass
