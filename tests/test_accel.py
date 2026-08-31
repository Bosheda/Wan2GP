"""CPU-only tests for shared/accel.py and for the three patched import-time call sites.

Runs anywhere. Imports torch nowhere. Every behavioural case is driven by a fake torch built
in this file, so the suite gives identical results on a CUDA box, on an Intel Arc, and in CI
with no accelerator at all.

    python tests/test_accel.py

TWO PARTS
=========
PART A exercises shared/accel.py directly.

PART B is a STRUCTURAL proof over the patched files. Part A can only show that the helper
behaves; it cannot show that the call sites actually use it. Part B parses shared/attention.py
and wgp.py with ast and asserts that no `torch.cuda.<anything>` call is reachable during
import except the permitted availability probe. That is the property that decides whether
`import wgp` works on a machine without CUDA, and it is the property Part A cannot see.

THE CENTRAL BEHAVIOURAL PROOF
=============================
`ForbiddenCuda` raises on every attribute except `is_available`, so any code path that reaches
`get_device_capability` while a non-CUDA backend is selected explodes rather than quietly
succeeding.

NEGATIVE CONTROLS
=================
Section 9 trips each guard on purpose. A tripwire nobody has watched fire is indistinguishable
from one that cannot fire, and a green suite that cannot go red proves nothing.
"""
import ast
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "shared"))

import accel as AD  # noqa: E402

PASS = []
FAIL = []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print("%s  %s" % ("PASS" if cond else "FAIL", label))


def raises(label, exc_type, fn):
    try:
        fn()
    except exc_type:
        check(label, True)
        return
    except Exception as e:
        check("%s (wrong exception %s: %s)" % (label, type(e).__name__, e), False)
        return
    check("%s (nothing raised)" % label, False)


class Ns(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ForbiddenCuda(object):
    """torch.cuda stub permitting ONLY is_available, detonating on anything else."""

    def __init__(self, available=False, record=None):
        object.__setattr__(self, "_available", available)
        object.__setattr__(self, "record", record if record is not None else [])

    def is_available(self):
        return self._available

    def __getattr__(self, name):
        self.record.append(name)
        raise AssertionError(
            "FORBIDDEN torch.cuda access: %r. On a non-CUDA backend this is the call that "
            "raises 'Torch not compiled with CUDA enabled'." % name)


def cuda_torch(cap=(8, 6), bf16=None, version="2.5.1+cu121"):
    c = Ns(is_available=lambda: True, get_device_capability=lambda device=None: cap)
    if bf16 is not None:
        c.is_bf16_supported = lambda: bf16
    return Ns(cuda=c, __version__=version)


def xpu_torch(bf16=None, record=None):
    x = Ns(is_available=lambda: True)
    if bf16 is not None:
        x.is_bf16_supported = lambda: bf16
    return Ns(cuda=ForbiddenCuda(record=record), xpu=x, __version__="2.13.0+xpu")


def mps_torch(bf16=None, record=None):
    m = Ns(is_available=lambda: True)
    if bf16 is not None:
        m.is_bf16_supported = lambda: bf16
    return Ns(cuda=ForbiddenCuda(record=record), backends=Ns(mps=m), __version__="2.5.1")


def bare_torch(record=None):
    return Ns(cuda=ForbiddenCuda(record=record), __version__="2.5.1+cpu")


# =======================================================================================
# PART A
# =======================================================================================
print("\n--- 1. CUDA available ---")
t = cuda_torch(cap=(8, 6))
check("backend is cuda", AD.detect_backend(t) == AD.BACKEND_CUDA)
check("capability is the real (8, 6)", AD.cuda_capability(t) == (8, 6))
check("bf16 True via the preserved major>=8 rule", AD.bfloat16_supported(t) is True)
check("no conflict flagged", AD.describe(t)["conflicting_signals"] is False)
check("sm_75 gives bf16 False", AD.bfloat16_supported(cuda_torch(cap=(7, 5))) is False)
check("native is_bf16_supported beats the major>=8 rule",
      AD.bfloat16_supported(cuda_torch(cap=(7, 5), bf16=True)) is True)

print("\n--- 2. XPU available ---")
rec = []
t = xpu_torch(bf16=True, record=rec)
check("backend is xpu", AD.detect_backend(t) == AD.BACKEND_XPU)
check("cuda_capability is None, NOT a fabricated (0,0)", AD.cuda_capability(t) is None)
check("bf16 True from the xpu probe", AD.bfloat16_supported(t) is True)
check("zero forbidden torch.cuda attributes reached", rec == [])
rec = []
t = xpu_torch(record=rec)
check("no xpu probe means the conservative default False",
      AD.bfloat16_supported(t, warn=False) is False)
check("explicit default True is honoured",
      AD.bfloat16_supported(t, default=True, warn=False) is True)
check("still zero forbidden torch.cuda attributes", rec == [])

print("\n--- 3. MPS available ---")
rec = []
t = mps_torch(record=rec)
check("backend is mps", AD.detect_backend(t) == AD.BACKEND_MPS)
check("MPS does NOT fabricate a CUDA capability", AD.cuda_capability(t) is None)
check("zero forbidden torch.cuda attributes reached", rec == [])
check("mps probe honoured when present",
      AD.bfloat16_supported(mps_torch(bf16=False), warn=False) is False)

print("\n--- 4. No accelerator ---")
rec = []
t = bare_torch(record=rec)
check("detect_backend is None", AD.detect_backend(t) is None)
check("available_backends is empty", AD.available_backends(t) == [])
raises("require_backend raises NoAcceleratorError", AD.NoAcceleratorError,
       lambda: AD.require_backend(t))
check("cuda_capability is None", AD.cuda_capability(t) is None)
check("zero forbidden torch.cuda attributes reached", rec == [])

print("\n--- 5. Conflicting backend signals ---")
t = Ns(cuda=Ns(is_available=lambda: True, get_device_capability=lambda device=None: (8, 9)),
       xpu=Ns(is_available=lambda: True, is_bf16_supported=lambda: True),
       backends=Ns(mps=Ns(is_available=lambda: True)), __version__="frankentorch")
check("all three reported", AD.available_backends(t) == ["cuda", "xpu", "mps"])
check("cuda wins, preserving existing behaviour", AD.detect_backend(t) == AD.BACKEND_CUDA)
check("the conflict is FLAGGED, not hidden", AD.describe(t)["conflicting_signals"] is True)
t2 = Ns(cuda=ForbiddenCuda(), xpu=Ns(is_available=lambda: True),
        backends=Ns(mps=Ns(is_available=lambda: True)), __version__="x")
check("xpu beats mps when cuda is absent", AD.detect_backend(t2) == AD.BACKEND_XPU)

print("\n--- 6. Malformed capability, and malformed PROBE values ---")
for bad in [None, "8.6", (8,), (8, 6, 1), [], (8.0, 6.0), (True, False), {"major": 8}]:
    raises("capability %r raises MalformedCapabilityError" % (bad,),
           AD.MalformedCapabilityError,
           lambda b=bad: AD.cuda_capability(cuda_torch(cap=b)))
check("a valid list [8, 6] normalises to a tuple",
      AD.cuda_capability(cuda_torch(cap=[8, 6])) == (8, 6))
check("describe() survives a malformed capability",
      AD.describe(cuda_torch(cap="nonsense"))["cuda_capability"] is None)

# This is the defect found in the review of the earlier standalone module: bool("false") is
# True, so coercing a probe return would turn a malformed answer into a confident yes.
for junk in ["false", "False", "0", "no", [], {}, 2, -1, 0.0, object()]:
    t = Ns(cuda=ForbiddenCuda(), xpu=Ns(is_available=lambda v=junk: v), __version__="x")
    check("probe returning %r is NOT accepted as available" % (junk,),
          AD.detect_backend(t) is None)
check("a real True is accepted",
      AD.detect_backend(Ns(cuda=ForbiddenCuda(), xpu=Ns(is_available=lambda: True))) == "xpu")
check("int 1 is accepted as True", AD._strict_bool(1) is True)
check("int 0 is accepted as False", AD._strict_bool(0) is False)
check("string 'false' maps to UNKNOWN, not False", AD._strict_bool("false") is None)

print("\n--- 7. PROOF: non-CUDA paths never touch torch.cuda beyond is_available ---")
for label, factory in (("xpu", xpu_torch), ("mps", mps_torch), ("none", bare_torch)):
    rec = []
    t = factory(record=rec)
    AD.available_backends(t)
    AD.detect_backend(t)
    AD.cuda_capability(t)
    AD.bfloat16_supported(t, warn=False)
    AD.describe(t)
    check("%s: all public functions run, forbidden attrs seen = %r" % (label, rec), rec == [])

print("\n--- 8. PROOF: existing CUDA behaviour is unchanged ---")
for cap in [(6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]:
    t = cuda_torch(cap=cap)
    check("cap %r capability identical to a direct query" % (cap,), AD.cuda_capability(t) == cap)
    check("cap %r bf16 matches legacy major>=8 (%s)" % (cap, cap[0] >= 8),
          AD.bfloat16_supported(t) == (cap[0] >= 8))

print("\n--- 9. NEGATIVE CONTROLS ---")
rec = []
fc = ForbiddenCuda(record=rec)
raises("ForbiddenCuda detonates on get_device_capability", AssertionError,
       lambda: fc.get_device_capability(None))
check("it recorded the attribute name", rec == ["get_device_capability"])
check("it still permits is_available", fc.is_available() is False)

_real = AD._probe


def _leaky(obj, name):
    """Simulates the bug being prevented: probing capability before selecting a backend.

    Note the bare try rather than a hasattr guard. `hasattr` swallows only AttributeError, so
    on ForbiddenCuda (whose __getattr__ raises AssertionError) a hasattr check propagates and
    the control never runs. That is itself a small instance of the lesson these tests are
    about: a guard that only handles the exception you expected is not a guard.
    """
    if name == "is_available":
        try:
            obj.get_device_capability(None)
        except Exception:
            pass
    return _real(obj, name)


AD._probe = _leaky
try:
    rec = []
    AD.detect_backend(xpu_torch(bf16=True, record=rec))
    check("a leaky probe IS caught by the section-7 assertion (saw %r)" % rec,
          rec == ["get_device_capability"])
finally:
    AD._probe = _real
rec = []
AD.detect_backend(xpu_torch(bf16=True, record=rec))
check("restored: clean again after the negative control", rec == [])


class Exploding(object):
    def is_available(self):
        raise RuntimeError("driver on fire")


check("an exploding probe is treated as absent, not fatal",
      AD.detect_backend(Ns(cuda=ForbiddenCuda(), xpu=Exploding())) is None)

# =======================================================================================
# PART B: structural proof over the patched call sites
# =======================================================================================
print("\n--- 10. STRUCTURAL: no unguarded torch.cuda reachable at import ---")

PERMITTED = {"is_available"}


def _cuda_attr_calls(tree):
    """Every `torch.cuda.<attr>` access in the tree, as (attr, lineno)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) \
                and node.value.attr == "cuda" and isinstance(node.value.value, ast.Name) \
                and node.value.value.id == "torch":
            out.append((node.attr, node.lineno))
    return out


def _module_level_nodes(tree):
    """Top-level statements only, so function bodies are excluded."""
    keep = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        keep.append(stmt)
    return keep


for rel in ("shared/attention.py", "wgp.py"):
    path = os.path.join(_ROOT, rel)
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    offenders = []
    for stmt in _module_level_nodes(tree):
        for attr, lineno in _cuda_attr_calls(stmt):
            if attr not in PERMITTED:
                offenders.append("%s:%d torch.cuda.%s" % (rel, lineno, attr))
    check("%s: no forbidden torch.cuda attribute at module level (found %r)"
          % (rel, offenders), offenders == [])

# get_supported_attention_modes is called at module level BY wgp.py, so its body counts as
# import-reachable even though it is a function. Assert the bare capability call is gone.
_att = io.open(os.path.join(_ROOT, "shared/attention.py"), encoding="utf-8").read()
_att_tree = ast.parse(_att)
_gsam = [n for n in ast.walk(_att_tree)
         if isinstance(n, ast.FunctionDef) and n.name == "get_supported_attention_modes"]
check("get_supported_attention_modes was found in the source", len(_gsam) == 1)
if _gsam:
    bad = [a for a, _ in _cuda_attr_calls(_gsam[0]) if a not in PERMITTED]
    check("get_supported_attention_modes makes no forbidden torch.cuda call (found %r)" % bad,
          bad == [])

# Negative control for Part B: the scanner must be able to SEE an offender.
_probe_tree = ast.parse("import torch\nmajor, minor = torch.cuda.get_device_capability(None)\n")
_seen = [a for stmt in _module_level_nodes(_probe_tree) for a, _ in _cuda_attr_calls(stmt)
         if a not in PERMITTED]
check("Part B scanner detects a deliberately planted offender (saw %r)" % _seen,
      _seen == ["get_device_capability"])

print("\n" + "=" * 78)
print("PASS %d   FAIL %d" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAILED: %s" % f)
print("=" * 78)
sys.exit(1 if FAIL else 0)
