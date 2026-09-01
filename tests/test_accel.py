"""CPU-only tests for shared/accel.py and for the three patched import-time call sites.

Runs anywhere. Imports torch nowhere. Every behavioural case is driven by a fake torch built
in this file, so the suite gives identical results on a CUDA box, on an Intel Arc, and in CI
with no accelerator at all.

    python tests/test_accel.py

TWO PARTS
=========
PART A exercises shared/accel.py directly.

PART B is a STRUCTURAL check over the patched files. Part A can only show that the helper
behaves; it cannot show that the call sites actually use it. Part B parses shared/attention.py
and wgp.py with ast and asserts that the three identified import-time CUDA capability
accesses are gone.

WHAT PART B DOES NOT ESTABLISH, stated because the distinction matters
=====================================================================
It clears THE THREE IDENTIFIED WALLS. It does NOT construct the complete import-time call
graph, so it cannot prove that no forbidden CUDA access is transitively reachable anywhere
during `import wgp`. A module imported four levels down could still hold one. Whether the
real `import wgp` succeeds on Intel XPU hardware is Phase 2 acceptance under a healthy GPU
lease and is NOT claimed here.

PART C covers the startup gate: the one place allowed to stop the program, and the refusals
it must produce.

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


def expect(label, fn, pred):
    """Run fn() and check pred(result), turning an UNEXPECTED exception into a FAIL.

    Without this, one unexpected raise aborts the process and every later section silently
    disappears. Measured on 2026-08-31: against the previous selection algorithm this suite
    died on the first case of section 12b after 98 passes, so the number of requirement
    violations could not be counted at all. A harness that stops at the first surprise reports
    "no failures" for everything it never reached.
    """
    try:
        result = fn()
    except Exception as e:
        check("%s (unexpected %s: %s)" % (label, type(e).__name__, e), False)
        return None
    check(label, pred(result))
    return result


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
    check("%s: no DIRECT forbidden torch.cuda attribute at module level (found %r). This "
          "clears the identified walls only; it is not a transitive import-graph proof."
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

# =======================================================================================
# PART C: the startup gate
# =======================================================================================
print("\n--- 11. PROBE STATUS is diagnosable, not collapsed ---")
rep = AD.probe_report(bare_torch())
check("absent xpu reports 'absent'", rep["xpu"][0] == AD.STATUS_ABSENT)
check("cuda that honestly says no reports 'unavailable'",
      rep["cuda"][0] == AD.STATUS_UNAVAILABLE)

rep = AD.probe_report(Ns(cuda=ForbiddenCuda(), xpu=Exploding()))
check("a raising probe reports 'exception', NOT absent",
      rep["xpu"][0] == AD.STATUS_EXCEPTION)
check("the exception detail names the real error",
      "driver on fire" in rep["xpu"][1] and "RuntimeError" in rep["xpu"][1])

rep = AD.probe_report(Ns(cuda=ForbiddenCuda(), xpu=Ns(is_available=lambda: "yes")))
check("a non-boolean probe reports 'malformed', NOT absent",
      rep["xpu"][0] == AD.STATUS_MALFORMED)
check("the malformed detail shows the offending value", "'yes'" in rep["xpu"][1])

check("available reports 'available'",
      AD.probe_report(xpu_torch(bf16=True))["xpu"][0] == AD.STATUS_AVAILABLE)
check("all five statuses are distinct",
      len({AD.STATUS_ABSENT, AD.STATUS_UNAVAILABLE, AD.STATUS_MALFORMED,
           AD.STATUS_EXCEPTION, AD.STATUS_AVAILABLE}) == 5)

print("\n--- 12. STARTUP REFUSES: the loud failure the reviewer required ---")
raises("no accelerator: startup raises NoAcceleratorError", AD.NoAcceleratorError,
       lambda: AD.startup_accelerator(bare_torch()))

t = Ns(cuda=ForbiddenCuda(), xpu=Exploding(), __version__="x")
raises("exploding xpu probe: startup raises AcceleratorProbeError", AD.AcceleratorProbeError,
       lambda: AD.startup_accelerator(t))
try:
    AD.startup_accelerator(t)
except AD.AcceleratorProbeError as e:
    check("the error names WHICH probe failed", e.backend == "xpu")
    check("the error carries the status", e.status == AD.STATUS_EXCEPTION)
    check("'driver on fire' is NOT reported as 'no accelerator found'",
          "driver on fire" in str(e) and "no supported accelerator" not in str(e))

t = Ns(cuda=ForbiddenCuda(), xpu=Ns(is_available=lambda: "yes"), __version__="x")
raises("malformed xpu availability: startup raises AcceleratorProbeError",
       AD.AcceleratorProbeError, lambda: AD.startup_accelerator(t))
try:
    AD.startup_accelerator(t)
except AD.AcceleratorProbeError as e:
    check("malformed error names the probe and status",
          e.backend == "xpu" and e.status == AD.STATUS_MALFORMED)

# A fault must not be REPORTED AS absence. Narrowed deliberately: an earlier version of this
# assertion implied every backend is scanned for faults before anything is selected, which is
# the defect fixed in 12b. See .planning/root-cause/test_accel.py.md.
t = Ns(cuda=ForbiddenCuda(), xpu=Exploding(), backends=Ns(mps=Ns(is_available=lambda: False)))
try:
    AD.startup_accelerator(t)
    check("a broken probe is not reported as an absent accelerator", False)
except AD.AcceleratorProbeError:
    check("a broken probe is not reported as an absent accelerator", True)
except AD.NoAcceleratorError:
    check("a broken probe is not reported as absent (got NoAcceleratorError)", False)

print("\n--- 12b. PRECEDENCE WALK: a fault only matters if it OUTRANKS the selection ---")
# Transcribed from the reviewer's required algorithm BEFORE the module was changed. Three of
# these failed against the previous implementation, which is the point: a new test that cannot
# fail against current code is not testing the requirement.
#
# The defect: require_backend() scanned every backend for faults before selecting anything, so
# a healthy CUDA box shipping a broken torch.xpu was refused startup by a backend it would
# never have used.


def _mps(available=True, bf16=None):
    m = Ns(is_available=lambda: available)
    if bf16 is not None:
        m.is_bf16_supported = lambda: bf16
    return m


def _healthy_cuda(cap=(8, 6)):
    return Ns(is_available=lambda: True, get_device_capability=lambda device=None: cap)


# Positive cases use expect(), so a refusal from a regressed implementation is recorded as a
# FAIL instead of aborting the run. See the expect() docstring.

# 1. Healthy CUDA plus exploding XPU: CUDA must succeed.
t = Ns(cuda=_healthy_cuda(), xpu=Exploding(), __version__="x")
cfg = expect("healthy CUDA + exploding XPU: CUDA selected, not refused",
             lambda: AD.startup_accelerator(t, warn=False), lambda c: c["backend"] == "cuda")
# DELETED, not softened: an earlier assertion here required the XPU fault to be "preserved as
# evidence", which can only pass if the XPU probe RAN. It certified the eager-probing defect
# the walk is supposed to prevent. See .planning/root-cause/accel.py.md.
check("  the unreached XPU is reported not_probed, NOT unavailable",
      bool(cfg) and cfg["probe_report"]["xpu"][0] == AD.STATUS_NOT_PROBED)
check("  not_probed is a distinct status from unavailable",
      AD.STATUS_NOT_PROBED != AD.STATUS_UNAVAILABLE)
check("  a backend we never looked at is NOT counted as a conflict",
      bool(cfg) and cfg["conflicting_signals"] is False)
check("  and the report says the conflict scan did not run",
      bool(cfg) and cfg["conflict_scanned"] is False)

# 2. Healthy CUDA plus malformed MPS: CUDA must succeed.
t = Ns(cuda=_healthy_cuda(), backends=Ns(mps=Ns(is_available=lambda: "sure")), __version__="x")
cfg = expect("healthy CUDA + malformed MPS: CUDA selected, not refused",
             lambda: AD.startup_accelerator(t, warn=False), lambda c: c["backend"] == "cuda")
check("  the unreached MPS is reported not_probed",
      bool(cfg) and cfg["probe_report"]["mps"][0] == AD.STATUS_NOT_PROBED)

# 3. CUDA unavailable, exploding XPU, healthy MPS: refuse on the XPU fault. Falling through to
#    MPS would silently demote the machine on the strength of a broken probe.
t = Ns(cuda=ForbiddenCuda(), xpu=Exploding(), backends=Ns(mps=_mps(True)), __version__="x")
raises("CUDA off + exploding XPU + healthy MPS: refuses on the XPU fault",
       AD.AcceleratorProbeError, lambda: AD.startup_accelerator(t, warn=False))
try:
    AD.startup_accelerator(t, warn=False)
except AD.AcceleratorProbeError as e:
    check("  and it blames XPU, not MPS", e.backend == "xpu")

# 4. CUDA unavailable, healthy XPU, exploding MPS: XPU must succeed.
t = Ns(cuda=ForbiddenCuda(), xpu=Ns(is_available=lambda: True, is_bf16_supported=lambda: True),
       backends=Ns(mps=Exploding()), __version__="x")
cfg = expect("CUDA off + healthy XPU + exploding MPS: XPU selected",
             lambda: AD.startup_accelerator(t, warn=False), lambda c: c["backend"] == "xpu")
check("  the exploding MPS was never reached, so it reports not_probed",
      bool(cfg) and cfg["probe_report"]["mps"][0] == AD.STATUS_NOT_PROBED)
check("  and CUDA, which WAS probed, reports its real unavailable status",
      bool(cfg) and cfg["probe_report"]["cuda"][0] == AD.STATUS_UNAVAILABLE)

# 5. CUDA exception plus healthy XPU: refuse on the CUDA fault. A higher-priority fault must
#    never be bypassed in favour of a lower-priority backend.
t = Ns(cuda=Exploding(), xpu=Ns(is_available=lambda: True), __version__="x")
raises("exploding CUDA + healthy XPU: refuses on the CUDA fault", AD.AcceleratorProbeError,
       lambda: AD.startup_accelerator(t, warn=False))
try:
    AD.startup_accelerator(t, warn=False)
except AD.AcceleratorProbeError as e:
    check("  and it blames CUDA, the higher-priority backend", e.backend == "cuda")

# 6. Multiple cleanly available backends, strict conflict off and on.
t = Ns(cuda=_healthy_cuda(cap=(8, 9)),
       xpu=Ns(is_available=lambda: True, is_bf16_supported=lambda: True),
       backends=Ns(mps=_mps(True)), __version__="x")
cfg = AD.startup_accelerator(t, warn=False)
check("three cleanly available, LAZY: CUDA wins and the others are never probed",
      cfg["backend"] == "cuda" and cfg["available_backends"] == ["cuda"])
check("  conflicting_signals is False because we did not look, and conflict_scanned says so",
      cfg["conflicting_signals"] is False and cfg["conflict_scanned"] is False)
cfg = AD.startup_accelerator(t, strict_conflict=False, warn=False)
check("  the others report not_probed rather than a status nobody measured",
      cfg["probe_report"]["xpu"][0] == AD.STATUS_NOT_PROBED
      and cfg["probe_report"]["mps"][0] == AD.STATUS_NOT_PROBED)
raises("three cleanly available + strict_conflict (opt-in FULL SCAN): refuses",
       AD.ConflictingBackendsError,
       lambda: AD.startup_accelerator(t, strict_conflict=True, warn=False))

# A faulty backend must not manufacture a CONFLICT refusal. In the lazy path it is never even
# probed, so it cannot.
t = Ns(cuda=_healthy_cuda(), xpu=Exploding(), __version__="x")
expect("lazy path: a broken lower-priority backend cannot cause any refusal",
       lambda: AD.startup_accelerator(t, warn=False), lambda c: c["backend"] == "cuda")

# SUPERSEDED ASSERTION, replaced rather than kept. Round 3 asserted that strict mode still
# selects CUDA here, which encoded the old behaviour where strict delegated to the early-
# stopping walk and a scanned fault was captured but never raised. Under the corrected
# semantics a fault ANYWHERE in a strict full scan invalidates the scan: an unknown backend
# could be a second available one, so "no conflict" is not establishable. Strict must refuse,
# and it must refuse with a PROBE error, not a conflict error.
raises("strict path: the same broken backend now raises, because the scan cannot be trusted",
       AD.AcceleratorProbeError,
       lambda: AD.startup_accelerator(t, strict_conflict=True, warn=False))

print("\n--- 13. STARTUP ACCEPTS what it should, and stays quiet about CUDA ---")
cfg = AD.startup_accelerator(cuda_torch(cap=(8, 6)), warn=False)
check("cuda startup selects cuda", cfg["backend"] == "cuda")
check("cuda startup reports the real capability", cfg["cuda_capability"] == (8, 6))
check("cuda startup bf16 matches legacy major>=8", cfg["bfloat16_supported"] is True)
check("cuda startup carries the probe report", "cuda" in cfg["probe_report"])

cfg = AD.startup_accelerator(xpu_torch(bf16=True), warn=False)
check("xpu startup selects xpu", cfg["backend"] == "xpu")
check("xpu startup capability is None, not fabricated", cfg["cuda_capability"] is None)
check("xpu startup bf16 from the xpu probe", cfg["bfloat16_supported"] is True)

conflict = Ns(cuda=Ns(is_available=lambda: True, get_device_capability=lambda device=None: (8, 9)),
              xpu=Ns(is_available=lambda: True, is_bf16_supported=lambda: True),
              __version__="x")
cfg = AD.startup_accelerator(conflict, warn=False)
check("default path does NOT refuse and does NOT scan for conflicts",
      cfg["backend"] == "cuda" and cfg["conflict_scanned"] is False)
raises("conflicting signals DO refuse under strict_conflict=True (opt-in full scan)",
       AD.ConflictingBackendsError,
       lambda: AD.startup_accelerator(conflict, strict_conflict=True, warn=False))

print("\n--- 14. NEGATIVE CONTROL: the startup refusal is observed to fire ---")
# Without this, sections 12 and 13 could both pass on a gate that never actually refuses.
_fired = {"no_accel": False, "probe": False, "conflict": False}
try:
    AD.startup_accelerator(bare_torch())
except AD.NoAcceleratorError:
    _fired["no_accel"] = True
try:
    AD.startup_accelerator(Ns(cuda=ForbiddenCuda(), xpu=Exploding()))
except AD.AcceleratorProbeError:
    _fired["probe"] = True
try:
    AD.startup_accelerator(conflict, strict_conflict=True, warn=False)
except AD.ConflictingBackendsError:
    _fired["conflict"] = True
check("all three refusal branches were observed firing: %r" % _fired, all(_fired.values()))
check("and the gate does NOT refuse a healthy machine",
      AD.startup_accelerator(cuda_torch(), warn=False)["backend"] == "cuda")

print("\n--- 15. STRUCTURAL: the loud failure is actually WIRED IN ---")
# Blocker 1 was that require_backend existed but no site called it. Assert the wiring, not
# the intention, because an uncalled gate is exactly what shipped last time.
_wgp = io.open(os.path.join(_ROOT, "wgp.py"), encoding="utf-8").read()
_wgp_tree = ast.parse(_wgp)
_imported = set()
for node in ast.walk(_wgp_tree):
    if isinstance(node, ast.ImportFrom) and node.module == "shared.accel":
        for a in node.names:
            _imported.add(a.asname or a.name)
check("wgp.py imports the startup gate from shared.accel",
      any(n in _imported for n in ("_accel_startup", "startup_accelerator")))
_called = [n for n in ast.walk(_wgp_tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
           and n.func.id in ("_accel_startup", "startup_accelerator")]
check("wgp.py actually CALLS it (%d call site(s))" % len(_called), len(_called) >= 1)

# And the discovery layer must stay non-raising: attention.py must NOT call the startup gate,
# or every CPU-only utility importing a model module would suddenly require a GPU.
_att_src = io.open(os.path.join(_ROOT, "shared/attention.py"), encoding="utf-8").read()
for forbidden in ("startup_accelerator", "require_backend"):
    check("shared/attention.py does NOT call %s (CPU-only imports stay GPU-free)" % forbidden,
          forbidden not in _att_src)

print("\n--- 16. CALL RECORDING: laziness is a claim about SIDE EFFECTS ---")
# Every assertion above this point inspects RETURN VALUES, and return values cannot tell
# "we stopped" apart from "we probed everything and then filtered". An earlier implementation
# returned the correct backend while still invoking every probe. These assertions watch the
# invocations instead. See .planning/root-cause/accel.py.md.


class Recorder(object):
    """A torch fake that logs every is_available invocation by backend name."""

    def __init__(self, cuda=None, xpu=None, mps=None, cap=(8, 6)):
        self.calls = []
        self.__version__ = "recorder"
        self.cuda = self._mk("cuda", cuda, cap)
        if xpu is not None:
            self.xpu = self._mk("xpu", xpu, None)
        if mps is not None:
            self.backends = Ns(mps=self._mk("mps", mps, None))

    def _mk(self, name, behaviour, cap):
        calls = self.calls

        def is_available():
            calls.append(name)
            if behaviour == "boom":
                raise RuntimeError("driver on fire")
            if behaviour == "junk":
                return "sure"
            return behaviour is True

        obj = Ns(is_available=is_available)
        if cap is not None:
            obj.get_device_capability = lambda device=None: cap
        if behaviour is True and name != "cuda":
            obj.is_bf16_supported = lambda: True
        return obj


def _calls_for(rec, fn):
    try:
        fn()
    except Exception:
        pass
    return list(rec.calls)


# 1. Healthy CUDA never invokes XPU or MPS availability.
r = Recorder(cuda=True, xpu=True, mps=True)
seq = _calls_for(r, lambda: AD.select_backend(r))
check("healthy CUDA: probe sequence is exactly ['cuda'] (saw %r)" % seq, seq == ["cuda"])
check("  XPU availability never invoked", "xpu" not in seq)
check("  MPS availability never invoked", "mps" not in seq)

# The reviewer measured ['cuda','xpu','cuda','cuda','xpu'] through startup_accelerator, so
# assert the whole entrypoint, not just the walk.
r = Recorder(cuda=True, xpu="boom", mps=True)
seq = _calls_for(r, lambda: AD.startup_accelerator(r, warn=False))
check("healthy CUDA + exploding XPU through startup: xpu NEVER invoked (saw %r)" % seq,
      "xpu" not in seq and "mps" not in seq)

# 2. Unavailable CUDA then healthy XPU never invokes MPS.
r = Recorder(cuda=False, xpu=True, mps=True)
seq = _calls_for(r, lambda: AD.select_backend(r))
check("CUDA off + healthy XPU: sequence is ['cuda','xpu'] (saw %r)" % seq,
      seq == ["cuda", "xpu"])
check("  MPS availability never invoked", "mps" not in seq)

# 3. Unavailable CUDA then exploding XPU raises and never invokes MPS.
r = Recorder(cuda=False, xpu="boom", mps=True)
seq = _calls_for(r, lambda: AD.select_backend(r))
check("CUDA off + exploding XPU: sequence is ['cuda','xpu'] (saw %r)" % seq,
      seq == ["cuda", "xpu"])
check("  MPS availability never invoked after the XPU fault", "mps" not in seq)
raises("  and it still raises", AD.AcceleratorProbeError, lambda: AD.select_backend(r))

# 4. Exploding CUDA raises and never invokes XPU or MPS.
r = Recorder(cuda="boom", xpu=True, mps=True)
seq = _calls_for(r, lambda: AD.select_backend(r))
check("exploding CUDA: sequence is exactly ['cuda'] (saw %r)" % seq, seq == ["cuda"])
check("  neither XPU nor MPS invoked", "xpu" not in seq and "mps" not in seq)

# The opt-in strict scan is ALLOWED to be eager, and must be, since the question it answers is
# about backends the walk would never reach. Asserted so the trade is explicit, not accidental.
r = Recorder(cuda=True, xpu=True, mps=True)
seq = _calls_for(r, lambda: AD.startup_accelerator(r, strict_conflict=True, warn=False))
check("strict_conflict DOES scan every backend, by design (saw %r)" % seq,
      set(seq) == {"cuda", "xpu", "mps"})

print("\n--- 16b. STRICT MODE: a scanned fault invalidates the whole scan ---")
# Strict mode used to delegate to select_backend(), which stops interpreting at the first
# available backend. So a scanned XPU returning `exception` was captured in the report and
# never raised, while conflict_scanned=True announced the conflict question had been asked AND
# answered. It had been asked and not answered: an unknown backend could be a second available
# one, so "no conflict" is not establishable while any status is unknown.

# a. Non-strict, healthy CUDA + exploding XPU: succeeds WITHOUT invoking XPU.
r = Recorder(cuda=True, xpu="boom", mps=True)
cfg = expect("non-strict CUDA + exploding XPU: succeeds",
             lambda: AD.startup_accelerator(r, warn=False), lambda c: c["backend"] == "cuda")
check("  and XPU was never invoked (saw %r)" % r.calls, "xpu" not in r.calls)
check("  conflict_scanned is False: we did not ask", bool(cfg) and cfg["conflict_scanned"] is False)

# b. Strict, healthy CUDA + exploding XPU: DOES invoke XPU and DOES raise.
r = Recorder(cuda=True, xpu="boom", mps=True)
raises("strict CUDA + exploding XPU: raises AcceleratorProbeError", AD.AcceleratorProbeError,
       lambda: AD.startup_accelerator(r, strict_conflict=True, warn=False))
check("  and the XPU probe WAS invoked, as strict mode requires (saw %r)" % r.calls,
      "xpu" in r.calls)
r2 = Recorder(cuda=True, xpu="boom", mps=True)
try:
    AD.startup_accelerator(r2, strict_conflict=True, warn=False)
except AD.AcceleratorProbeError as e:
    check("  and it blames XPU", e.backend == "xpu")

# c. Strict, healthy CUDA + malformed MPS: raises.
r = Recorder(cuda=True, xpu=False, mps="junk")
raises("strict CUDA + malformed MPS: raises AcceleratorProbeError", AD.AcceleratorProbeError,
       lambda: AD.startup_accelerator(r, strict_conflict=True, warn=False))
r2 = Recorder(cuda=True, xpu=False, mps="junk")
try:
    AD.startup_accelerator(r2, strict_conflict=True, warn=False)
except AD.AcceleratorProbeError as e:
    check("  and it blames MPS with the malformed status",
          e.backend == "mps" and e.status == AD.STATUS_MALFORMED)

# d. Strict, healthy CUDA + cleanly unavailable XPU and MPS: succeeds, scan is trustworthy.
r = Recorder(cuda=True, xpu=False, mps=False)
cfg = expect("strict CUDA + clean unavailable XPU/MPS: succeeds",
             lambda: AD.startup_accelerator(r, strict_conflict=True, warn=False),
             lambda c: c["backend"] == "cuda")
check("  conflict_scanned is True only because every status was clean",
      bool(cfg) and cfg["conflict_scanned"] is True)
check("  and conflicting_signals is a real negative answer, not a gap",
      bool(cfg) and cfg["conflicting_signals"] is False)
check("  every backend really was probed (saw %r)" % r.calls,
      set(r.calls) == {"cuda", "xpu", "mps"})

# e. Strict, multiple clean available: refuses.
r = Recorder(cuda=True, xpu=True, mps=True)
raises("strict, three cleanly available: raises ConflictingBackendsError",
       AD.ConflictingBackendsError,
       lambda: AD.startup_accelerator(r, strict_conflict=True, warn=False))

# f. Strict with no accelerator at all: NoAcceleratorError, not a conflict error.
r = Recorder(cuda=False, xpu=False, mps=False)
raises("strict, nothing available: raises NoAcceleratorError", AD.NoAcceleratorError,
       lambda: AD.startup_accelerator(r, strict_conflict=True, warn=False))

print("\n--- 16c. NEGATIVE CONTROL: conflict_scanned=True is unreachable with a dirty scan ---")
# Exhaustive over every combination of the five real statuses across three backends. If ANY
# result comes back with conflict_scanned=True while a scanned status was malformed or
# exception, strict mode is lying about what it established.
_BEHAVIOURS = {"avail": True, "unavail": False, "boom": "boom", "junk": "junk", "gone": None}
_dirty_true = []
_clean_true = 0
_checked = 0
for _c in _BEHAVIOURS:
    for _x in _BEHAVIOURS:
        for _m in _BEHAVIOURS:
            _checked += 1
            rr = Recorder(cuda=_BEHAVIOURS[_c], xpu=_BEHAVIOURS[_x], mps=_BEHAVIOURS[_m])
            try:
                res = AD.startup_accelerator(rr, strict_conflict=True, warn=False)
            except Exception:
                continue
            statuses = [res["probe_report"][b][0] for b in AD.BACKEND_PRECEDENCE]
            dirty = [s for s in statuses if s in AD.FAULT_STATUSES]
            if res["conflict_scanned"] and dirty:
                _dirty_true.append((_c, _x, _m, statuses))
            elif res["conflict_scanned"]:
                _clean_true += 1
check("exhaustive %d strict combinations: conflict_scanned=True never coexists with a fault "
      "(violations: %r)" % (_checked, _dirty_true), _dirty_true == [])
check("  and the sweep did reach clean conflict_scanned=True cases (%d), so it is not vacuous"
      % _clean_true, _clean_true > 0)

print("\n--- 17. NEGATIVE CONTROL: the call-order assertions can fail ---")
# Without this, section 16 could be green against an eager implementation and nobody would
# know. Inject the exact defect the reviewer found and require the recorder to catch it.
_real_select = AD.select_backend


def _eager_select(torch_mod, report=None):
    """The previous behaviour: compute a full report first, then walk it."""
    return _real_select(torch_mod, AD.probe_report(torch_mod) if report is None else report)


AD.select_backend = _eager_select
try:
    r = Recorder(cuda=True, xpu=True, mps=True)
    seq = _calls_for(r, lambda: AD.select_backend(r))
    check("an eager implementation IS caught by the recorder (saw %r)" % seq,
          seq != ["cuda"] and "xpu" in seq)
finally:
    AD.select_backend = _real_select
r = Recorder(cuda=True, xpu=True, mps=True)
seq = _calls_for(r, lambda: AD.select_backend(r))
check("restored: lazy again after the negative control (saw %r)" % seq, seq == ["cuda"])

print("\n" + "=" * 78)
print("PASS %d   FAIL %d" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAILED: %s" % f)
print("=" * 78)
sys.exit(1 if FAIL else 0)
