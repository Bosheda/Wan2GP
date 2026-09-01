"""Provider-neutral accelerator detection.

WHY
===
`torch.cuda.get_device_capability(None)` raises `AssertionError: Torch not compiled with CUDA
enabled` on a torch build without CUDA. Three sites call it while a module is being imported,
so on an XPU-only build `import wgp` fails outright and the render entrypoint does not exist:

  shared/attention.py:14   module level
  shared/attention.py:240  inside get_supported_attention_modes(), which wgp.py calls at
                           module level
  wgp.py:2430              module level

The cause is not a CUDA dependency in any algorithm. It is a device guard that knows exactly
two worlds, MPS and CUDA, so every other backend lands in the CUDA branch by default. This
module supplies the missing branch.

WHAT THIS DELIBERATELY IS NOT
=============================
It is NOT a `torch.cuda` shim. `shared/mps/device_patch.py` solves the same problem by
monkeypatching `torch.cuda.get_device_capability` to return an invented `(11, 0)`. That is
less invasive but it makes `torch.cuda` lie: every later caller believes CUDA is present with
a capability no hardware reports, downstream code selects CUDA-only kernels on a device that
cannot run them, and the failure surfaces far from its cause. This module reports the truth
and lets callers branch on it.

TWO LAYERS, AND WHY THEY ARE SEPARATE
=====================================
DISCOVERY answers "which backend, if any" and never raises: `detect_backend`,
`available_backends`, `cuda_capability`, `bfloat16_supported`. Import-time code paths and
CPU-only utilities use this layer, which is why importing this module never requires an
accelerator and never fails.

STARTUP answers "is this machine fit to run" and DOES raise: `startup_accelerator`. An
application entrypoint calls it once. It is the loud failure. Keeping it out of the discovery
layer is deliberate: a CPU-only utility that happens to import a model module must not be
made to require a GPU.

WHY DISCOVERY COLLAPSES AND STARTUP DOES NOT
============================================
`_probe` maps a missing attribute, a raising probe and a malformed return all to None,
because for SELECTION they mean the same thing: not a backend we can safely choose. But that
collapse is unacceptable for DIAGNOSIS. "driver on fire" must never be reported as "no
accelerator found". `probe_report` therefore preserves the distinction as an explicit status
per backend, and `startup_accelerator` raises an error naming the backend, the status and the
underlying detail.

RULES
=====
1. CUDA branch may query CUDA compute capability.
2. Non-CUDA backends never call `torch.cuda` beyond the availability probe below.
3. No backend fabricates a CUDA capability. `cuda_capability()` returns None when the
   question does not apply, because `(0, 0)` and `(11, 0)` are both lies that a later
   `major >= 8` will act on without complaint.
4. Absence of any accelerator fails loudly at startup.
5. Backend identity is separate from CUDA capability. Different questions.
6. Feature support is decided per backend, never by reusing `major >= 8`, which is a CUDA
   notion and meaningless on an Arc or an M-series part.
7. Existing CUDA behaviour is preserved. CUDA is probed first, and on a CUDA machine every
   answer is what the unmodified code produced.

THE ONE PERMITTED torch.cuda CALL
=================================
`torch.cuda.is_available()` is the only `torch.cuda` attribute touched before a backend is
chosen. It is documented not to initialize a CUDA context and returns False rather than
raising on a build without CUDA. Everything else, `get_device_capability` above all, is
reached only after CUDA has been positively selected.

STRICT PROBE VALUES
===================
`_strict_bool` accepts only a real bool or the ints 0 and 1. It does NOT use `bool(value)`.
`bool("false")` is True in Python, so coercing a probe's return would turn a malformed answer
into a confident yes.
"""

BACKEND_CUDA = "cuda"
BACKEND_XPU = "xpu"
BACKEND_MPS = "mps"

# CUDA first. That ordering is what preserves existing behaviour on a CUDA machine.
BACKEND_PRECEDENCE = (BACKEND_CUDA, BACKEND_XPU, BACKEND_MPS)

# Probe outcomes, kept distinct for diagnosis. Only AVAILABLE means selectable.
STATUS_ABSENT = "absent"            # the submodule or the probe attribute is not there
STATUS_UNAVAILABLE = "unavailable"  # the probe ran and honestly said no
STATUS_MALFORMED = "malformed"      # the probe returned something not recognisably boolean
STATUS_EXCEPTION = "exception"      # the probe raised
STATUS_AVAILABLE = "available"      # the probe ran and said yes
STATUS_NOT_PROBED = "not_probed"    # NEVER CALLED. Not a measurement, and must not be read as
                                    # one. A higher-priority backend already decided the
                                    # outcome, so this probe was deliberately not invoked.

# The statuses that mean the machine is broken rather than merely lacking a backend.
FAULT_STATUSES = (STATUS_MALFORMED, STATUS_EXCEPTION)

# Sentinel for "the caller did not tell us which backend was selected". Distinct from None,
# which is a real answer meaning "there is no accelerator" and must not trigger a rescan.
_MISSING_BACKEND = object()


class AccelError(Exception):
    """Base class so a caller can catch everything from this module at once."""


class NoAcceleratorError(AccelError):
    """No supported accelerator is present. Raised only from the startup layer."""


class AcceleratorProbeError(AccelError):
    """A backend probe misbehaved: it raised, or returned a non-boolean.

    Distinct from NoAcceleratorError on purpose. A machine whose XPU probe throws is NOT the
    same as a machine with no accelerator, and reporting the second when the first is true
    sends whoever reads the log looking in the wrong place.
    """

    def __init__(self, backend, status, detail):
        self.backend = backend
        self.status = status
        self.detail = detail
        super(AcceleratorProbeError, self).__init__(
            "accelerator probe for %r reported %s: %s. This is a probe fault, not an absent "
            "accelerator. Fix the driver or the torch install rather than assuming CPU."
            % (backend, status, detail))


class ConflictingBackendsError(AccelError):
    """More than one backend reported present and strict conflict handling was requested."""


class MalformedCapabilityError(AccelError):
    """A backend reported a capability that is not a pair of integers.

    Raised rather than coerced. A capability that cannot be parsed is unknown, and an unknown
    value defaulted to something plausible is indistinguishable from a measured one.
    """


def _strict_bool(value):
    """True, False, or None for a value that is not recognisably boolean.

    Deliberately not `bool(value)`. See the STRICT PROBE VALUES note above.
    """
    if value is True or value is False:
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return value == 1
    return None


def _probe_detailed(obj, name):
    """(status, detail) for one probe. This is the layer that does NOT collapse."""
    if obj is None:
        return (STATUS_ABSENT, "submodule not present")
    fn = getattr(obj, name, None)
    if fn is None:
        return (STATUS_ABSENT, "%s() not present" % name)
    try:
        raw = fn()
    except Exception as exc:
        return (STATUS_EXCEPTION, "%s: %s" % (type(exc).__name__, exc))
    strict = _strict_bool(raw)
    if strict is None:
        return (STATUS_MALFORMED, "%s() returned %r, which is not a bool" % (name, raw))
    return (STATUS_AVAILABLE if strict else STATUS_UNAVAILABLE, "%s() returned %r" % (name, raw))


def _probe(obj, name):
    """Tri-state for SELECTION. Collapses every ambiguous case to None on purpose.

    A backend we cannot get a clean yes from is not a backend we should select. Diagnosis
    uses `_probe_detailed`; this is the discovery layer and it never raises.
    """
    status, _ = _probe_detailed(obj, name)
    if status == STATUS_AVAILABLE:
        return True
    if status == STATUS_UNAVAILABLE:
        return False
    return None


def _mps_backend(torch_mod):
    """MPS lives at `torch.backends.mps`, not `torch.mps`, so it needs its own accessor."""
    return getattr(getattr(torch_mod, "backends", None), "mps", None)


def _backend_objects(torch_mod):
    return (
        (BACKEND_CUDA, getattr(torch_mod, "cuda", None)),
        (BACKEND_XPU, getattr(torch_mod, "xpu", None)),
        (BACKEND_MPS, _mps_backend(torch_mod)),
    )


def _backend_object(torch_mod, name):
    if name == BACKEND_MPS:
        return _mps_backend(torch_mod)
    return getattr(torch_mod, name, None)


def probe_report(torch_mod):
    """FULL SCAN. {backend: (status, detail)} for every backend. Never raises.

    EAGER BY DESIGN. This invokes EVERY availability probe, including backends a precedence
    walk would never reach. Use it for diagnostics and for the opt-in strict conflict scan. Do
    NOT use it inside ordinary selection: a lower-priority probe that hangs, initializes
    hardware, raises outside Exception, or faults natively would then be able to damage a
    healthy higher-priority machine. `select_backend` walks lazily for exactly that reason.
    """
    return dict((name, _probe_detailed(obj, "is_available"))
                for name, obj in _backend_objects(torch_mod))


def _with_not_probed(report):
    """Fill every unexamined backend with NOT_PROBED rather than leaving the key missing.

    A caller must be able to tell "we looked and it said no" apart from "we never looked".
    Omitting the key invites a `.get(name, default)` that quietly invents the first from the
    second.
    """
    full = dict(report)
    for name in BACKEND_PRECEDENCE:
        if name not in full:
            full[name] = (STATUS_NOT_PROBED,
                          "not probed: a higher-priority backend already decided the outcome")
    return full


def format_probe_report(report):
    return "; ".join("%s=%s (%s)" % (b, report[b][0], report[b][1]) for b in BACKEND_PRECEDENCE)


def available_backends(torch_mod):
    """Every backend reporting itself present, in precedence order.

    A list rather than one value, so CONFLICTING SIGNALS stay visible instead of being
    silently collapsed.
    """
    return [name for name, obj in _backend_objects(torch_mod)
            if _probe(obj, "is_available") is True]


def detect_backend(torch_mod):
    """The backend to use, or None when there is no accelerator. Never raises.

    Returns None rather than raising: "is there an accelerator" is a fair question with a fair
    negative answer, and import-time code needs to ask it without risk.
    """
    found = available_backends(torch_mod)
    return found[0] if found else None


def select_backend(torch_mod, report=None):
    """(backend, report), probing ONE BACKEND AT A TIME and stopping at the first available.

    LAZY, and that is a statement about SIDE EFFECTS, not about the return value. Probes for
    backends after the selected one are never invoked at all. An earlier version computed a
    full `probe_report()` first and then walked it, which returned the right answer while
    still executing every probe. That is not equivalent: a lower-priority probe can hang,
    initialize hardware, raise something outside `Exception`, terminate the process, or fault
    natively, and none of those are things a healthy higher-priority machine should be exposed
    to. Ignoring a result after the fact is not the same as not asking.

    Per backend, in precedence order:
      available            -> select it and STOP. No further probe is called.
      unavailable / absent -> keep walking. Neither is an error.
      malformed / exception-> RAISE, without probing anything further. This backend outranks
                              everything unexamined, so whether it should have been selected
                              cannot be determined, and demoting the machine on the strength of
                              a broken probe is worse than stopping.

    The returned report marks unexamined backends `not_probed`, never `unavailable`, so no
    caller can mistake silence for a measurement.

    `report` may be supplied by a caller that has already done a deliberate full scan (the
    opt-in strict-conflict path). Passing one makes this function walk that snapshot instead of
    probing, which is the ONLY way it becomes eager, and the caller has to ask for it.
    """
    if report is not None:
        walked = dict(report)
        for name in BACKEND_PRECEDENCE:
            status, detail = walked[name]
            if status == STATUS_AVAILABLE:
                return (name, walked)
            if status in FAULT_STATUSES:
                raise AcceleratorProbeError(name, status, detail)
        raise NoAcceleratorError(
            "no supported accelerator found. Probes: %s. Raised rather than falling back to "
            "CPU, because a silent fallback makes a broken environment look like a working "
            "one." % format_probe_report(walked))

    seen = {}
    for name in BACKEND_PRECEDENCE:
        status, detail = _probe_detailed(_backend_object(torch_mod, name), "is_available")
        seen[name] = (status, detail)
        if status == STATUS_AVAILABLE:
            return (name, _with_not_probed(seen))
        if status in FAULT_STATUSES:
            raise AcceleratorProbeError(name, status, detail)
    # Only here has every backend genuinely been probed, so the report is complete.
    raise NoAcceleratorError(
        "no supported accelerator found. Probes: %s. Raised rather than falling back to CPU, "
        "because a silent fallback makes a broken environment look like a working one."
        % format_probe_report(seen))


def require_backend(torch_mod):
    """The backend to use, raising when there is none or when a probe that outranks it broke."""
    return select_backend(torch_mod)[0]


def _parse_capability(raw):
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        a, b = raw
        # bool is a subclass of int and (True, False) is not a capability.
        if isinstance(a, int) and isinstance(b, int) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            return (int(a), int(b))
    raise MalformedCapabilityError("capability is not a pair of integers: %r" % (raw,))


def cuda_capability(torch_mod, device=None):
    """The CUDA compute capability, or None when the question does not apply.

    None on XPU, on MPS and with no accelerator. It never fabricates a value.
    `torch.cuda.get_device_capability` is reached only after CUDA is positively selected, so
    this never triggers the AssertionError on an XPU-only build.
    """
    if _probe(getattr(torch_mod, "cuda", None), "is_available") is not True:
        return None
    return _parse_capability(torch_mod.cuda.get_device_capability(device))


def bfloat16_supported(torch_mod, default=False, warn=True, backend=_MISSING_BACKEND):
    """Whether bf16 is usable, decided PER BACKEND. Never raises.

    Pass `backend` when the caller has ALREADY selected one. Without it this function calls
    `detect_backend()`, which is a full eager scan, and that silently re-probed every
    lower-priority backend after `select_backend` had deliberately avoided them. Caught by the
    call-recording tests, not by any assertion on return values: the answer was right and the
    side effects were wrong.

    Returns a concrete bool because the call sites need one. When a backend cannot answer,
    `default` is used AND a line is printed, so the assumption is visible in the log rather
    than buried. The default is False on purpose: fp16 runs everywhere, so guessing wrong that
    way costs quality, while guessing True on a device without bf16 kernels costs a crash or
    silent garbage.

      cuda -> `torch.cuda.is_bf16_supported()` when present, else the `major >= 8` rule, which
              is what the unmodified code used and is preserved deliberately.
      xpu  -> `torch.xpu.is_bf16_supported()` when present, else `default`. NOT inferred from
              the device name: hardcoding "Arc supports bf16" here would be the same class of
              error as hardcoding a CUDA capability.
      mps  -> `torch.backends.mps.is_bf16_supported()` when present, else `default`.
    """
    if backend is _MISSING_BACKEND:
        backend = detect_backend(torch_mod)
    if backend is None:
        return default

    if backend == BACKEND_CUDA:
        probed = _probe(torch_mod.cuda, "is_bf16_supported")
        if probed is not None:
            return probed
        try:
            cap = cuda_capability(torch_mod)
        except MalformedCapabilityError:
            cap = None
        if cap is None:
            if warn:
                print("[accel] CUDA reported no usable capability; assuming bf16=%s" % default)
            return default
        return cap[0] >= 8

    if backend == BACKEND_XPU:
        probed = _probe(getattr(torch_mod, "xpu", None), "is_bf16_supported")
    else:
        probed = _probe(_mps_backend(torch_mod), "is_bf16_supported")
    if probed is None:
        if warn:
            print("[accel] %s exposes no usable is_bf16_supported(); assuming bf16=%s. Set it "
                  "explicitly if this is wrong." % (backend, default))
        return default
    return probed


def startup_accelerator(torch_mod, device=None, strict_conflict=False, warn=True):
    """THE LOUD FAILURE. Call once from an application entrypoint, never at library import.

    Returns a dict describing the selected backend, its CUDA capability (None off CUDA) and
    bf16 support.

    Raises:
      AcceleratorProbeError    a probe raised or returned a non-boolean. Checked FIRST so a
                               broken driver is never reported as an absent accelerator.
      NoAcceleratorError       every probe honestly said no.
      ConflictingBackendsError more than one backend present AND strict_conflict=True.

    On conflicting signals with strict_conflict=False (the default) selection is NOT refused,
    because the precedence rule can still choose safely: CUDA first, which is exactly what the
    unmodified code did. The conflict is reported in the returned dict and printed. Callers
    that would rather stop than proceed on an ambiguous machine pass strict_conflict=True.
    """
    if strict_conflict:
        # OPT-IN FULL SCAN. Detecting "more than one backend is available" is inherently a
        # question about backends the precedence walk would never reach, so it CANNOT be
        # answered lazily. Asking for it means accepting that every probe runs, including ones
        # that may hang or fault. That is why it is off by default and why it is a separate
        # code path rather than an extra flag threaded through the lazy walk.
        backend, report = select_backend(torch_mod, probe_report(torch_mod))
    else:
        backend, report = select_backend(torch_mod)

    # Only backends that CLEANLY reported available count as a conflict. A backend whose probe
    # raised or returned junk is not a competing candidate, it is a broken one, and if it
    # outranked the selection we would already have raised above.
    #
    # Without strict_conflict the walk stopped early, so backends after the selected one are
    # NOT_PROBED and this list is by construction just the selected backend. `conflicting` is
    # therefore False in the lazy path: not because no conflict exists, but because we did not
    # look, which is what `conflict_scanned` records.
    found = [n for n in BACKEND_PRECEDENCE if report[n][0] == STATUS_AVAILABLE]
    conflicting = len(found) > 1
    if conflicting and strict_conflict:
        raise ConflictingBackendsError(
            "multiple accelerators cleanly reported available (%s) and strict_conflict was "
            "requested. Probes: %s" % (", ".join(found), format_probe_report(report)))
    if conflicting and warn:
        print("[accel] multiple backends available (%s); selecting %r by precedence"
              % (", ".join(found), backend))

    capability = cuda_capability(torch_mod, device) if backend == BACKEND_CUDA else None
    # backend is passed explicitly so this does NOT re-run detect_backend(), which would scan
    # every lower-priority backend the lazy walk just avoided.
    bf16 = bfloat16_supported(torch_mod, warn=warn, backend=backend)
    if warn:
        # Bounded startup line so the decision and its evidence are visible in the console
        # rather than living only inside the returned dict. NOT_PROBED entries print as such,
        # so a reader can never mistake "we did not look" for "we looked and it said no".
        # This is NOT a durable receipt; shared/ACCEL_GOVERNANCE.md records capture as partial
        # for exactly that reason.
        print("[accel] backend=%s capability=%s bf16=%s conflict_scanned=%s | %s"
              % (backend, capability, bf16, strict_conflict, format_probe_report(report)))

    return {
        "backend": backend,
        "available_backends": found,
        "conflicting_signals": conflicting,
        # False in the lazy path means "not looked for", not "looked for and absent". Read
        # conflicting_signals only when this is True.
        "conflict_scanned": bool(strict_conflict),
        "cuda_capability": capability,
        "bfloat16_supported": bf16,
        "probe_report": report,
        "torch_version": getattr(torch_mod, "__version__", None),
    }


def describe(torch_mod):
    """Every answer in one dict, for logging and receipts. Never raises.

    `cuda_capability` is None on non-CUDA backends. That None is meaningful and must reach a
    receipt as UNKNOWN or NOT-APPLICABLE, never as a defaulted number.
    """
    found = available_backends(torch_mod)
    try:
        cap = cuda_capability(torch_mod)
    except MalformedCapabilityError:
        cap = None
    return {
        "backend": found[0] if found else None,
        "available_backends": found,
        "conflicting_signals": len(found) > 1,
        "cuda_capability": cap,
        "bfloat16_supported": bfloat16_supported(torch_mod, warn=False),
        "probe_report": probe_report(torch_mod),
        "torch_version": getattr(torch_mod, "__version__", None),
    }
