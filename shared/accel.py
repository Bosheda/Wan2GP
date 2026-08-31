"""Provider-neutral accelerator detection.

WHY
===
`torch.cuda.get_device_capability(None)` raises `AssertionError: Torch not compiled with CUDA
enabled` on a torch build without CUDA. Three sites call it while a module is being imported,
so on an XPU-only build `import wgp` fails outright and the render entrypoint does not exist:

  shared/attention.py:14   module level
  shared/attention.py:240  inside get_supported_attention_modes(), which wgp.py:2426 calls at
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
a capability that no hardware reports. Downstream code then selects CUDA-only kernels on a
device that cannot run them, and the failure surfaces far from its cause. This module reports
the truth instead and lets callers branch on it.

RULES
=====
1. CUDA branch may query CUDA compute capability.
2. Non-CUDA backends never call `torch.cuda` beyond the availability probe described below.
3. No backend fabricates a CUDA capability. `cuda_capability()` returns None when the
   question does not apply, because `(0, 0)` and `(11, 0)` are both lies that a later
   `major >= 8` will act on without complaint.
4. Absence of any accelerator is available as a loud failure via `require_backend()`.
5. Backend identity is separate from CUDA capability. They are different questions.
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
into a confident yes. A value that is not recognisably boolean is treated as UNKNOWN, and
unknown never silently becomes available.
"""

BACKEND_CUDA = "cuda"
BACKEND_XPU = "xpu"
BACKEND_MPS = "mps"

# CUDA first. That ordering is what preserves existing behaviour on a CUDA machine.
BACKEND_PRECEDENCE = (BACKEND_CUDA, BACKEND_XPU, BACKEND_MPS)


class AccelError(Exception):
    """Base class so a caller can catch everything from this module at once."""


class NoAcceleratorError(AccelError):
    """No supported accelerator is present."""


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


def _probe(obj, name):
    """Call `obj.name()` and return a strict tri-state.

    None covers every ambiguous case at once: the attribute is missing, the call raised, or
    the return value is not recognisably boolean. A backend we cannot get a clean yes from is
    not a backend we should select.
    """
    if obj is None:
        return None
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return _strict_bool(fn())
    except Exception:
        return None


def _mps_backend(torch_mod):
    """MPS lives at `torch.backends.mps`, not `torch.mps`, so it needs its own accessor."""
    return getattr(getattr(torch_mod, "backends", None), "mps", None)


def available_backends(torch_mod):
    """Every backend reporting itself present, in precedence order.

    A list rather than one value, so CONFLICTING SIGNALS stay visible instead of being
    silently collapsed.
    """
    found = []
    if _probe(getattr(torch_mod, "cuda", None), "is_available") is True:
        found.append(BACKEND_CUDA)
    if _probe(getattr(torch_mod, "xpu", None), "is_available") is True:
        found.append(BACKEND_XPU)
    if _probe(_mps_backend(torch_mod), "is_available") is True:
        found.append(BACKEND_MPS)
    return found


def detect_backend(torch_mod):
    """The backend to use, or None when there is no accelerator.

    Returns None rather than raising: "is there an accelerator" is a fair question with a fair
    negative answer. Callers that cannot proceed without one should use `require_backend`.
    """
    found = available_backends(torch_mod)
    return found[0] if found else None


def require_backend(torch_mod):
    """The backend to use, raising `NoAcceleratorError` when there is none."""
    backend = detect_backend(torch_mod)
    if backend is None:
        raise NoAcceleratorError(
            "no supported accelerator found (checked cuda, xpu, mps). Raised rather than "
            "falling back to CPU, because a silent fallback makes a broken environment look "
            "like a working one.")
    return backend


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


def bfloat16_supported(torch_mod, default=False, warn=True):
    """Whether bf16 is usable, decided PER BACKEND.

    Returns a concrete bool because the call sites need one. When a backend cannot answer,
    `default` is used AND a line is printed, so an assumption is visible in the log rather
    than buried. The default is False on purpose: fp16 runs everywhere, so guessing wrong in
    that direction costs quality, while guessing True on a device without bf16 kernels costs
    a crash or silent garbage.

      cuda -> `torch.cuda.is_bf16_supported()` when present, else the `major >= 8` rule. That
              rule is what the unmodified code used and is preserved deliberately.
      xpu  -> `torch.xpu.is_bf16_supported()` when present, else `default`. NOT inferred from
              the device name: hardcoding "Arc supports bf16" here would be the same class of
              error as hardcoding a CUDA capability.
      mps  -> `torch.backends.mps.is_bf16_supported()` when present, else `default`.
    """
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
            print("[accel] %s exposes no is_bf16_supported(); assuming bf16=%s. Set it "
                  "explicitly if this is wrong." % (backend, default))
        return default
    return probed


def describe(torch_mod):
    """Every answer in one dict, for logging and receipts.

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
        "torch_version": getattr(torch_mod, "__version__", None),
    }
