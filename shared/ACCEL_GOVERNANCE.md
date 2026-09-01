# Governance mapping: `shared/accel.py` and `tests/test_accel.py`

Scope note: this file documents the accelerator helper only. It is **self-contained**.
Nothing here is imported at runtime, and this fork has **no runtime dependency on any
external governance repository**. It exists so the substrate added by this branch is
accounted for rather than unowned.

## Ownership

| Artifact | Role |
|---|---|
| `shared/accel.py` | Provider-neutral accelerator discovery and the single startup gate |
| `tests/test_accel.py` | CPU-only behavioural and structural verification |
| `shared/attention.py` | Consumer, walls 1 and 2 |
| `wgp.py` | Consumer, wall 3 and the sole `startup_accelerator()` caller |

## Two layers, one rule each

- **Discovery** (`detect_backend`, `available_backends`, `cuda_capability`,
  `bfloat16_supported`, `describe`, `probe_report`): **never raises**. Safe at import, safe
  in CPU-only utilities.
- **Startup** (`startup_accelerator`, `require_backend`): **raises**. Called exactly once,
  from the application entrypoint. Never from a library import.

Adding a raising call to the discovery layer, or calling the startup layer at import time in
a library module, is the regression this split exists to prevent.

## DCLA mapping

```yaml
dcla_mapping:
  layer: accelerator-detection
  name: "Provider-neutral accelerator detection and startup gate"
  properties:
    detection: yes
      # probe_report() classifies every backend as absent / unavailable / malformed /
      # exception / available. Faults are distinguishable from absence.
    capture: partial
      # startup_accelerator() returns probe_report in its result AND emits one bounded console
      # line naming the selected backend, its capability, bf16, and every probe status. That
      # makes the decision and its evidence visible at startup.
      #
      # It is NOT durable capture, and this was previously overclaimed here. Returning a dict
      # is not recording: wgp.py holds `_accel` in memory and writes nothing. Nothing survives
      # process exit, so a later question of the form "which backend did the run on the 31st
      # actually select, and what did the probes say" cannot be answered from an artifact.
      #
      # MISSING ARTIFACT, named so it is not hand-waved: a startup receipt written to disk
      # (backend, capability, bf16, per-backend probe status, torch version, timestamp) that a
      # reviewer can read after the fact. Not added here because writing files is outside this
      # branch's scope and the path would need to be an operator decision.
    learning: partial
      # The bf16 default for a backend that exposes no probe is a documented conservative
      # assumption, printed when used. It is not learned from history, and it should not be
      # inferred from a device name.
    automation: partial
      # CPU-only tests run anywhere and gate the behaviour. Real-hardware acceptance
      # (does `import wgp` actually succeed on Intel XPU) is NOT automated here and is
      # deliberately out of scope for this branch.
  gaps:
    - Real-XPU import acceptance is unverified. Static checks clear the three identified
      import-time walls; they do not construct the full import-time call graph and therefore
      cannot prove no forbidden CUDA access is transitively reachable.
    - bf16 support on a backend without an is_bf16_supported() probe falls back to a default
      rather than a measurement.
```

## Invariants a reviewer should re-check on any change here

1. `torch.cuda.is_available()` remains the only `torch.cuda` attribute touched before a
   backend is selected.
2. `cuda_capability()` returns `None` off CUDA and never a fabricated pair.
3. Probe faults (`exception`, `malformed`) are reported **before** absence, so a broken
   driver is never described as a missing one.
4. CUDA precedence is unchanged, so a CUDA machine behaves exactly as before.
5. `tests/test_accel.py` still contains negative controls that are observed to fire.
6. **Selection is LAZY as a side effect, not merely in its result.** Probes for backends after
   the selected one are never invoked. Any new code path that computes a full `probe_report()`
   before selecting re-introduces the defect, and the returned value will look correct while
   doing so. Section 16 records probe invocations for this reason; assertions on return values
   cannot see it.
7. Unexamined backends carry `not_probed`, never `unavailable`. A status nobody measured must
   never be presentable as a measurement, and `conflict_scanned` says whether the conflict
   question was even asked.
8. `strict_conflict=True` is an **opt-in full scan** and is allowed to be eager, because
   "is more than one backend available" is inherently a question about backends the walk would
   never reach. It is off by default and lives on a separate code path so the trade is
   deliberate rather than inherited.

## Known pre-existing defects in this tree, NOT addressed by this branch

Reported so they are not mistaken for regressions introduced here. Verified against pristine
`c2d7c3e7`; line numbers are from that commit.

- `wgp.py:2917` undefined name `transformer_dtype`
- `wgp.py:6885`, `wgp.py:6888` undefined name `reuse_frames`
- `wgp.py:7726` undefined name `previous_last_frame`
- `shared/attention.py` `__all__` exports a bare `attention` that is defined nowhere in the
  module, so `from shared.attention import *` raises `AttributeError`.

None are on the import path being corrected. Each needs its own change, because resolving an
undefined name means deciding what it should have been, which is a judgement call and not a
mechanical fix.
