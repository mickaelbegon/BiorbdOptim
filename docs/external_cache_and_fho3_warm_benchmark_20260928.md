# External C cache and FHO3 cold/warm benchmark

## Cache contract

`compile_external_with_exact_derivatives` now creates a content-addressed
artifact whose key includes the serialized CasADi function, CasADi version,
platform/machine, compiler path and version, and the compiler flags.  Each
shared object has a sidecar manifest.  The loader rejects a missing, empty,
incompatible, or unloadable artifact before it can be reused.

Filling one key is guarded by an advisory POSIX `flock`; generation,
compilation, and atomic publication are therefore serialized across separate
FHO processes.  Existing FHO runs keep the native path: this remains an
explicit opt-in facility.

The cold/warm harness launches two **distinct OS processes** against one cache
directory.  Each measures complete `f`, `g`, `J`, and `H` calls and records
the cache contents before and after construction.  This prevents a false warm
result caused by Python or CasADi in-memory state.

## FHO3 result so far

The first full-NLP cold run generated the exact `STATE_CONTINUITY` source
(2,761,844 bytes) but did not finish compiling its monolithic `-O3` external
on the available isolated worker.  No `.so` or manifest was published, so the
cache correctly exposed no invalid artifact to a successor process.  The run
therefore has **no valid global hot measurement yet**.

This is an important negative result: the local compiled kernels remain fast,
but compiling the whole exact mapped state-continuity function as one C
translation unit is too resource-intensive here.  The next viable benchmark is
to cache multiple smaller homogeneous stage packets, each with its own exact
derivative external.  That avoids the monolithic source/optimizer memory peak
while preserving cross-FHO reuse.

## Safety conclusion

Do not enable the global `STATE_CONTINUITY` External in an FHO campaign yet.
The cache layer is correct and tested, but the integration must use bounded
packet kernels before it can be considered operational.
