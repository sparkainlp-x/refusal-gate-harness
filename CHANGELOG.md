# CHANGELOG

## v0.1.0 (2026-10-30)
- Initial release: unified refusal-gate harness, stdlib-only.
- Vendored gates with provenance: refusing-membrane (first-best),
  membrane-brain (midpoint), membrane-heart (strategy parameter:
  midpoint longest-run / first / midpoint_all / median),
  world-projection (midpoint + deterministic bootstrap interval),
  symmetry-quantum-flip-membrane (midpoint, 0.01 steps, 4dp).
- Harness: `run_all` (same probes, all gates, grid=101),
  `compare` (strategy keyword-only, no default), `evaluate_fresh`,
  `deltas`, `reproduction_report`.
- Reproduction verified against the source repos' own demo runs:
  brain 0.238/J 1.000, heart 0.351/J 1.000, refusing 0.700/J 0.735,
  symmetry 0.69/J 0.875; world-projection publishes no number.
- Tests: 20/20 pass, including fixture-based reproduction tests
  and the anti-silent-default strategy test.
