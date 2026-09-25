# License policy

This is a hygiene policy for THIRD-PARTY DEPENDENCIES (models, datasets,
libraries this code imports) — it governs what this codebase is built out
of, not what license this codebase itself is distributed under. ClearCut
itself is distributed under CC BY-NC 4.0 (see [`LICENSE`](../../LICENSE));
commercial use requires separate arrangement (see the top-level README).

## Every third-party dependency stays permissive, always

Everything at the repo root and in `checkpoints/`, `dev/`, `tests/` — i.e.
everything except `plugins_noncommercial/` — depends only on Apache-2.0 /
MIT / BSD-3 third-party components. No AGPL, no GPL, no dataset with a
restrictive terms-of-use, anywhere in that scope. This is independent of
ClearCut's OWN license above: keeping every dependency permissive means a
future re-license of ClearCut itself (e.g. to something more permissive)
would never be blocked by an inherited restriction from a dependency. See
[`docs/THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the full
per-dependency breakdown.

This is a hard invariant, not a preference: `tests/test_license_guard.py`
fails the build if it's violated (currently checks for AGPL-licensed
`ultralytics` imports and GPL-licensed RVM code; extend it if a new
restricted-license temptation shows up).

## Where non-commercial experiments go instead

Research into non-commercial or restrictively-licensed components (models,
datasets, algorithms) is still worth doing — several are documented in
[`docs/DECISIONS.md`](DECISIONS.md) as promising but license-blocked (e.g.
MatAnyone, SAM2Matting, RMBG-2.0 for matting; DIS5K/P3M-10k/AM-2k for
eval-only data). If any of these — or anything else non-permissive — is ever
actually implemented, it goes in [`plugins_noncommercial/`](../plugins_noncommercial/)
under its own subdirectory, never mixed into the core, per the rules in that
directory's `README.md`.

**Why a directory boundary instead of just "be careful"**: a license
violation from an accidental `import` is a silent, easy mistake in a
codebase that otherwise looks uniform. A physical, tested boundary (own
directory + a guard test that fails CI) makes the mistake loud instead.

## What this means for distribution

`plugins_noncommercial/` is optional, gitignored-friendly research
territory, not part of what gets deployed. If a build/release script is
ever written for this project, it should exclude `plugins_noncommercial/`
by default (opt-in only). Whether the deployed core itself may be used
commercially is a separate question, governed by ClearCut's own license
(CC BY-NC 4.0, see above) — NOT by this dependency-hygiene policy. A
future re-license of ClearCut to something more permissive (e.g.
Apache-2.0) would be unblocked by dependency licensing, since every
in-scope dependency is already permissive; it is not itself an act this
policy authorizes.
