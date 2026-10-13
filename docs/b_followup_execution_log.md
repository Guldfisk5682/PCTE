# R4 follow-up execution log

## Completed work

- Re-audited and reused the completed R2/R4 seeds 41/42/43.
- Ran all seven follow-up variants for Art, Clipart, Product, Real World,
  seed 42: 28 formal runs, each with 1,000 Source and 1,000 updates per Target.
- Formal matrix finished on 2026-10-13 at approximately 09:57 UTC+8.
  The runner consumed stdout events and reported ETA; completion checks followed
  those estimates instead of frequent training-status polling.
- Passed seven 24-update real-data smoke runs. The uniform reference control
  exactly reproduced the original R4 smoke Prompt parameters and Source Bank.
- Passed 50 synthetic tests, including unanimous wrong votes corrected by
  similarity-level fusion. All four formal FUniform runs exactly reproduced R4.
- Audited 28,000 Source and 84,000 Target updates, fixed final checkpoints,
  immutable R4 references, identical beta-scan banks, and own-target exclusion.
- Reconstructed 1,044 actual active-training contribution records on CUDA,
  with exact final Source Bank equality. This audit performs no optimizer updates
  and reads no target labels; earlier temporary CPU statistics were replaced.
- Completed three-seed R4 offline error analysis and final leave-one-out Teacher
  inference by loading original final Source parameters on CPU, without training.
  CPU fusion differs from saved GPU predictions at one sample point; error
  partitions always use the saved GPU predictions.
- Verified all 28 completed runs skip safely under unchanged configuration.

## Results and boundaries

Seed-42 Macro: R4 74.6470%, beta 0/0.1/1/10 respectively
74.6224/74.6557/74.6884/74.3588%; all-at-start initial Teachers 72.5806%;
all-at-start mature R4 references 74.6533%. No new variant reached the
predeclared additional-seed gate. New beta and admission findings are single-seed
controlled explorations; R2/R4 gains have three-seed support.

Reference studies share the existing R4 preparation budget of 1,000 updates per
branch, recorded separately from the new student budget. The all-at-start mature
reference study uses future snapshots and is an offline timing control. It is not
an online method with the same end-to-end compute as R4.

The new CLI's `FBeta0`, `FBeta0p1`, `FBeta1`, `FBeta10` presets determine beta;
the inherited `--beta` argument does not override them. The original R0–R6 CLI
retains its freely configurable beta option. All original entry points remain.

## Sealed runtime provenance

- Original A/B/C runtime:
  `4ea3c0f993cdea0b38ff165236b94044e1308d2bec7c55b178fb9a13727edc28`
- Preserved R0–R6 runtime:
  `f33008a054906a428f079d001a7eb84df87a97c8bd6789d231662b39c6ff5bb2`
- Follow-up runtime:
  `67de3f781e3c9565d874a5e3e641b71c3e0972fafbef99a4fb9aee4e9135084a`

Artifacts: `runs/b_followup/`, `reports/b_followup/report.md`,
`reports/b_followup/findings.md`, `reports/b_followup/validation.json`,
`reports/b_followup/agreement/`, `reports/b_followup/variant_agreement/`,
and the standalone PNG/PDF temperature/admission figure.
