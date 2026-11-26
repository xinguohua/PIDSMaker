# E3 baseline execution

This entry reuses PIDSMaker's native pipeline and the project's existing remote.py/tmux controller. No data processing runs on the development laptop. Native dependencies and existing database inputs are being verified by separately recorded server jobs; the 15 training cells are not accepted results.

Static `prepare` exports the 15-row matrix and source-derived dates/GT hashes only. It neither prepares data nor verifies DB coverage. Use a new output directory to avoid replacing an existing ledger:

```sh
python src/baseline_matrix.py prepare --output /LOCAL_SMALL_RECORDS
```

Copy config/baselines/server.example.json to a private server configuration location; fill paths and DB connection metadata there. Never commit the filled file. The named password environment variable must be provided by the server job environment, not the command line or manifest. Each input manifest must contain dataset, integrity_verified, raw_files (paths/checksums), database_snapshot (source/schema/version), date_coverage (split counts and observed bounds), uuid_mapping_sha256, and boundary_counts. These are evidence fields to fill after actual inspection, not a boolean-only bypass. Current DB names follow upstream dataset defaults; mismatched DB names require a reviewed adaptation.

The following are templates, not commands ready to execute: the private input manifests still require mode-specific verification; preflight e verified the isolated detector interpreter. From the outer research project, after commit/push and server verification, substitute every uppercase placeholder:

```sh
python3 src/remote.py run --id e3-velox-smoke-UNIQUE -- env PYTHONHASHSEED=0 /VERIFIED_DETECTOR_PYTHON src/baseline_matrix.py run --dataset CADETS_E3 --model velox --mode smoke --server-config /SERVER_PRIVATE_CONFIG/baselines.json
python3 src/remote.py status e3-velox-smoke-UNIQUE
python3 src/remote.py fetch e3-velox-smoke-UNIQUE
```

Because the command begins with `env`, the inner interpreter must be the verified detector-environment interpreter (use its absolute path on the server). Do not assume remote.py's controller Python has detector dependencies. Do not modify connection settings. Inspect smoke status and outputs before launching `--mode full` for CADETS_E3/velox, then the other 14 rows serially. Smoke uses the native test_mode: first date of each split, first window/limited edges (2000 cutoff in default builder), one training and feature-fitting epoch. It is not a full result. Native code may fail on tiny graphs; retain that failure and diagnose without test tuning.

Each cell uses a new artifact directory, so no cross-version or partial stage reuse occurs. A failed job is preserved; fix the cause, commit/push, and deploy a new run ID. No pull into running directories. Full logs stay in the remote job. Inputs, configuration, seed=0, commit, package versions, device, task ID, checkpoint and output paths are recorded. Status `pipeline_finished_pending_acceptance` is deliberately not a complete experiment: check full split coverage, DB/GT mapping, and reconstructed-model checkpoint reload before acceptance.

`fixed_budget` is a newly implemented, validated configuration value, not an unsupported `last` alias. It disables OCR-APT validation stopping, preserves fixed num_epochs (100 OCR-APT, 12 others), saves the final state dict, and performs validation/test inference only at that epoch. It avoids both epoch selection on test metrics and test inference between training epochs. Other selection modes are unchanged. OCR-APT's upstream calibration consults global GT UUIDs; fixed_budget instead uses the prespecified configured contamination=0.05 and per-type validation quantiles without reading GT for calibration. This threshold adaptation must be reported. Orthrus embedding fitting is restricted to train; native MAGIC test-window kNN, KAIROS online memory, and Orthrus clustering remain unchanged. No model architecture or learning objective is changed. No tuned configs are loaded.

Objects JSONL contains all scored nodes, not just alerts, with scores (higher anomalous), decisions, labels, UUIDs, decision thresholds/rules, and available time/window mappings. Native per-window loss CSVs and graph intermediates are retained. missing_gt.json distinguishes absent DB UUIDs from mapped-but-unscored objects. It does not establish coverage of all raw objects; reconcile the DB/input manifest and graph preprocessing to identify other exclusions before acceptance. Checkpoint deserialization does not prove reconstructed-model reloadability.

```sh
python src/baseline_matrix.py status /SERVER_JOB_OUTPUT
python src/baseline_matrix.py summarize /SERVER_CELL/objects.jsonl
```

Metrics are node-level with malicious positive. Zero denominators are JSON null (NA); no pooling with window or queue metrics. Summarize recomputes P/R/F1/FPR/FNR from exact labels. The entry remains pending real server integration; local unit checks do not establish train/val/test success.

Fixed-budget checkpoints are saved after the existing evaluation transition flushes pending TGN messages, before validation. Reconstruct a fresh model from the recorded configuration and graph metadata, then use `load_evaluation_checkpoint` in `src/baseline_matrix.py` (eval before load); replay the same validation then test stream without reset. Compare scores/predictions and recompute metrics before acceptance. Deserialization or the synthetic TGN unit test alone does not meet this gate.

Exports are staged before publication. `export_complete.json` is written last and hashes each output; absent marker means unusable export. Missing optional metadata and non-finite optional values become null with per-object `export_issues`; invalid required scores or labels fail export rather than become zeros. `export_failed.json` records failure without publishing partial object files. A failed export or lost tmux process is not completion. Preserve logs/artifacts, diagnose the failed stage, commit any correction, and start a new run ID; do not rerun into the existing cell or hand-edit its status. All five resolved configurations use node_evaluation; native internal windows do not imply window-level reported metrics.

Local regression: `python -m unittest discover -s tests -p 'test_baseline*.py' -v`. Real TGN test requires torch and torch_geometric; a skip is an outstanding dependency/verification item. No local test processes APT data or trains a detector. Fixed commits have been pushed and deployed for native preflight and read-only data audits; see the outer project run ledger. Those jobs are not detector-training acceptance.

Summarize now validates the completion marker, the exact five export files and their SHA256 hashes, and rejects export_failed.json or a failed cell. The regression suite covers interruption after publication of objects but before the marker and tampering with every export file.

Server launch uses `src/baseline_server.py run ... --server-config <private JSON>` when the existing Docker service is the credential source. `server_credentials.py` reads only the configured container environment variable into memory. The private JSON can specify `credential_container` and `credential_variable`; neither a secret value nor the filled JSON belongs in Git. A manifest can restrict `allowed_modes` to `smoke`; such a manifest is rejected by full runs. These restrictions do not replace actual data verification.

`src/baseline_preflight.py` runs real native imports, five resolved configurations, and the real TGN checkpoint test. `src/baseline_data_audit.py` performs sequential, read-only PostgreSQL counts and GT/UUID checks with statement and temporary-file limits. A timed-out query is recorded, never replaced with a catalog estimate. `--dates-only --statement-timeout-seconds 300` retries only the missing date coverage without repeating the completed mapping audit.

2026-11-23 recovery: SSH/Git and PG17 are verified; preflight e passed native imports, five configs and real TGN. Date audit b completed all three queries. Smoke now reads at most 4096 ordered real events per first split date before node features, then only referenced node attributes; native graph truncation and one epoch remain. Full is unchanged. DB-only smoke evidence is distinct from full raw provenance integrity. Existing cache provenance remains unverified.

## Native full preparation and selected database input

For an existing PostgreSQL input, `input_kind=existing_postgresql` and `database_verified=true` identify the audited selected database, separately from `raw_complete=false`. Require database identity/schema evidence, date coverage, node mapping fingerprint and GT checks; raw case-directory completeness is not a prerequisite. Do not claim the current DB is byte-identical to a dump without restore/hash evidence.

Four operations: (1) `run --mode smoke` validates bounded native execution; (2) `run --mode full --stop-after-preparation` executes native construction/transformation using the unchanged full dates and stops before feature fitting/materialization; (3) `run --mode full` performs 12-epoch Velox training only after measured capacity and compatible prepared-artifact sharing are configured; (4) controller `status`/`fetch` plus independent reconstruction/export checks establish acceptance. Never count preparation or pending acceptance as a completed baseline.

Preparation writes windows.jsonl (split/date/window, native filtered events, dedup events, fused edges, node occurrences, serialized bytes) and disk_progress.json. It keeps 4 GiB disk reserve under a one-second watchdog; a reserve stop exits74 and retains partial new artifacts. No old cache or learned states are deleted. The maximum allowed new write budget is initial free minus reserve, not an assertion that the stage will fit. Measured tensor payload uses the actual fused edge count, native float32 message dimensions and four int64 vectors; serialization/model/prediction overhead remains separate.

Full continuation after this run's preparation: supply `--prepared-cell /SERVER_PREPARATION/output/CADETS_E3-velox-full` to the normal full command. The runner requires a completed preparation, the same input-manifest digest, full effective settings, matching native Python/config source hashes, exact expected dates and window sizes. It hashes the graph and dictionary files, shares only `nx` and `indexid2msg` read-only, and copies tiny native done markers into the new task directories. Logs, feature fitting, features, checkpoints, predictions and evaluation caches remain in the new run. Native task restart checks must skip construction/transformation. `prepared_reuse.json` retains source-to-target links, hashes, measured counts, exact tensor payload and explicitly estimated storage allowances. A 4GiB reserve monitor stops without deletion if space unexpectedly falls below reserve. Preparation/full completion still requires independent acceptance; no tuning on test labels.

## Persistent canonical queue (2026-11-23)

Canonical server ledger: `/mnt/bigdata/nsas2020/tdsc-runs/baseline-matrix/matrix.json`.
`prepare` now leaves an existing matrix untouched. Only the locked server queue
mutates the canonical ledger; local `baseline_manifest/matrix.json` is a downloaded
mirror, not scheduling authority. Other phase ledgers are historical snapshots.
The queue retains all attempt IDs, original metadata, observed controller exit,
pipeline status, verification directory and first accepted run. Exit zero alone
never grants completion. CAD Velox's existing ad81 run is adopted, not relaunched.

Actual server commands (use the verified detector Python, not controller Python):

```sh
python src/baseline_matrix.py query --canonical /mnt/bigdata/nsas2020/tdsc-runs/baseline-matrix/matrix.json
python src/baseline_matrix.py resume --canonical /mnt/bigdata/nsas2020/tdsc-runs/baseline-matrix/matrix.json --server-config /mnt/bigdata/nsas2020/tdsc-config/velox-db-baselines-v1.json
python src/baseline_matrix.py verify --full --cell /mnt/bigdata/nsas2020/tdsc-runs/cadets-velox-full-20261123a/output/CADETS_E3-velox-full --server-config /mnt/bigdata/nsas2020/tdsc-config/velox-db-baselines-v1.json
```

`verify` requires a **new** server RUN_OUTPUT_DIR. Queue invokes it automatically
in a separate persistent tmux and retains its logs. Do not manually invoke a second
verification while it is active. `resume` requires RUN_OUTPUT_DIR for the queue's
small progress mirror and exits if another writer holds the advisory lock.
`--adopt-run /mnt/bigdata/nsas2020/tdsc-runs/cadets-velox-full-20261123a`
is supported on `queue`/`resume`. `--once` performs one reconciliation.
Existing `run`, `status`, `summarize` remain available; direct `run` is standalone
and must be registered before the queue can own it. Summarize is hash/metric checking,
not full acceptance.

After resolving the imported active run, the queue advances CAD remaining four,
THEIA five, CLEARSCOPE five. It checks full-mode input evidence, runs a fresh native
preparation in an independent fixed local clone of the deployed public snapshot,
then checks measured edge/dimension capacity before full continuation. Only verified
same-model native preparation is shared by the existing attach_prepared helper.
No old cache is assumed compatible and no files are deleted. Preparation and workers
have 4GiB disk guards. Capacity estimates explicitly include allowances; they are not
measurements. Capacity-paused rows are reconsidered on each reconciliation.
An ambiguous interrupted launch stays active/unknown and is never duplicated.
Failures retain artifacts and the queue advances independent rows, without test ranking.

Acceptance currently qualifies **Velox edge-level replay only**. Other detector
pipelines may execute but remain acceptance_blocked until their fitted/online state
and independent coverage protocols are implemented and actually verified. No local
unit test establishes model verification. Velox verification reuses the smoke tool,
checks native preprocessed edge identities against every val/test CSV, reconciles
full split edge totals and prepared input/source/GT hashes, checks the test node
universe, rebuilds a seeded native model and loads its learned checkpoint, replays
val then test, reruns native threshold/evaluation/export in a separate directory,
compares continuous scores/decisions/metrics/mapping, and hashes source artifacts
before and after. Mismatch fails closed. Native raw filtering exclusions remain in
the preparation/input evidence; this does not prove raw dump equivalence.

### Coverage repair and scheduler pause

The actual ad81 test export collided at batches 1408/1409, both named
`2018-04-12 11:52:09.266362231~2018-04-12 11:54:11.446358030.csv`.
The surviving file equals batch 1409; 1024 identities are absent. Validation's
566 batch identity multisets are complete. This is not timestamp truncation.
New native CSV names retain nanoseconds and append stable group/batch IDs.
Evaluation removes only that suffix when parsing time bounds. Existing files are
never overwritten. Legacy exports require explicit collision diagnostics/repair.

`verify --full --repair-predictions` reuses the recorded trained checkpoint,
records original train commit/config/checkpoint hashes and corrected inference
commit, writes new predictions, and reconstructs a second fresh model for a second
native val->test replay. Corrected and independent exports must agree, including
all score identities, thresholds, decisions and metrics. It does not relabel the
original failed attempt accepted. A derived attempt needs canonical review of
its lineage and resource evidence. No metric-based attempt selection is allowed.
Original manifests remain immutable; the inaccurate old no-sharing cache text is
superseded only by a separate provenance correction referencing prepared_reuse.json.

The scheduler now defaults to paused. Its canonical sibling `control.json` contains
`{"paused":true,"reason":"..."}`; atomic replacement with paused=false requests
resume only after required verifier capability and resources are qualified.
While paused it reconciles active worker state but launches no native tasks.
Non-Velox full launches remain blocked until independent model qualification;
implemented replay profiles are not evidence that those models passed.

Native checkpoint persistence now records inference-relevant non-state_dict state
and RNG snapshots. OCR-APT restores c/r buffers and frozen-type state; Orthrus and
KAIROS restore TGN buffers/messages and replay val then test without reset;
MAGIC uses isolated native validation sampling/calibration and compares the
resulting train_distance artifact. RNG snapshots document checkpoint state; native
inference's existing set_seed reset is preserved. Fitted feature states are hashed.
Missing required state artifacts fail explicitly rather than inventing state.

`process_resources.py OUTPUT -- COMMAND ...` wraps each future queue worker and
records process-tree summed RSS (shared pages may be counted twice), NVIDIA device
memory and per-process GPU memory, elapsed time and filesystem net growth. These
are sampled, not exact continuous peaks; filesystem/device totals include other
jobs. A 4GiB guard terminates only the wrapper's own child process group. Original
ad81 total RSS/VRAM cannot be reconstructed retrospectively and remain unavailable.

Explicit controls from the current fixed server code:
`python src/baseline_matrix.py control --canonical /mnt/bigdata/nsas2020/tdsc-runs/baseline-matrix/matrix.json --paused true --reason 'verification repair'`.
Add `--stop-controller` for graceful exit after reconciliation; worker tmux sessions
are untouched. Resume scheduling requires paused=false and must not be confused
with launching a paused controller using `resume`. The control record may register
a `repair_attempts` list with dataset/model/path/original_training_run_id. The
single canonical writer accepts a derived repair only after full replay checks
and successful real process-resource supervision, preserving the original failed
attempt. Training resource gaps remain explicitly recorded in the repair lineage.

### Default research operating point

The baseline runner and its verifier apply the same cost-knee rule to the native
node scores from Velox, OCR-APT, Orthrus, MAGIC and KAIROS. Each new cell includes
`knee/report.json` and, for a selected cutoff, per-node `knee/objects.jsonl`.
In new run and replay summaries, `metrics` and `default_metrics` both use the selected knee. Read `threshold_scope` with the metrics. `default_predictions` points to the matching decisions. `native_metrics` records the detector-native comparison. For historical summaries, read the frozen knee report directly.
During verification, the model is reloaded, the frozen cutoff is reused and
the per-node knee decisions are checked.

To evaluate a threshold on independent data, point the server-config entry
`knee_calibration_manifests["CADETS_E3/velox"]` (or the respective dataset/model)
at a calibration manifest. It contains `model`, `dataset`, the trained model's
`checkpoint_sha256`, `split: "calibration"`, the source `dates`, and the
calibration `objects` path (relative to the manifest or absolute) with
`objects_sha256`. The source must contain a node score and truth label in the
same native score units and direction for each node, scored using that exact
checkpoint. Its separate run dates must be disjoint from training and test
dates. A missing source triggers a retrospective recommendation for the
current scored set. This is labeled `retrospective_test`; its results are
available as the default descriptive metrics and in `exploratory_metrics`; `held_out_metrics` is null.
A dataset without both label classes cannot establish a recall/error tradeoff.

The shared cost allowance is 100 false-positive nodes per extra true-positive
node by default. Set the single server option `knee_fp_per_tp` to use another
allowance across detectors. Each policy record retains the fitted allowance.
For a completed native export or verified `corrected_export`, run:

```sh
python src/baseline_knee.py --cell /RUN/EXPORT --manifest /TRAIN_CELL/manifest.json --output /NEW/RESULT --calibration-manifest /CALIBRATION/manifest.json
```

The post-hoc command writes a fresh derived directory and also accepts
`--fp-per-tp`. Omit the calibration manifest for an explicit retrospective
analysis. A selected endpoint with reason `no_alerts_at_this_cost` means
the fitted cost favors zero alerts over every nonempty alert set.
