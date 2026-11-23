# E3 baseline execution

This entry reuses PIDSMaker's native pipeline and the project's existing remote.py/tmux controller. No data processing runs on the development laptop. Native dependencies and existing database inputs are being verified by separately recorded server jobs; the 15 training cells are not accepted results.

Static `prepare` exports the 15-row matrix and source-derived dates/GT hashes only. It neither prepares data nor verifies DB coverage. Use a new output directory to avoid replacing an existing ledger:

```sh
python src/baseline_matrix.py prepare --output /LOCAL_SMALL_RECORDS
```

Copy config/baselines/server.example.json to a private server configuration location; fill paths and DB connection metadata there. Never commit the filled file. The named password environment variable must be provided by the server job environment, not the command line or manifest. Each input manifest must contain dataset, integrity_verified, raw_files (paths/checksums), database_snapshot (source/schema/version), date_coverage (split counts and observed bounds), uuid_mapping_sha256, and boundary_counts. These are evidence fields to fill after actual inspection, not a boolean-only bypass. Current DB names follow upstream dataset defaults; mismatched DB names require a reviewed adaptation.

The following are templates, not commands ready to execute: server config and detector interpreter are unverified. From the outer research project, after commit/push and server verification, substitute every uppercase placeholder:

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
