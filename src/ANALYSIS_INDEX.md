# Shared provenance analysis index

Each dataset revision owns one immutable SQLite file at `ROOT/DATASET/VERSION/events.sqlite`. It contains raw event identity, integer nanosecond time, original graph direction, node identity and two endpoint indexes. Analysts can read it independently of the research question, model or seed. Each node-scored run owns its prediction overlay at `ROOT/DATASET/VERSION/runs/RUN_ID/predictions.sqlite`. Query only the dataset's `events` and `nodes` tables for shared facts. Attach the intended run overlay for scores and labels. In particular, incidental tables in adopted legacy files belong to the original run.

Set ROOT to the server analysis store, VERSION to the frozen dataset revision, and RUN_ID to a unique, completed model run. The CLI takes these values as arguments. For example:

```sh
python src/analysis_index.py --root "$INDEX_ROOT" --dataset "$DATASET" --version "$VERSION" build-events --server-config "$SERVER_CONFIG"
python src/analysis_index.py --root "$INDEX_ROOT" --dataset "$DATASET" --version "$VERSION" node-predictions --run "$RUN_ID" --export "$EXPORT_DIR"
python src/analysis_index.py --root "$INDEX_ROOT" --dataset "$DATASET" --version "$VERSION" incident "$NODE_ID" --op FORK
```

Building a new dataset uses a single read-only PostgreSQL repeatable-read transaction. Publication is atomic and serialized for concurrent jobs. Reusing a published revision skips the build. Use a new version for any change to the source. Prediction registration checks the completed export digest and verifies every node ID and UUID against the dataset. The overlay applies to native **node-level** exports. Other evaluation units can use the same event index with their own output format.

A completed SQLite index with the documented event schema and a matching source manifest can use `adopt-frozen-events --source FILE --manifest MANIFEST`. The source must be a finished, quiescent analysis snapshot on the same filesystem. Adoption verifies the row count and endpoint indexes, creates a permanent hard link, and seals the shared file read-only. Never resume a writer on a frozen snapshot. Overlays do not modify the shared file.
