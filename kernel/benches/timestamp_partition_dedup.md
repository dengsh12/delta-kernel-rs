# Timestamp partition dedup benchmark

Generate a reusable metadata-only table:

```sh
cargo bench -p delta_kernel --features arrow-60,default-engine-base \
  --bench timestamp_partition_dedup -- generate /absolute/path/to/table
```

The table has 1,000,000 Adds in 100 JSON commits, eight LONG data columns, and two
TIMESTAMP partition columns. The generator uses Kernel's create-table, write-context,
Add-metadata, and transaction APIs. It checks that every serialized timestamp ends in
`Z`. The referenced data-file paths are synthetic; only metadata scans are supported.

Compare the default protection with `with_timestamp_partition_dedup(false)`:

```sh
cargo bench -p delta_kernel --features arrow-60,default-engine-base \
  --bench timestamp_partition_dedup -- run /absolute/path/to/table 10
```

Each predicate selects 0%, 1%, or 100% of files using `ts_a`. Both settings use the same
preloaded snapshot and release binary, with default row transforms and stats output.
Each scenario warms up both settings twice, then alternates the order of paired runs.
The timed interval consumes the entire `scan_metadata` iterator and excludes snapshot
construction, scan construction, and metrics extraction. It includes local log reads
with a warm filesystem cache.

Each measured run prints a JSON record with elapsed time, visitor time, transform time,
predicate time, and peak dedup-set size. Assertions check input, pruned, and selected
counts and dedup-set size. Because every timestamp has an offset, this measures the
inspection overhead without introducing extra dedup keys.
