# Plan-based CRC validation: implementation handoff

Prepared 2026-09-24. Base: `origin/main` at
`dc740f805633853d7a1a421f599c741fbe454e45`. Branch: `crc_plan_validation`.

This file preserves the implementation discussion. It is not an approved API specification.
No implementation accompanies this handoff. Names marked **proposed** are not existing APIs
or operators. Recheck the code and protocol before implementing unresolved details.

## Requested outcome and agreed direction

The user wants two capabilities:

1. Validate CRC file totals during a declarative metadata scan: initially `numFiles` and
   `tableSizeBytes`, with optional histogram validation as a follow-up.
2. Provide a snapshot API, described as `validate_checksum()`, that produces a plan to
   independently reconstruct table state from checkpoint/log actions and validate the checksum.
   The exact method name, signature, options, and result contract are undecided.

The architectural constraint is explicit: **no Kernel-side O(actions) visitor or Rust loop to
compute, reconcile, or compare validation state.** That work must execute inside the engine.
Kernel can construct plans and consume bounded aggregate/status results. Do not disguise a
collection-sized loop as plan construction by turning CRC arrays into individual `Values` rows
in Rust. Engine implementations may use their native data structures; Kernel must keep
`EngineData` opaque.

For scan validation, the user agreed to a pass-through validation operator: accumulate totals,
fail execution on mismatch, and otherwise forward the same live Add rows. There is no need to
add checksum fields to scan output or introduce a new `PlanResult` variant. An ordinary
`Aggregate` alone is insufficient because it replaces file rows with aggregate rows.

This is checksum **validation**, not checksum generation. Relational comparison does not need
`CollectList`/`ArrayAgg`. Expected CRC arrays do need engine-side `Unnest`/`Explode`, or another
explicit engine-side collection comparison facility.

## Existing plan interfaces

At the recorded base, `Plan` is a DAG stored in a flat, topologically ordered node vector.
Each `PlanNode` contains an `Operator` and input node indices. The last node is the sole
terminal output. Sharing a node does not guarantee materialization or a single physical read.

```rust
let result = engine
    .require_plan_executor()?
    .execute_op(Operation::QueryPlan(plan))?
    .into_data()?;
```

`PlanResult::Data` contains a fallible iterator of `EngineData` batches. Execution can fail
while consuming the iterator, not only when calling `execute_op`. Connector-owned scan output
can also remain in the connector's native representation, subject to the same plan semantics.

Existing operators:

- Sources: `ScanJson`, `ScanParquet`, `Values`.
- Unary: `Project`, `Filter`, `DynamicScan`, `Aggregate`.
- Binary: `SemiJoin`, including anti-join mode.
- N-ary: `UnionAll`.

Existing aggregates: `Min`, `Max`, `Sum`, `Count`, `CountStar`, `MinNonNullBy`, `MaxNonNullBy`.
`MaxNonNullBy(value, null_sentinel, key)` can select a NULL value at the winning key when its
sentinel is non-NULL. This matters when a Remove must suppress an older Add.

There is no assertion, pass-through aggregate validation, unnest, collection aggregate, general
join, or window operator. Do not assume array indexing, CASE/IF, histogram binning, or arbitrary
nested-value equality is already expressible by the current expressions/executors. Fixed
expression array construction is not `ArrayAgg` and does not explode array elements into rows.

## Goal 1: file-stat validation during scans

Entry: `Scan::declarative_metadata_scan_plan(engine)` in `kernel/src/scan/mod.rs`.
Builder: `Scan::build_metadata_scan_plan` in `kernel/src/scan/scan_plan.rs`.

The existing plan, with the proposed addition marked `[NEW]`, is:

```text
Commits after checkpoint                    Checkpoint / sidecars
  ScanJson                                    ScanParquet / ScanJson
    |                                         (DynamicScan for V2 when needed)
  Filter Add OR Remove                          |
    |                                         Filter Add
  Project normalization + file key              |
    |                                         Project normalization + file key
  Optional pruning                              |
  (retain all Removes)                        Optional pruning
    |                                           |
  D: Aggregate by file key                       |
     latest action's Add                         |
     (NULL for winning Remove) ------------------+---+
    |                                               |
  Filter add IS NOT NULL                      AntiJoin(checkpoint probe, D build)
    |                                               |
  Project(add)                                Project(add)
    +-------------------- UnionAll -----------------+
                             |
                         Live Adds
                             |
              [NEW] ValidateAggregates (proposed)
                COUNT(*) == expected numFiles
                SUM(add.size) == expected tableSizeBytes
                             |
                  Same live Add rows and schema
```

The file key includes the path and deletion-vector identity. Both commit branches use `D`.
The diagram is logical dataflow, not a commitment to a physical execution strategy.

Start with scans that have no pruning predicate. Existing scan pruning happens before
reconciliation, so appending validation to a filtered plan would compare a subset with
whole-table totals. Supporting predicate scans needs full-state validation before pruning or a
separate unfiltered computation; it is not solved by simply appending this operator.

The operator contract still needs to be written, but must cover:

- Accumulate globally across all batches/partitions, not independently per batch or worker.
- Preserve input rows and schema. The expected totals can be bounded control-plane values;
  decide how they enter the operator/plan.
- Compare totals at successful input exhaustion. A mismatch can therefore be reported after
  some batches have been yielded. Consumers must exhaust the stream to establish validation.
- Early drop, LIMIT, cancellation, or an upstream error must not report validation success.
  Requiring validation before the first output row would need buffering/materialization instead.
- Prevent optimizers from deleting validation or pushing pruning beneath it in a way that
  changes the totals. Native connector execution must preserve error/completion semantics too.
- Validate empty input as `(numFiles, tableSizeBytes) = (0, 0)`. A builder returning `None` for
  an empty relation must not accidentally bypass an enabled validation check.
- Reject missing/negative file sizes and arithmetic overflow; do not silently skip malformed
  rows via SQL NULL aggregation. Normalize an empty SUM to zero.
- Compare only against a CRC for the same table version. Missing CRCs must not break ordinary
  table reads. Decide separately how explicit validation reports missing CRCs.

These are table file totals, not validation of each file's `add.stats` JSON against data rows.
No customer data-file reads or DV bitmap reads are needed for these totals.

## Goal 2: full snapshot checksum validation

The following is a proposed logical DAG. Boxes such as comparison, binning, and assertion
describe required semantics; they are not all available operators at the recorded base.

### Independently reconstructed actual state

Let `V` be the snapshot version and `C` the selected checkpoint version. Without a checkpoint,
read commits from version zero. Use discovered/ratified log files, including catalog-provided
tails where applicable; do not construct a filesystem-only tail from guessed filenames.

```text
Checkpoint C                                  Commits C+1 .. V
  ScanParquet / ScanJson                        ScanJson
  + DynamicScan sidecars as needed               |
  Include manifest non-file actions             |
  Attach log_version=C                          Attach each log_version
          +---------------- UnionAll ----------------+
                                |
                         Version-tagged actions
                                |
       +------------------------+---------------------------+
       |                        |                           |
  Files: Add/Remove       Protocol / Metadata          Txn / DomainMetadata
  reconcile by file key   latest action by            latest action per
  and action precedence   log version                 appId / domain
       |                        |                           |
  Keep live Adds          existence + payload         Txn: retention policy
       |                                            Domain: drop tombstones
       +-- COUNT(*) -> numFiles                      AFTER latest-wins
       +-- SUM(size) -> tableSizeBytes
       +-- DV descriptor count -> numDeletionVectorsOpt
       +-- SUM(DV cardinality) -> numDeletedRecordsOpt
       +-- file-size bins -> fileSizeHistogram
       +-- deletion-count bins -> deletedRecordCountsHistogramOpt
       +-- normalized live Add relation -> allFiles comparison

Target commit V (reuse scanned input if available; otherwise read it separately)
  ScanJson -> Filter commitInfo -> txnId / inCommitTimestamp
```

Reuse the existing file reconciliation structure where appropriate. A single combined
latest-action grouping is another logical representation, not a mandate to replace the scan's
checkpoint anti-join optimization. Preserve same-version action precedence; a non-deterministic
tie in `MaxNonNullBy` is not a substitute for reconciliation rules.

Important independence and completeness constraints:

- Do not use metadata/protocol cached from the CRC being validated as the independent actual
  result. Independently replay them. Also audit whether CRC-derived snapshot state can affect
  plan construction, schema selection, or retention in a way that hides corruption.
- Full checkpoint input needs the V2 manifest's non-file actions, not just the Add-sidecar
  projection used by scans. Do not count every historical Protocol/Metadata action: counts
  describe reconciled state, where each must exist exactly once.
- For transactions, preserve Kernel's latest-log-action semantics and verify the protocol
  interpretation; do not substitute MAX(app transaction version) without checking behavior.
  Retention cutoff and treatment of missing `lastUpdated` need one consistent policy.
- For domains, retain tombstones through latest-wins reconciliation, then remove them. Filtering
  tombstones first resurrects removed domains.
- `commitInfo` comes from exactly version V, not the latest earlier non-NULL commitInfo.
  Checkpoints do not preserve it. If that commit is unavailable, report unavailable coverage
  rather than claiming the corresponding fields were validated.
- Some checkpoint representations omit or normalize Add fields. Define what can be compared
  semantically, especially for `allFiles`; unavailable historical detail is not proof of equality.

### Expected state and comparison

```text
V.crc
  ScanJson -> check exactly one object, required fields, types, and invariants
    |
    +-- required/optional scalar and P/M expected values
    +-- setTransactions -> [NEW] Unnest -> expected txn rows
    +-- domainMetadata  -> [NEW] Unnest -> expected domain rows
    +-- allFiles        -> [NEW] Unnest -> expected Add rows
    +-- histograms      -> [NEW] positional/zip array handling -> expected bins

Actual relations + expected relations
    |
    +-- Scalar/P/M semantic comparison
    +-- Collection comparison:
    |     actual ANTI JOIN expected
    |     expected ANTI JOIN actual
    |     plus duplicate-key and shape checks
    +-- Histogram comparison with identical boundaries and explicit zero bins
    |
  Project each discrepancy into a common mismatch schema
    |
  UnionAll -> Aggregate mismatch count (or bounded counts per field)
    |
  [NEW] Assert(mismatch_count == 0)
    |
  Project bounded success/coverage result
```

Both-direction anti-joins must compare payloads, not just keys. They detect missing or differing
entries, but not duplicate multiplicity; validate expected key uniqueness separately. Verify
that executor equality/hashing supports the chosen normalized nested payload representation.

Optional field absence is not an empty collection. Gate validation on field presence before
exploding arrays; present empty arrays assert empty actual state. Equality must deliberately
handle NULLs, unordered maps, protocol feature order, encoded stats, and checkpoint-normalized
fields. Do not rely on permissive stats `ParseJson` to turn malformed checksum content into NULL
and silently skip a check. Preserve CRC parser compatibility, including the `histogramOpt` alias
for `fileSizeHistogram` and the invalid case where both names are supplied.

A bounded result can stay within `PlanResult::Data`; `{validated: true}` is only illustrative.
The API must distinguish fully checked, partially checked, unavailable, and failed validation
if it supports partial coverage. Returning success without checking present fields must not be
described as full validation. Assertion failure should identify the field and table version,
without collecting all mismatching actions into Kernel memory.

## Checksum field coverage

The protocol body fields and their intended actual-state computation are:

| Field | Presence | Computation / comparison |
| --- | --- | --- |
| `tableSizeBytes` | Required | Sum live Add sizes |
| `numFiles` | Required | Count live Adds |
| `numMetadata` | Required | Reconciled Metadata existence/count; must be 1 |
| `numProtocol` | Required | Reconciled Protocol existence/count; must be 1 |
| `metadata` | Required | Latest Metadata payload |
| `protocol` | Required | Latest Protocol payload |
| `txnId` | Optional | Target commit's commitInfo transaction ID, when available |
| `inCommitTimestampOpt` | Iff ICT enabled | Target commit's in-commit timestamp |
| `setTransactions` | Optional | Latest transaction per appId, with defined retention semantics |
| `domainMetadata` | Optional | Latest domain action per domain, excluding tombstones |
| `fileSizeHistogram` | Optional | Counts and byte sums using expected size boundaries |
| `allFiles` | Optional | Reconciled live Add relation with defined normalization |
| `numDeletedRecordsOpt` | Optional | Sum live Add DV cardinalities |
| `numDeletionVectorsOpt` | Optional | Count live Adds with DV descriptors, including cardinality zero |
| `deletedRecordCountsHistogramOpt` | Optional | File counts in protocol deletion-count bins |

`Crc.version` is derived from the filename, not a JSON body field. CRC `txnId` is distinct from
the `txn` actions in `setTransactions`. Both commitInfo and CRC transaction IDs describe the
target transaction, but the protocol does not give a separate explicit equality MUST; document
the chosen validation rule and missing-value handling rather than inventing one.

File-size histograms contain `sortedBinBoundaries`, `fileCounts`, and `totalBytes`. Boundaries
may be custom; the first is zero and the last bin has no upper bound. Deletion-count histograms
contain `deletedRecordCounts` for inclusive ranges:
`[0,0]`, `[1,9]`, `[10,99]`, `[100,999]`, `[1000,9999]`, `[10000,99999]`,
`[100000,999999]`, `[1000000,9999999]`, `[10000000,2147483646]`, `[2147483647,infinity)`.

## Missing capabilities and open decisions

1. **Pass-through aggregate validation:** new IR semantics, builder support, wire conversion,
   and executor implementation. Decide whether this shares machinery with a general assertion
   operator. Do not hide CRC-specific replay inside an opaque operator merely to call it a plan.
2. **Assertion/error semantics:** an ordinary Filter that drops mismatches does not fail a plan.
   Define NULL behavior, eager versus terminal error timing, and optimizer obligations.
3. **Unnest/Explode:** needed to turn expected collection arrays into rows. Define empty/NULL
   arrays, element schema/nullability, parent-column preservation, and optional element position.
   Histograms require positions or zip semantics; independently exploding parallel arrays would
   produce a Cartesian product rather than aligned bins.
4. **Histogram assignment:** Unnest alone is insufficient for runtime custom boundaries.
   Either allow a bounded control-plane read of boundaries and build per-bin predicates, or add
   suitable engine-side range matching/indexing/binning. Verify bounds on any control-plane data.
5. **Nested comparison:** specify semantic normalization and verify support in both executors.
   Existing anti-join availability does not prove arbitrary map/struct payload comparison works.
6. **Expected input:** start from an on-disk same-version CRC, or also support in-memory CRCs?
   In-memory maps must not be expanded with action-sized Kernel loops. An older CRC plus an
   incremental update is not the independent reconstruction requested for validation.
7. **API policy:** choose name/signature, opt-in scan behavior, missing/malformed CRC handling,
   unsupported fields/features, coverage result, and behavior when target commit data is gone.

Concrete Unnest example: a V10 CRC contains domains `[A=old, B=keep]`. Commit V11 updates A
and removes B. To compute domain state from that base CRC in a future incremental plan, the
engine needs one row per domain, tagged version 10, unioned with V11 domain rows, then
latest-wins by domain and tombstone removal. Here, full validation reconstructs actual state
from checkpoint/log actions instead, so it avoids a base-CRC input dependency. It still needs
to flatten V11's expected CRC `domainMetadata` array to compare rows. The same distinction
applies to `setTransactions` and `allFiles`.

Conclusion: the approach fits the plan architecture, but **full validation is not expressible
unchanged with the current operator set**. Two scalar file totals are the smallest first slice.
The remaining work includes comparison and histogram semantics, not just adding Unnest.

## Source map and protocol references

- `kernel/src/scan/scan_plan.rs`, `kernel/src/scan/mod.rs`: live-Add plan and public entry.
- `kernel/src/plans/mod.rs`: executor and result contracts.
- `kernel/src/plans/builder.rs`, `kernel/src/plans/ir/{plan,nodes,operation}.rs`: plan IR.
- `kernel/src/plans/proto/convert.rs`, `kernel/proto/`: wire representation and conversion.
- `kernel/src/engine/sync/{plan,aggs}.rs`: reference executor and aggregate behavior.
- `datafusion-executor/src/{operator,plan,expression}.rs`: DataFusion implementation.
- `kernel/src/log_segment/protocol_metadata_replay.rs`: plan-based P/M replay to reuse.
- `kernel/src/crc/{mod,state,file_stats,file_size_histogram}.rs`: CRC fields and invariants.
- `kernel/src/log_segment/crc_replay.rs`: visitor-based CRC computation; semantics reference,
  not a permitted fallback for action-sized work in this design.
- `kernel/src/action_reconciliation/log_replay.rs`: existing file reconciliation semantics.
- `kernel/src/snapshot/mod.rs`: `resolve_crc_for_write` and checkpoint/commit CRC fallback.
- `kernel/src/scan/scan_plan/tests.rs`, `kernel/tests/integration/log/crc.rs`, and
  `kernel/tests/integration/cross_product/mod.rs`: test starting points.

Public source of truth: [Delta protocol](https://github.com/delta-io/delta/blob/master/PROTOCOL.md),
especially [Version Checksum File Schema](https://github.com/delta-io/delta/blob/master/PROTOCOL.md#version-checksum-file-schema),
[State Validation](https://github.com/delta-io/delta/blob/master/PROTOCOL.md#state-validation), and
[Action Reconciliation](https://github.com/delta-io/delta/blob/master/PROTOCOL.md#action-reconciliation).

The separate existing branch/worktree `crc_val_prototype` contains evolving visitor-based work.
It is not this branch's base. Do not modify it or assume its APIs exist on main. It may be useful
read-only for test scenarios, but do not carry its visitor implementation into this design.

## Implementation sequence and verification

Suggested sequence, subject to the user's implementation instructions:

1. Specify the minimal pass-through validation contract and implement predicate-free file totals
   through IR, protobuf, plan builder, scan integration, and both executors.
2. Establish snapshot validation API/coverage semantics, independent source construction, and
   scalar plus Protocol/Metadata validation.
3. Add Unnest and normalized relational comparison for optional collections.
4. Finish histograms and optional target-commit fields with explicit availability rules.

Tests should cover:

- Empty tables; Add/Remove/superseding Add sequences; DV identity changes and zero-cardinality DVs.
- JSON-only history; checkpoint plus tail; same-version checkpoint; V1 and V2 manifests/sidecars.
- Correct CRCs and each field mismatch; absent/stale/malformed CRCs; absent versus empty arrays;
  duplicate expected keys; malformed shapes; histogram alias handling.
- Missing/negative sizes and overflow; zero bins and custom boundaries; missing P/M actions.
- Domain tombstones; transaction retention and missing timestamps; exact-version commitInfo;
  checkpoint normalization and unavailable historical details.
- Arbitrary batch boundaries, input errors, early drop/cancellation, late validation errors,
  and identical scan output with validation enabled.
- Predicate gating, optimizer preservation, and global distributed aggregation semantics.
- Sync/DataFusion conformance, native connector execution semantics, and protobuf round trips.

Use `TestTableBuilder` and existing integration helpers where possible, with `rstest` cases
instead of duplicated flows. Read repository instructions before implementation. Root Cargo
workspace commands do not cover `datafusion-executor`; read its `CLAUDE.md` and run its tests
separately. After code changes, run formatting, relevant tests, clippy, and documentation checks
per `AGENTS.md`. This handoff itself introduces no Rust changes.
