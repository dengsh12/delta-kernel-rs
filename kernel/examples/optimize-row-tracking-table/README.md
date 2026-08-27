# Optimize a row-tracking table

This example shows the connector workflow for an `OPTIMIZE` operation that preserves stable
row-tracking metadata. It creates a row-tracking table, appends four small files in separate
commits, selects three files by size, and put their live rows into one file.

Run the example against a new local directory:

```shell
cargo run -p optimize-row-tracking-table -- /tmp/row-tracking-optimize
```
