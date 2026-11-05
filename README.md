# MadNLP benchmark results

This branch archives the result files of the benchmark in `benchmark/` of MadNLP.jl. It
shares no history with the source branches.

## Layout

```
<hardware>/<commit>/<set>_<configuration>.csv
<hardware>/<commit>/<set>_<configuration>.csv.meta.toml
```

- `<hardware>` is the machine the run was made on: the CPU model for a run on the host, the
  GPU name for a run on a CUDA device. Timings are comparable only within one directory.
- `<commit>` is the full hash of the MadNLP commit that was benchmarked.
- A CSV file has one row per instance; `benchmark/README.md` on the source branches lists
  the columns and states the protocol. The `.meta.toml` file records the machine, the
  package versions and the options of the run.

- `summary.md` in the same directory gives the machine and, per set and configuration, the
  instances solved and the shifted geometric mean times.

## How results get here

The `benchmark` workflow archives the results of every commit on `master` once its tests
have passed, and of any run dispatched by hand with the archive option; `benchmark/archive.jl`
does the same from a checkout. Nothing is edited or removed afterwards.

An entry can also come from a run made outside this workflow. Its metadata says so in the
`source` and `note` fields: the entry under
`NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/47cc85e9...` holds the results of a benchmark
study on the `dev` branch, converted to this layout.

## Using the archive

From a checkout of MadNLP.jl:

```sh
julia --project=benchmark benchmark/compare.jl <base revision> <candidate revision or directory> --hardware=<hardware>
```

A revision without archived results is replaced by its nearest archived ancestor, which the
output names.
