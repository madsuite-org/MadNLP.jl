#!/bin/sh
# Zero-compile check of a built libmadnlp:
#
#   build/check.sh BUNDLE RUNS NL OUT
#
# Runs, with MADNLPAPP_TRACE=1, every line of RUNS (`LIB MODEL ARGS...`) through
# `cnlp-solve --solver BUNDLE/lib/libmadnlp.so` and the .nl file NL through BUNDLE/bin/madnlp,
# each with stdout to a file, a pipe and a terminal. Writes OUT/compiled.tsv (one row per
# run, written as it finishes: case, stdout, status, Julia compile ms) and
# OUT/precompile_statements.jl (the methods that were compiled, for build/libmadnlp.jl).
# Exits 1 when any run compiled something. CHECK_DEVICE=cuda runs the models through
# cnlp_solve_device (a CUDA build, models with a CUDA tier).
B=$1; RUNS=$2; NL=$3; OUT=$4
S=${CNLP_SOLVE:-$HOME/.cnlplib/bin/cnlp-solve}
mkdir -p "$OUT"; T="$OUT/compiled.tsv"; P="$OUT/trace.txt"
printf 'case\tstdout\tstatus\tjulia_ms\n' > "$T"; : > "$P"
row() {  # case kind logfile
    st=$(grep -o -m1 '^[a-z_]*  objective\|EXIT: .*\|MadNLP: [A-Z_]*' "$3" | head -1 | sed 's/  objective//')
    # no report line means the trace did not run: counted as a failure, never as 0
    ms=$(grep -o 'compiled at run time: [0-9]* ms' "$3" | grep -o '[0-9]* ms' | awk '{s+=$1; n++} END {print n ? s : "noreport"}')
    # a device solve also reports GPU kernels compiled at run time; any counts as compiling
    k=$(grep -o 'GPU kernels compiled at run time: [0-9]*' "$3" | grep -o '[0-9]*$' | awk '{s+=$1} END {print s+0}')
    [ "$ms" != noreport ] && [ "$k" -gt 0 ] && ms="$ms+${k}kernels"
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "${st:-?}" "$ms" >> "$T"
    grep -o 'precompile(.*' "$3" >> "$P"
}
run3() {  # case cmd...
    c=$1; shift
    "$@" > "$OUT/run.log" 2>&1 < /dev/null; row "$c" file "$OUT/run.log"
    "$@" 2>&1 < /dev/null | cat > "$OUT/run.log"; row "$c" pipe "$OUT/run.log"
    TERM=xterm script -q -e -c "$*" /dev/null > "$OUT/run.log" 2>&1 < /dev/null; row "$c" tty "$OUT/run.log"
}
export MADNLPAPP_TRACE=1
while read -r lib model args; do
    case "$lib" in ''|'#'*) continue ;; esac
    run3 "$(basename "$lib" .so):$model $args" "$S" "$lib:$model" $args --solver "$B/lib/libmadnlp.so" ${CHECK_DEVICE:+--device $CHECK_DEVICE}
done < "$RUNS"
d=$(mktemp -d); cp "$NL" "$d/"; stub="$d/$(basename "$NL" .nl)"
run3 "$(basename "$NL")" "$B/bin/madnlp" "$stub" -AMPL
run3 "$(basename "$NL") print" "$B/bin/madnlp" "$stub"
sort -u "$P" > "$OUT/precompile_statements.jl"; rm -rf "$d"
n=$(wc -l < "$OUT/precompile_statements.jl")
bad=$(awk -F'\t' 'NR > 1 && $4 != "0"' "$T" | wc -l)
awk -F'\t' 'NR > 1 {r++; if ($4 == "noreport") q++; else if ($4 > 0) c++} END {printf "%d runs, %d compiled something, %d without a report\n", r, c, q}' "$T"
echo "$n methods compiled (in $OUT/precompile_statements.jl)"
[ "$n" -eq 0 ] && [ "$bad" -eq 0 ]
