#!/usr/bin/env bash
#
# Run the paper's retrievals: every run file in THIS folder (*.py except make_runs.py), several side by side. The run files are copies of
# ../run_dynesty.py with other settings, written by make_runs.py; they save to ../dynesty_results unless make_runs.py was given --outdir.
#
#   python make_runs.py                      write the run files first
#   ./run_all.sh                             all of them, in alphabetical order
#   ./run_all.sh toi2431b_seed40 k2141b.py   only these (with or without .py)
#   SLOTS=8 ./run_all.sh                     on at most 8 cores (default: all logical cores)
#   PYTHON=/path/to/python ./run_all.sh      with another interpreter (default: python, i.e. the active environment)
#   nohup ./run_all.sh > run_all.log 2>&1 &  the usual way to run them overnight
#   kill -TERM <pid of run_all.sh>           stop it and every run it started (Ctrl-C does the same); the files stay here
#
# Side by side: a run whose sampler line says nprocs = n occupies n of the SLOTS cores (nprocs = 1: one process, no pool), and the next run
# starts as soon as enough slots are free. One fit on many workers scales poorly (dynesty's pool waits for its slowest worker in every round:
# 12 workers give only 3.6-5x), so single-process runs side by side use the cores two to three times better. Runs start in alphabetical
# order; one that needs more slots than are free waits, and the ones behind it wait with it (a run with nprocs > SLOTS starts once nothing
# else runs). Runs of this folder that are already going - started by hand or by another run_all.sh - are not started again, and their
# slots count as taken.
#
# Each run's stdout/stderr goes to logs/<name>.log. A run that finishes (exit 0, also when some of its bands were skipped for lack of a
# noise level) is MOVED to done/, so what is left in this folder is what still has to run; a failing run stays here and is reported. Exit
# status is 0 only if every run finished.
#
# Never edit this file in place while an instance runs: bash reads a script as it goes, so the running instance would continue at its old
# byte offset in the new text. Write the new version to another file and mv it over this one (running instances keep reading the old one).

set -u
cd "$(dirname "$0")" || exit 1

PYTHON="${PYTHON:-python}"
SLOTS="${SLOTS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"   # cores to fill
POLL="${POLL:-10}"     # seconds between checks for finished runs and free slots
mkdir -p logs done

# One BLAS/OpenMP thread per process: each run already has `nprocs` dynesty workers, and numpy's BLAS otherwise starts one thread per core
# in EVERY worker. With 12 free-geometry likelihoods running at once on 16 cores, this takes a call from 3.8 s to 0.61 s.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

# which scripts: the arguments as run names, or every run file sorted
scripts=()
if [ "$#" -gt 0 ]; then
    for n in "$@"; do scripts+=("${n%.py}.py"); done
else
    while IFS= read -r f; do [ "$f" = "make_runs.py" ] || scripts+=("$f"); done < <(ls *.py 2>/dev/null | sort)
fi
[ "${#scripts[@]}" -gt 0 ] || { echo "no run files (*.py) in $(pwd); 'python make_runs.py' writes them"; exit 1; }

# The runs this script started and that are still going (parallel arrays). macOS bash 3.2 treats an empty array as unset under set -u,
# hence the ${a[@]+"${a[@]}"} spelling.
pids=(); pfiles=(); pslots=(); pt0=()
failed=(); skipped=(); n_started=0

nprocs_of() {   # the cores run file $1 occupies: nprocs of its sampler line (SamplerConfig's default 1 if the line gives none)
    local n
    n=$(grep -E '^sampler = dict\(' "$1" | grep -oE 'nprocs=[0-9]+' | head -n 1 | cut -d= -f2)
    if [ -n "$n" ] && [ "$n" -ge 1 ]; then echo "$n"; else echo 1; fi
}

running_elsewhere() {   # is run file $1 being run by a python process this script did not start? (PS_NOW: the process list of this round)
    local f
    for f in ${pfiles[@]+"${pfiles[@]}"}; do [ "$f" = "$1" ] && return 1; done
    printf '%s\n' "$PS_NOW" | grep -qE "python[0-9.]* ${1}\$"
}

run_one() {   # in the background: one run, moved to done/ if it finished; returns the run's exit status
    "$PYTHON" "$1" > "logs/${1%.py}.log" 2>&1 || return $?
    mv -f "$1" "done/$1"
}

stop_runs() {   # Ctrl-C or kill: stop the runs this script started (background jobs of a script ignore Ctrl-C, so they must be stopped here)
    trap - INT TERM
    local p c
    for p in ${pids[@]+"${pids[@]}"}; do
        for c in $(pgrep -P "$p"); do pkill -TERM -P "$c"; kill -TERM "$c"; done   # the run's python, and its pool workers if it has any
        kill -TERM "$p"
    done 2>/dev/null
    echo "$(date '+%F %T')  stopped: ${#pids[@]} run(s) killed and left in place: ${pfiles[*]-}"
    exit 130
}
trap stop_runs INT TERM

queue=("${scripts[@]}")
echo "$(date '+%F %T')  ${#queue[@]} run(s) on $SLOTS slots, $(command -v "$PYTHON" || echo "$PYTHON")"
t_all=$SECONDS
waiting=""

while :; do
    # the runs that ended since the last round
    left=()
    for i in ${pids[@]+"${!pids[@]}"}; do
        if kill -0 "${pids[i]}" 2>/dev/null; then left+=("$i"); continue; fi
        wait "${pids[i]}"; st=$?
        m=$(( (SECONDS - pt0[i]) / 60 ))
        if [ "$st" -eq 0 ]; then
            printf '%s  done    %-40s %5d min  -> done/%s\n' "$(date +%H:%M)" "${pfiles[i]}" "$m" "${pfiles[i]}"
        else
            printf '%s  FAILED  %-40s %5d min  -> logs/%s.log\n' "$(date +%H:%M)" "${pfiles[i]}" "$m" "${pfiles[i]%.py}"
            failed+=("${pfiles[i]}")
        fi
    done
    a=(); b=(); c=(); d=()
    for i in ${left[@]+"${left[@]}"}; do a+=("${pids[i]}"); b+=("${pfiles[i]}"); c+=("${pslots[i]}"); d+=("${pt0[i]}"); done
    pids=(${a[@]+"${a[@]}"}); pfiles=(${b[@]+"${b[@]}"}); pslots=(${c[@]+"${c[@]}"}); pt0=(${d[@]+"${d[@]}"})

    # slots in use: this script's runs, and the runs of this folder started elsewhere
    PS_NOW=$(ps -A -o command=)
    used=0
    for n in ${pslots[@]+"${pslots[@]}"}; do used=$(( used + n )); done
    for f in *.py; do
        if [ -f "$f" ] && [ "$f" != "make_runs.py" ] && running_elsewhere "$f"; then used=$(( used + $(nprocs_of "$f") )); fi
    done

    # start the next runs while their slots are free
    while [ "${#queue[@]}" -gt 0 ]; do
        f="${queue[0]}"
        if [ ! -f "$f" ]; then
            if [ -f "done/$f" ]; then printf '%s  skip    %-40s already in done/\n' "$(date +%H:%M)" "$f"; skipped+=("$f")
            else printf '%s  MISSING %-40s\n' "$(date +%H:%M)" "$f"; failed+=("$f"); fi
        elif running_elsewhere "$f"; then
            printf '%s  skip    %-40s already running (started elsewhere)\n' "$(date +%H:%M)" "$f"; skipped+=("$f")
        else
            n=$(nprocs_of "$f")
            if [ "$used" -gt 0 ] && [ $(( used + n )) -gt "$SLOTS" ]; then
                [ "$waiting" = "$f" ] || printf '%s  wait    %-40s needs %d slot(s); %d of %d in use\n' "$(date +%H:%M)" "$f" "$n" "$used" "$SLOTS"
                waiting="$f"
                break
            fi
            run_one "$f" &
            pids+=("$!"); pfiles+=("$f"); pslots+=("$n"); pt0+=("$SECONDS"); n_started=$(( n_started + 1 ))
            used=$(( used + n ))
            printf '%s  start   %-40s nprocs %d; %d of %d slots in use\n' "$(date +%H:%M)" "$f" "$n" "$used" "$SLOTS"
        fi
        queue=("${queue[@]:1}")
    done

    [ "${#queue[@]}" -eq 0 ] && [ "${#pids[@]}" -eq 0 ] && break
    sleep "$POLL"
done

echo "$(date '+%F %T')  finished: $n_started run(s) in $(( (SECONDS - t_all) / 60 )) min"
[ "${#skipped[@]}" -eq 0 ] || echo "skipped: ${skipped[*]}"
if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED (left in place): ${failed[*]}"
    exit 1
fi
