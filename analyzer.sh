#!/usr/bin/env bash
#
#SBATCH --job-name=analyze-kernels
#SBATCH --output=analyze-kernels-%j.out
#SBATCH --error=analyze-kernels-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

# ==== Edit: files or directories to analyze ====
ROUTES=(
  "/home/kevin/algorithms/Laplace-Algorithm/laplace2d_2.cu"
  "/home/kevin/algorithms/Poisson/poisson.cu"
)

# ==== Job script map ====
declare -A JOB_MAP
JOB_MAP["poisson.cu"]="/home/kevin/algorithms/Poisson/poisson.sh"
JOB_MAP["laplace2d_2.cu"]="/home/kevin/algorithms/Laplace-Algorithm/laplace.sh"

# Ejecutar jobs (1) o solo parsear logs existentes (0)
RUN_JOBS="${RUN_JOBS:-1}"

# ==== Outputs ====
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${SLURM_SUBMIT_DIR:-$PWD}/kernel_scan_${TS}"
LOG_TXT="${OUT_DIR}/kernel_summary.log"
LOG_CSV="${OUT_DIR}/kernel_summary.csv"
RUN_DIR="${OUT_DIR}/runs"
TMP="${OUT_DIR}/.tmp_stderr"

mkdir -p "$OUT_DIR" "$RUN_DIR"
: > "$TMP"
echo "script_name,kernel_name,params,has_template,script_path" > "$LOG_CSV"

total_scripts=0
total_kernels=0

ensure_exec()  { [[ -x "$1" ]] || chmod +x "$1" || true; }

# ---------- 1) Kernel detection ----------
process_file() {
  local fpath="$1"
  local fname; fname="$(basename "$fpath")"

  awk -v FILEPATH="$fpath" -v FNAME="$fname" '
    BEGIN{ in_tag=0; capturing=0; sig=""; saw_template=0; kcount=0; }
    { sub(/\/\/.*/,""); }
    /KERNEL_TAG/ { in_tag=1; saw_template=0; next }
    {
      if (in_tag==1) {
        if ($0 ~ /template[[:space:]]*</) { saw_template=1 }
        if ($0 ~ /__global__/) { capturing=1 }
      }
      if (capturing==1) {
        sig = sig $0 " ";
        if (index($0,")")>0) {
          gsub(/[[:space:]]+/," ",sig)
          gsub(/__launch_bounds__\s*\([^)]*\)/,"",sig)
          pos = index(sig,"("); left = substr(sig,1,pos-1)
          rest = substr(sig,pos+1); rpos = index(rest,")")
          params = substr(rest,1,rpos-1)
          match(left, /([A-Za-z_][A-Za-z0-9_:]*)[[:space:]]*$/, m)
          kname = m[1]
          if (kname != "") {
            kcount++
            printf("%s,%s,\"%s\",%s,%s\n", FNAME, kname, params, (saw_template? "1":"0"), FILEPATH)
            printf("__KLINE__ %s|%s|%s\n", kname, params, (saw_template? "template":"")) > "/dev/stderr"
          }
          in_tag=0; capturing=0; sig=""; saw_template=0;
        }
      }
    }
    END{ printf("__KCOUNT__ %d\n", kcount) > "/dev/stderr" }
  ' "$fpath" 1>>"$LOG_CSV" 2> >(tee -a "$TMP" >/dev/null)

  local kcount
  kcount="$(awk '/__KCOUNT__/ {c=$2} END{print (c==""?0:c)}' "$TMP")"
  if [[ $kcount -gt 0 ]]; then
    {
      echo "[script] $fname"
      echo "    kernels: $kcount"
      awk -v INDENT="        - " -F"|" '
        /^__KLINE__/ {
          name=$2; params=$3; gsub(/^"|"$/, "", params); templ=$4
          if (templ=="template") printf("%s%s(%s)  [template]\n", INDENT, name, params);
          else                   printf("%s%s(%s)\n", INDENT, name, params);
        }' "$TMP"
    } >> "$LOG_TXT"
    total_kernels=$(( total_kernels + kcount ))
    total_scripts=$(( total_scripts + 1 ))
  fi
  : > "$TMP"
}

# ---------- 2) Build file list ----------
declare -a FILES=()
for root in "${ROUTES[@]}"; do
  if [[ -f "$root" ]]; then
    FILES+=("$root")
  elif [[ -d "$root" ]]; then
    while IFS= read -r -d '' f; do FILES+=("$f"); done < <(
      find "$root" -type f \( -iname "*.cu" -o -iname "*.cuh" \) -print0
    )
  else
    echo "WARNING: path not found: $root" >&2
  fi
done

# ---------- 3) Detect kernels ----------
for f in "${FILES[@]:-}"; do
  process_file "$f"
done

# ---------- 4) Run jobs (con sbatch, incluso dentro de SLURM) ----------
if [[ "$RUN_JOBS" == "1" ]]; then
  TYPES=(double float half)
  for f in "${FILES[@]}"; do
    base="$(basename "$f")"
    jobscript="${JOB_MAP[$base]:-}"
    if [[ -z "${jobscript}" || ! -f "$jobscript" ]]; then
      echo "[run-sequence] No job script for $base, skipping." | tee -a "$LOG_TXT"
      continue
    fi
    ensure_exec "$jobscript"

    echo "" | tee -a "$LOG_TXT"
    echo "[run-sequence] $base using job script: $jobscript" | tee -a "$LOG_TXT"

    for t in "${TYPES[@]}"; do
      echo "  Launching TYPE=$t for $base ..." | tee -a "$LOG_TXT"
      (
        cd "$(dirname "$jobscript")" || exit 1
        echo "    [RUN] (sbatch) cd $(pwd)" | tee -a "$LOG_TXT"
        jobid="$(sbatch --parsable --wait \
                  --export=ALL,TYPE="$t",AUTOTUNE=1 \
                  --output="${RUN_DIR}/${base}_${t}_%j.out" \
                  "$(basename "$jobscript")")" || {
          echo "    ERROR sbatch TYPE=$t for $base" | tee -a "$LOG_TXT"; exit 0;
        }
        out="${RUN_DIR}/${base}_${t}_${jobid}.out"
        echo "    wrote: $out" | tee -a "$LOG_TXT"
      )
    done
  done
fi

# ---------- 5) Parse GLOBAL_BEST (robusto, CSV con comas) ----------
echo "" >> "$LOG_TXT"
echo "==================== GLOBAL BEST SUMMARY ====================" >> "$LOG_TXT"
printf "script_name,kernel_name,dtype,bx,by,gflops,bw_gbps,time_ms,log_path\n" > "${OUT_DIR}/kernel_best.csv"

shopt -s nullglob
for log in "${RUN_DIR}"/*.out; do
  base="$(basename "$log")"
  # <script>_<dtype>_<jobid>.out  con dtype en {double,float,half}
  if [[ "$base" =~ ^(.+)_((double|float|half))_([0-9]+)\.out$ ]]; then
    script="${BASH_REMATCH[1]}"
    dtype="${BASH_REMATCH[2]}"
  else
    script="$(echo "$base" | sed -E 's/_(double|float|half)_[0-9]+\.out$//')"
    dtype="$(echo  "$base" | sed -E 's/^.*_(double|float|half)_[0-9]+\.out$/\1/')"
  fi

  awk -v SCRIPT="$script" -v DTYPE="$dtype" -v LOG="$log" '
    BEGIN { kernel=""; OFS="," }
    /=== KERNEL:/ { if (match($0, /KERNEL:\s*([A-Za-z0-9_]+)/, m)) kernel=m[1] }
    /\[KATTR\]/  { if (match($0, /kernel=([A-Za-z0-9_]+)/, m)) kernel=m[1] }
    /GLOBAL_BEST/ {
      match($0, /bx=([0-9]+)/, bx);
      match($0, /by=([0-9]+)/, by);
      match($0, /gflops=([0-9.]+)/, gf);
      match($0, /BW=([0-9.]+)/, bw);
      match($0, /time_ms=([0-9.]+)/, tm);
      if (kernel == "") kernel="unknown";
      print SCRIPT, kernel, DTYPE, bx[1], by[1], gf[1], bw[1], tm[1], LOG;
    }
  ' "$log" >> "${OUT_DIR}/kernel_best.csv"
done
shopt -u nullglob

# ---- 5b) Reparar kernel_name=unknown si el script tiene 1 solo kernel en LOG_CSV ----
kmap="${OUT_DIR}/.kmap"
awk -F',' '
  NR>1 {
    key=$1 "|" $2
    if (!(key in seen)) {
      seen[key]=1
      cnt[$1]++
      if (!( $1 in first )) first[$1]=$2
    }
  }
  END {
    for (s in cnt) printf("%s,%d,%s\n", s, cnt[s], first[s])
  }
' "$LOG_CSV" > "$kmap"

kb_fixed="${OUT_DIR}/kernel_best.fixed.csv"
awk -F',' -v OFS=',' '
  FNR==NR { cnt[$1]=$2; first[$1]=$3; next }
  {
    if (NR==1) { print; next }
    # $1=script_name, $2=kernel_name
    if ($2=="unknown" && cnt[$1]==1 && first[$1]!="") $2=first[$1]
    print
  }
' "$kmap" "${OUT_DIR}/kernel_best.csv" > "$kb_fixed" && mv "$kb_fixed" "${OUT_DIR}/kernel_best.csv"

# ---------- 6) FINAL SUMMARY (une por CSV, no por parseo de texto) ----------
{
  echo
  echo "==================== FINAL SUMMARY ===================="
  echo "Total scripts analyzed : $total_scripts"
  echo "Total kernels detected : $total_kernels"
  echo "-------------------------------------------------------"

  for f in "${FILES[@]}"; do
    fname=$(basename "$f")
    echo "[script] $fname"

    # Lista de kernels detectados para el script en LOG_CSV
    klist=$(awk -F',' -v FNAME="$fname" 'NR>1 && $1==FNAME { print $2 }' "$LOG_CSV" | sort -u)
    kcount=$(echo "$klist" | awk 'NF{c++} END{print (c?c:0)}')
    printf "    kernels: %d\n" "$kcount"

    while IFS= read -r kname; do
      [[ -z "$kname" ]] && continue

      # Firma (params) desde LOG_CSV
      sig=$(awk -F',' -v FNAME="$fname" -v K="$kname" '
        NR>1 && $1==FNAME && $2==K { p=$3; gsub(/^"|"$/, "", p); print p; exit }
      ' "$LOG_CSV")
      [[ -z "$sig" ]] && sig="..."

      printf "        - %s(%s)\n" "$kname" "$sig"

      # BESTs desde kernel_best.csv (puede haber varias dtypes)
      awk -F',' -v FNAME="$fname" -v K="$kname" '
        NR==1{next}
        {
          # campos: script_name,kernel_name,dtype,bx,by,gflops,bw_gbps,time_ms,log_path
          s=$1; kn=$2; dt=$3; bx=$4; by=$5; gf=$6; bw=$7; tm=$8;
          gsub(/^ +| +$/, "", s); gsub(/^ +| +$/, "", kn);
          if (s==FNAME && kn==K) {
            printf("            \342\206\222 [BEST %s] bx=%s by=%s GFLOPS=%s BW=%sGB/s time=%sms\n",
                   dt, bx, by, gf, bw, tm);
          }
        }
      ' "${OUT_DIR}/kernel_best.csv"
      echo
    done <<< "$klist"

    echo
  done

  echo "-------------------------------------------------------"
  echo "Detailed logs stored in: ${RUN_DIR}"
  echo "======================================================="
} | tee -a "$LOG_TXT"

echo "Done. Artifacts stored under: $OUT_DIR"
