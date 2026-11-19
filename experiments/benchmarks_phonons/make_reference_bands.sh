#!/usr/bin/env bash
# The four DFT reference bands of the MgO phonon benchmark, rebuilt from the two NIMS MDR downloads (A. Togo, CC BY 4.0):
#   reference/MgO_PBE_togo_band_{nac,nonac}.yaml        Togo's phono3py PBE MgO set (fc2 on the 4x4x4 conventional
#                                                       supercell, its own Born charges and eps_inf), via
#                                                       experiments/benchmarks_4g/phonons_mgo/compare_phono3py_pbe.py
#   reference/MgO_PBEsol_phonondb_band_{nac,nonac}.yaml phonondb mp-1265 PBEsol, via
#                                                       experiments/benchmarks_4g/phonons_mgo/compare_phonondb.py
# Both along Gamma-X-K-Gamma-L, 51 points per segment, with and without NAC.  The released files were written with
# phonopy 2.17.2 (and PyYAML 6.0, numpy 1.24); another phonopy version may change the last digits, which the sha256
# check at the end reports.
#
# Raw downloads (not redistributed; see data/README.md for the records and checksums), in $TRACE_PHONON_RAW
# (default <repo>/experiments/benchmarks_phonons/raw):
#   $TRACE_PHONON_RAW/phono3py_pbe_mgo/phono3py_params.yaml.xz   https://mdr.nims.go.jp/datasets/1fd84240-abc0-4d8f-b4d7-1db304d6166b
#   $TRACE_PHONON_RAW/phonondb_mp-1265/phonopy_params.yaml.xz    https://mdr.nims.go.jp/concern/datasets/bk128j14x
# Each .xz is checked by sha256 and unpacked beside itself (the .yaml is checked too); the downloaded yaml files are read
# with `python -I` (untrusted input, paths as arguments).
#
# Usage (any directory): experiments/benchmarks_phonons/make_reference_bands.sh [OUT_DIR]
#   OUT_DIR defaults to experiments/benchmarks_phonons/reference (the files the scorer and tests read).
#   PYTHON selects the interpreter (default python3); it needs numpy, PyYAML and phonopy.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$REPO/experiments/benchmarks_phonons"
TOOLS="$REPO/experiments/benchmarks_4g/phonons_mgo"
RAW="${TRACE_PHONON_RAW:-$HERE/raw}"
OUT="${1:-$HERE/reference}"
PY="${PYTHON:-python3}"

sha256() { "$PY" -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }

# fetch_checked SUBDIR NAME SHA_XZ SHA_YAML: verify the download, unpack it if needed, verify the unpacked yaml
fetch_checked() {
  local dir="$RAW/$1" name="$2" sha_xz="$3" sha_yaml="$4"
  if [ ! -f "$dir/$name.yaml" ]; then
    if [ ! -f "$dir/$name.yaml.xz" ]; then
      echo "missing $dir/$name.yaml.xz: download it from the record named at the top of this script" >&2; exit 1
    fi
    local got; got="$(sha256 "$dir/$name.yaml.xz")"
    if [ "$got" != "$sha_xz" ]; then
      echo "sha256 mismatch for $dir/$name.yaml.xz: $got, expected $sha_xz" >&2; exit 1
    fi
    "$PY" -I -c 'import lzma, sys; open(sys.argv[2], "wb").write(lzma.open(sys.argv[1]).read())' "$dir/$name.yaml.xz" "$dir/$name.yaml"
  fi
  local got; got="$(sha256 "$dir/$name.yaml")"
  if [ "$got" != "$sha_yaml" ]; then
    echo "sha256 mismatch for $dir/$name.yaml: $got, expected $sha_yaml" >&2; exit 1
  fi
  echo "ok  $dir/$name.yaml"
}

fetch_checked phono3py_pbe_mgo phono3py_params \
  13b12ff2f0c16d5db00ae14ee6d32b667d4ba0e3932ddd03522bb83f81d7adce d22ebbc2aec853c192204aac1474f2e586e187169b02d641254424d1306ec1df
fetch_checked phonondb_mp-1265 phonopy_params \
  25b82cafd3efc18ba3f01c99e63e9f58510350978e90c8cb4100bc00637bddce 8bbb516cdd8dd701798897ce25d8477e7bf303d13dd64efa07d9f0ef0d92fceb

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"$PY" -I "$TOOLS/compare_phono3py_pbe.py" "$RAW/phono3py_pbe_mgo/phono3py_params.yaml" "$WORK/pbe"
"$PY" -I "$TOOLS/compare_phonondb.py" "$RAW/phonondb_mp-1265/phonopy_params.yaml" "$WORK/pbesol"

mkdir -p "$OUT"
cp "$WORK/pbe/band_dft_nac.yaml" "$OUT/MgO_PBE_togo_band_nac.yaml"
cp "$WORK/pbe/band_dft_nonac.yaml" "$OUT/MgO_PBE_togo_band_nonac.yaml"
cp "$WORK/pbesol/band_dft_nac.yaml" "$OUT/MgO_PBEsol_phonondb_band_nac.yaml"
cp "$WORK/pbesol/band_dft_nonac.yaml" "$OUT/MgO_PBEsol_phonondb_band_nonac.yaml"

# sha256 of the released reference files (phonopy 2.17.2)
status=0
while read -r want name; do
  got="$(sha256 "$OUT/$name")"
  if [ "$got" = "$want" ]; then echo "identical to the released file: $OUT/$name"
  else echo "DIFFERS from the released file: $OUT/$name ($got, released $want)" >&2; status=1; fi
done <<'EOF'
725ae4bfba95efff1ef63ebff752707bc30a37f117cc78dde1b70dced6a71341 MgO_PBE_togo_band_nac.yaml
35f65f2e93b8b30e68cf26abab0cd67bd566de3ebe85cdada9962fd5dd28efb7 MgO_PBE_togo_band_nonac.yaml
2b1ef918a2651366a9f10f84f1b2732b2d0381883f0c07da4a532461b3c981e1 MgO_PBEsol_phonondb_band_nac.yaml
12b34fc2306a9648df3ea9079f718cdb701b98a0eb787aae4b9fc64581899579 MgO_PBEsol_phonondb_band_nonac.yaml
EOF
exit $status
