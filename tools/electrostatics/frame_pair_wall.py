"""The frame pair's own frozen-density PBE repulsion below the data (2026-10-30): the non-electrostatic interaction
(Watson-sphere convention, 'sph_nonelst_pbe') of a frame ion pair at short R, through the Pauli build's own cached route
(compare_overlap.run_pair; cache $TRACE_FP_CACHE/pauli_fp, TRACE_FP_CACHE defaulting to ./fp_cache).  SPEC 25's table is frame-relative (carrier minus frame), so
the frame pair's own repulsion is carried by the learned backbone alone, which the 4G data constrain only above the
pair's smallest contact (Au-Mg: 2.43 A).  Usage: python frame_pair_wall.py ZA QA ZB QB R [R ...]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_overlap as C
SYM = {1: "H", 6: "C", 8: "O", 11: "Na", 12: "Mg", 13: "Al", 17: "Cl", 47: "Ag", 79: "Au"}   # the library (2026-11-13)
za, qa, zb, qb = (int(x) for x in sys.argv[1:5])
lib = C.library()
work = os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), "pauli_fp")
for R in (float(x) for x in sys.argv[5:]):
    o = C.run_pair((SYM[za], qa), (SYM[zb], qb), float(round(R, 4)), work, lib)
    print(f"{SYM[za]}{qa:+d}-{SYM[zb]}{qb:+d} R {R:.3f}: sph_nonelst_pbe {o['fp']['sph_nonelst_pbe']:+.4f} eV (kin {o['fp']['kin']:+.3f}, "
          f"elst {o['fp']['sph_elst']:+.4f}); checks {[k for k, v in o['checks'].items() if isinstance(v, dict) and v.get('pass') is False]}", flush=True)
