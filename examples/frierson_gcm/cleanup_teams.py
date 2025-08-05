import glob
import shutil
import sys
from os.path import join

pathname_skeleton_base = "/orcd/archive/pog/001/ju26596/TEAMS/examples/frierson_gcm/2025-05-16/2/abs1_resT42_pertSPPT_std0p3_clip2_tau6h_L500km/TEAMS_N16_T60_ast*b4thx_kill0p5N_kpop_ipas0_popctrlPOG_*/seedinc*/data"
pathname_skeleton_anc = join(pathname_skeleton_base, "init_conds/anc*_temp")
pathname_skeleton_dsc = join(pathname_skeleton_base, "mem*_temp")
dry_run = False
if len(sys.argv) > 1:
    dry_run = not bool(eval(sys.argv[1]))
for skel in [pathname_skeleton_anc,pathname_skeleton_dsc]:
    for dir2remove in glob.glob(skel):
        print(f"about to remove {dir2remove}")
        if not dry_run:
            shutil.rmtree(dir2remove)


