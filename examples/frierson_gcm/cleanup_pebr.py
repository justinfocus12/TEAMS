import glob
import shutil
from os.path import join

pathname_skeleton_base = "/orcd/archive/pog/001/ju26596/TEAMS/examples/frierson_gcm/2025-05-16/1/abs1_resT42_pertSPPT_std*_clip2_tau6h_L500km/PeBr_bpg12_ibi6p0_bd50p0_mmd50p0_si*/data"
pathname_skeleton_anc = join(pathname_skeleton_base, "bole/anc*_temp")
pathname_skeleton_dsc = join(pathname_skeleton_base, "mem*_temp")
for skel in [pathname_skeleton_anc,pathname_skeleton_dsc]:
    for dir2remove in glob.glob(skel):
        print(f"about to remove {dir2remove}")
        shutil.rmtree(dir2remove)


