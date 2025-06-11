import glob
import shutil

pathname_skeleton = "/orcd/archive/pog/001/ju26596/TEAMS/examples/frierson_gcm/2025-05-16/1/abs1_resT42_pertSPPT_std0p3_clip2_tau6h_L500km/TEAMS_N16_T60_ast*b4thx_kill0p5N_kpop_ipas0_popctrlPOG_*_lon180tavg*/seedinc*/data/mem*_temp"
for dir2remove in glob.glob(pathname_skeleton):
    print(f"about to remove {dir2remove}")
    shutil.rmtree(dir2remove)


