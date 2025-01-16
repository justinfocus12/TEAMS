import glob
import shutil

pathname_skeleton = "/orcd/archive/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-0*"
for dir2remove in glob.glob(pathname_skeleton):
    shutil.rmtree(dir2remove)
    print(f"removed {dir2remove}")


