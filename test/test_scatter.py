import tools
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
from tools import common
import asyncio

if not stage_utils.open_stage("/home/avent/Desktop/IsaacAssets/SDG-Only/warehouse_stage.usd"):
    raise RuntimeError("Failed to ope usd.")
tools.app_update(10)


prim_pathes = ["/World/Objects/eu", "/World/Objects/palstic_1", "/World/Objects/plastic_2"]

grid, camera_path = tools.scatter(prim_pathes, num_for_each=5, lower_bound=(-5.0, -4.0), upper_bound=(3.6, 5.0))
tools.app_update(2)

target_prim_path = "/World/Objects/Obstacles/HeavyDutyTrafficCone_A01_46cm_PR_V_NVD_01"

async def task():
    for pose in camera_path:
        common.set_world_trasform(target_prim_path, (pose[0], pose[1], 0.3), scale=(0.01, 0.01, 0.01))
        await asyncio.sleep(0.001)

tools.SIMU_APP.run_coroutine(task())

import cv2 
cv2.imwrite("images/grid.png", grid)

while tools.SIMU_APP.is_running():
   tools.SIMU_APP.update()

tools.SIMU_APP.close()