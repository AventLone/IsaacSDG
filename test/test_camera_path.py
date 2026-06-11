import tools

# from tools import SIMU_APP, app_update

from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
# from tools.orthographic_views import OrthographicProject
import cv2
import numpy as np
from tools import common
import asyncio

if not stage_utils.open_stage("/home/avent/Desktop/IsaacAssets/SDG-Only/warehouse_stage.usd"):
    raise RuntimeError("Failed to ope usd.")

tools.app_update(10)

# from tools import generate_lawnmower_path

camera_path = tools.generate_lawnmower_path("/World/Objects")
# target_prim_path = "/Environment/warehouse_trailor/traffic_cones/HeavyDutyTrafficCone_A04_46cm_PR_V_NVD_04"
target_prim_path = "/World/Objects/Obstacles/HeavyDutyTrafficCone_A01_46cm_PR_V_NVD_01"
# target_prim = prims_utils.get_prim_at_path("/Environment/warehouse_trailor/traffic_cones/HeavyDutyTrafficCone_A04_46cm_PR_V_NVD_04")

async def task():
    for pose in camera_path:
        common.set_world_trasform(target_prim_path, (pose[0], pose[1], 0.3), scale=(0.01, 0.01, 0.01))
        await asyncio.sleep(0.001)

tools.SIMU_APP.run_coroutine(task())

while tools.SIMU_APP.is_running():
    tools.SIMU_APP.update()

tools.SIMU_APP.close()