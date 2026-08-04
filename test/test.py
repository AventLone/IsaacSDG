import tools
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
from tools import common, traverse_path
import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
# import omni.usd, omni.physx, omni.timeline
# from pxr import Gf, UsdGeom, Vt




if not stage_utils.open_stage("/media/avent/DATA/IsaacAssets/SDG-Only/warehouse_stage.usd"):
    raise RuntimeError("Failed to ope usd.")
tools.app_update(10)

cone_prim = prims_utils.get_prim_at_path("/Environment/warehouse_trailor/traffic_cones/HeavyDutyTrafficCone_A04_46cm_PR_V_NVD_01")
target_prim_path = "/Environment/warehouse_trailor/traffic_cones/HeavyDutyTrafficCone_A04_46cm_PR_V_NVD_01"


async def func():
    path = await traverse_path.get_traverse_path(lower_boundary=(-13.0, -9.0, 0.5), upper_boundary=(13.0, 40.0, 0.5),
                                                 lane_gap=2.0, sample_step=0.1)
    print(f"Size of path is {len(path)}")

    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True)

    xs = [position[0] for position in path]
    ys = [position[1] for position in path]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, linewidth=1.0, color="tab:green")
    plt.scatter(xs[::20], ys[::20], s=8, color="tab:red")
    plt.title("Traverse Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "traverse_path.png", dpi=200)
    plt.close()
    

    # for position in path:
    #     common.set_world_trasform(target_prim_path, position, scale=(0.01, 0.01, 0.01))
    #     await asyncio.sleep(0.01)

tools.SIMU_APP.run_coroutine(func())

tools.SIMU_APP.close()