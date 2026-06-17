from tools.common import SIMU_APP

import time
from isaacsim.core.utils import extensions, stage as stage_utils, prims as prims_utils
from pxr import Gf, Sdf, UsdPhysics
from tools import common

# environment_prim_path = "/World/Environment"
# stage_utils.add_reference_to_stage(
#     usd_path="/home/avent/Desktop/IsaacAssets/Environments/warehouse_trailer.usd",
#     prim_path=environment_prim_path
# )
# common.set_world_trasform(prim=environment_prim_path, translation=[-8.2, 15.1, 0.0], orientation=common.yaw2quat(90.0))
# for _ in range(100):
#     simu_app.update()

# common.add_colliders(environment_prim_path)

extensions.enable_extension("isaacsim.asset.gen.omap")
common.app_update_async(10)   # 给 Kit 几帧完成 extension startup
import omni.physx, omni.usd, omni.timeline, omni.kit.app
from isaacsim.asset.gen.omap.bindings import _omap as omap_utils
from isaacsim.asset.gen.omap.utils import compute_coordinates,  update_location
import numpy as np
import PIL.Image


class OccGridGenerator:
    app_interface = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    omap_interface = omap_utils.acquire_omap_interface()

    def __init__(self, cell_size: float) -> None:
        self.omap_interface.set_cell_size(cell_size)

    def __del__(self):
        omap_utils.release_omap_interface(self.omap_interface)

    def set_transform(self, origin, bound_min, bound_max):
        update_location(self.omap_interface, origin, bound_min, bound_max)

    def generate(self):
        self.omap_interface.update()
        self.app_interface.update()
        self.timeline.play()
        self.app_interface.update()
        self.omap_interface.generate()
        self.app_interface.update()
        self.timeline.stop()

    @property
    def occ_grid(self):
        # Format Image
        buffer = self.omap_interface.get_buffer()
        dims = self.omap_interface.get_dimensions()
        buffer = np.array(buffer)
        buffer = np.reshape(buffer, (dims[1], dims[0]))
        occupied_mask = buffer == 1.0
        freespace_mask = buffer == 0.0
        unknown_mask = ~(occupied_mask | freespace_mask)

        unknown_as_freespace = True

        if unknown_as_freespace:
            freespace_mask[unknown_mask] = True
            unknown_mask = np.zeros_like(unknown_mask)

        image = np.zeros(occupied_mask.shape, dtype=np.uint8)
        image[occupied_mask] = 255
        image[unknown_mask] = 255
        # image[freespace_mask] = 255
        return PIL.Image.fromarray(image)



common.add_colliders("/Pallet")
dim_x, dim_y, dim_z = common.get_dimensions("/Pallet")

print(f"dim_x: {dim_x}, dim_y: {dim_y}, dim_z: {dim_z}")


occ_grid_generator = OccGridGenerator(cell_size=0.05)
occ_grid_generator.set_transform((0.0, 0.0, 0.0),
                                (-dim_x / 2.0, -dim_y / 2.0, 0.0),
                                (dim_x / 2.0, dim_y / 2.0, dim_z))
occ_grid_generator.generate()
occ_grid_generator.occ_grid.save("omap_1_1.png")


while SIMU_APP.is_running():
    SIMU_APP.update()
SIMU_APP.close()






