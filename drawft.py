import tools
from tools import common
import os, random, asyncio, sys, carb.settings
from tqdm import tqdm
from typing import Optional
import omni.replicator.core as rep
from writers import BaseSDG
from randomization.evnet_randomizer import LightRandomizer, MaterialRandomizer
from randomization import stack_boxes_on_pallet_async, volume_stack
from sample import PermuAndCombi
from tools import common

from omni.kit.async_engine import run_coroutine
from datetime import datetime
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
from tools.path_generation import generate_rectangle_path
from omni import usd
from tools.logger import LOGGER
from pxr import UsdGeom, Gf
import numpy as np

from randomization import LooksRandomizer

# async def prepare_loads_with_goods(prim_paths: list[str], loads_count: int, boxes_urls_and_weights: list):
#     pca_list = []
#     idx = 0.0
   
#     with tqdm(total=len(prim_paths) * loads_count, desc="Preparation Progress", unit="Pallet", file=sys.stdout) as pbar:
#         for prim_path in prim_paths:
#             pca = PermuAndCombi([prim_path])
#             pca.set_pose(translation=(0.0, 999.0 + idx, 0.0), yaw=0.0)
#             pca.create_columns(columns=loads_count, direction='x', gap=0.2)

#             for col in pca.column_prims:
#                 num_boxes = random.randint(10, 50)
#                 await stack_boxes_on_pallet_async(pallet_prim=col, boxes_urls_and_weights=boxes_urls_and_weights,
#                                                   num_boxes=num_boxes, overhang=0.1)
#                 pbar.update(1)

#             pca_list.append(pca)
#             idx += 3.6
#     return pca_list


class SDG(BaseSDG):
    
    def __init__(self, stage_url: str, dome_texture_urls: list, 
                 boxes_urls_and_weights: Optional[list]=None,
                 save_path=None) -> None:
        super().__init__(save_path=save_path)

        if not stage_utils.open_stage(stage_url):
            raise RuntimeError(f"Failed to open {stage_url}")
        
        self._this_stage = prims_utils.get_current_stage()
        
        self._dome_prim_path = "/World/Lights/DomeLight"
        self._dome_texture_urls = dome_texture_urls
        self._dome_prim = prims_utils.create_prim(prim_path=self._dome_prim_path, prim_type="DomeLight",
                                            attributes={"inputs:intensity": 1000.0,
                                                        "inputs:texture:file": dome_texture_urls[0]})
        # 2. Get the specific texture attribute
        self._dome_texture = self._dome_prim.GetAttribute("inputs:texture:file")

        self._light_randomizer = LightRandomizer()
        self._looks_randomizer = LooksRandomizer()
        # self._material_randomizer = MaterialRandomizer(self._pac.prim_path)
        self.create_camera()

    def _random_dome_texture(self, show_environment_prob: float = 0.5):
        show_environment = random.random() <= show_environment_prob
        common.make_visible("/Environment", show_environment)
        if show_environment:
            return
        self._dome_texture.Set(random.choice(self._dome_texture_urls))

    async def prepare_objects(self):
        prepared_prim_paths = ["/World/Objects/Prepared/eu", "/World/Objects/Prepared/palstic_1", 
                           "/World/Objects/Prepared/plastic_2", "/World/Objects/Prepared/KKP"]
        pass
        pile_prim_paths = {"eu": list(), "plastic_1": list(), "plastic_2": list()}

        # 1. Create piles
        for prim_path in prepared_prim_paths[:3]:

            a = tools.Piles.generate(parent_prim_path="/World/Objects/Piles")

        # 2. Create pallets with goods




    def scatter_objects(self, prim_pathes, num_for_each):
        _, camera_path = tools.scatter(prim_pathes, num_for_each=num_for_each,
                                       lower_bound=(-5.0, -4.0), upper_bound=(3.6, 5.0))
        return camera_path[::100]   # Takes every 5th element from the path

    def randomize_scene(self, i):
        self._random_dome_texture(show_environment_prob=0.7)
        if i % 2 == 0:
            self._looks_randomizer.randomize()
        if i % 5 == 0:
            self._light_randomizer.randomize()


    async def generate(self, sample_interval: int):
        prim_pathes = ["/World/Objects/eu", "/World/Objects/palstic_1", "/World/Objects/plastic_2"]
        targets = [(-5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (3.6, 0.0, 0.0), (0.0, -4.0, 0.0), (0.0, 0.0, 0.0)]

        await self.prepare_objects()
        
        scatter_prim_path = "/World/Objects/Scatter"
        camera_path = self.scatter_objects(prim_pathes, 5)

        self._looks_randomizer.set_prim(scatter_prim_path)

        with tqdm(total=len(camera_path) * len(targets), desc="SDG Progress", unit=" Frames", file=sys.stdout) as pbar:
            for i, camera_pose in enumerate(camera_path):
                self.randomize_scene(i)
                for target in targets:
                    self.set_camera_pose_lootat((camera_pose[0], camera_pose[1], 0.75), lookat_target=target)
                    await rep.orchestrator.step_async(rt_subframes=6)
                    pbar.update(1)

        self._this_stage.RemovePrim(scatter_prim_path)
        
        await rep.orchestrator.wait_until_complete_async()
        self.detach_renderproduct()


def main() -> None:
    boxes_urls_and_weights = [
        ("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", 0.1),
        ("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd", 0.12),
        ("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd", 0.22),
        ("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd", 0.56)
    ]

    dome_textures = common.find_files(
        "/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/NVIDIA/Assets/Skies", "hdr")

    sdg_train = SDG(stage_url="/home/avent/Desktop/IsaacAssets/SDG-Only/warehouse_stage.usd", 
                    dome_texture_urls=dome_textures,
                    boxes_urls_and_weights=boxes_urls_and_weights,
                    save_path="/media/avent/DATA/generated_data/train")
    try:
        tools.SIMU_APP.run_coroutine(sdg_train.generate(sample_interval=3))
    except Exception as e:
        LOGGER.error(f"Something went wrong: {e}")
    finally:
        tools.app_update(5)
        tools.SIMU_APP.close()

if __name__ == "__main__":
    main()
