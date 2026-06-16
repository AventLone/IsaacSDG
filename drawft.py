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


class SDG(BaseSDG):
    
    def __init__(self, stage_url: str, dome_texture_urls: list, 
                 boxes_urls_and_weights: Optional[list]=None,
                 save_path=None) -> None:
        super().__init__(save_path=save_path)

        if not stage_utils.open_stage(stage_url):
            raise RuntimeError(f"Failed to open {stage_url}")
        tools.app_update(2)
        LOGGER.info("SDG is starting...")
        
        self._this_stage = prims_utils.get_current_stage()
        
        self._dome_prim_path = "/World/Lights/DomeLight"
        self._dome_texture_urls = dome_texture_urls
        self._dome_prim = prims_utils.create_prim(prim_path=self._dome_prim_path, prim_type="DomeLight",
                                            attributes={"inputs:intensity": 1000.0,
                                                        "inputs:texture:file": dome_texture_urls[0]})
        self._dome_texture = self._dome_prim.GetAttribute("inputs:texture:file")

        self._light_randomizer = LightRandomizer()
        self._looks_randomizer = LooksRandomizer(random_looks_count=300)
        self.create_camera()

    def _random_dome_texture(self, show_environment_prob: float = 0.5):
        show_environment = random.random() <= show_environment_prob
        common.make_visible("/Environment", show_environment)
        if show_environment:
            return
        self._dome_texture.Set(random.choice(self._dome_texture_urls))

    async def prepare_objects(self):
        # prepared_prim_paths = ["/World/Objects/Prepared/eu", "/World/Objects/Prepared/plastic_1", 
        #                    "/World/Objects/Prepared/plastic_2", "/World/Objects/Prepared/KKP"]
        pile_prim_paths = {"eu": list(), "plastic_1": list(), "plastic_2": list()}

        with tqdm(total=len(pile_prim_paths) * 4, desc="Preparation Progress", unit=" unit", file=sys.stdout) as pbar:
            # 1. Create piles
            parent_prim_path = "/World/Objects/Piles"
            for pallet_name, piles in pile_prim_paths.items():
                for i in range(2, 6):
                    piles.append(tools.Pile.generate(f"/World/Objects/Prepared/{pallet_name}", parent_prim_path, i))
                    pbar.update(1)

        # 2. Create pallets with goods
        parent_prim_path = "/World/Objects/WithGoods"

        await asyncio.sleep(0.001)

        return pile_prim_paths

        
    def scatter_objects(self, prim_pathes, num_for_each):
        _, camera_path = tools.scatter(prim_pathes, num_for_each, (-5.0, -4.0), (3.6, 5.0))
        return camera_path[::300]   # Takes every 5th element from the path

    async def randomize_scene(self, i):
        self._random_dome_texture(show_environment_prob=0.7)
        if i % 2 == 0:
            self._looks_randomizer.randomize()
        if i % 5 == 0:
            self._light_randomizer.randomize()

        await common.wait_for(1)


    async def generate(self):
        # prim_pathes =  ["/World/Objects/Prepared/eu", "/World/Objects/Prepared/palstic_1", 
        #                    "/World/Objects/Prepared/plastic_2", "/World/Objects/Prepared/KKP"]
        TARGETS = [(-5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (3.6, 0.0, 0.0), (0.0, -4.0, 0.0), (0.0, 0.0, 0.0)]

        await common.wait_for(2)
        pile_prim_paths = await self.prepare_objects()
        await common.wait_for(2)

        for num_for_each in range(3, 6):
            prim_pathes = [random.choice(piles) for _, piles in pile_prim_paths.items()]
            prim_pathes.append("/World/Objects/Prepared/KKP")
            scatter_prim_path = "/World/Objects/Scatter"
            camera_path = self.scatter_objects(prim_pathes, num_for_each)
            self._looks_randomizer.set_prim(scatter_prim_path)

            with tqdm(total=len(camera_path) * len(TARGETS), 
                      desc=f"SDG Progress {num_for_each}", unit=" Frames", file=sys.stdout) as pbar:
                for i, camera_pose in enumerate(camera_path):
                    await self.randomize_scene(i)
                    for target in TARGETS:
                        self.set_camera_pose_lootat((camera_pose[0], camera_pose[1], random.uniform(0.5, 1.2)),
                                                    lookat_target=target)
                        await rep.orchestrator.step_async(rt_subframes=16)
                        pbar.update(1)

            await common.wait_for(2)
            self._this_stage.RemovePrim(scatter_prim_path)
            await common.wait_for(2)
        
        await rep.orchestrator.wait_until_complete_async()
        self.detach_renderproduct()


def main() -> None:
    boxes_urls_and_weights = [
        ("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", 0.1),
        ("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd", 0.12),
        ("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd", 0.22),
        ("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd", 0.56)
    ]

    dome_textures = common.find_files("/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/NVIDIA/Assets/Skies", "hdr")

    sdg_train = SDG(stage_url="/home/avent/Desktop/IsaacAssets/SDG-Only/warehouse_stage.usd", 
                    dome_texture_urls=dome_textures,
                    boxes_urls_and_weights=boxes_urls_and_weights,
                    save_path="/media/avent/DATA/generated_data/valid")
    try:
        tools.SIMU_APP.run_coroutine(sdg_train.generate())
    except KeyboardInterrupt:
        LOGGER.warning("Simulation interrupted by user (Ctrl+C). Cleaning up...")
    except Exception as e:
        LOGGER.error(f"Something went wrong: {e}")
    finally:
        tools.app_update(2)
        sdg_train.evaluate_datset()
        tools.SIMU_APP.close()
            

if __name__ == "__main__":
    main()
