import tools
from tools import common
import random, sys
from tqdm import tqdm
from typing import Optional
import omni.replicator.core as rep
from writers import BaseSDG
from randomization.evnet_randomizer import LightRandomizer
from randomization import LooksRandomizer
from tools import common

# from omni.kit.async_engine import run_coroutine
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils


class SDG(BaseSDG):
    def __init__(self, stage_url: str, dome_texture_urls: list,
                 boxes_urls_and_weights: Optional[list]=None,
                 save_path=None) -> None:
        super().__init__(save_path=save_path)

        if not stage_utils.open_stage(stage_url):
            raise RuntimeError(f"Failed to open {stage_url}")
        tools.app_update(2)
        tools.LOGGER.info("SDG is starting...")

        tools.LoadedPallet.set_assets(boxes_urls_and_weights)
        
        self._this_stage = prims_utils.get_current_stage()
        
        self._dome_prim_path = "/World/Lights/DomeLight"
        self._dome_texture_urls = dome_texture_urls
        self._dome_prim = prims_utils.create_prim(prim_path=self._dome_prim_path, prim_type="DomeLight",
                                            attributes={"inputs:intensity": 1000.0,
                                                        "inputs:texture:file": dome_texture_urls[0]})
        self._dome_texture = self._dome_prim.GetAttribute("inputs:texture:file")

        self._light_randomizer = LightRandomizer()
        self._looks_randomizer = LooksRandomizer(random_looks_count=300) 

        self._random_count_index = 1
        self._environment_prim = prims_utils.get_prim_at_path("/Environment")
        tools.app_update(2)

    def _random_dome_texture(self):
        show_environment = random.choice([True, False])
        prims_utils.set_prim_visibility(self._environment_prim, show_environment)
        if show_environment:
            return
        self._dome_texture.Set(random.choice(self._dome_texture_urls))

    async def prepare_objects(self):
        # prepared_prim_paths = ["/World/Objects/Prepared/eu", "/World/Objects/Prepared/KKP"]
        pile_prim_paths = {"eu": list(), "plastic_1": list(), "plastic_2": list()}
        loaded_pallet_paths = {"eu": list(), "plastic_1": list(), "plastic_2": list(), "KKP": list()}
        num_piles = len(pile_prim_paths) * 4
        num_loaded_pallets = len(loaded_pallet_paths) * 5

        with tqdm(total=num_piles + num_loaded_pallets, desc="Preparation Progress", unit=" unit", file=sys.stdout) as pbar:
            # 1. Create piles
            parent_prim_path = "/Assets/Piles"
            for pallet_name, piles in pile_prim_paths.items():
                for i in range(2, 6):
                    piles.append(tools.Pile.generate(f"/World/Objects/Prepared/{pallet_name}", parent_prim_path, i))
                    pbar.update(1)

            # 2. Create pallets with goods
            parent_prim_path = "/Assets/LoadedPallets"
            for pallet_name, loaded_pallets in loaded_pallet_paths.items():
                overhang = 0.0 if pallet_name == "KKP" else 0.1
                for i in range(5):
                    box_num = random.randint(2 * i + 2, 5 * i + 8)
                    loaded_pallet = await tools.LoadedPallet.generate(f"/World/Objects/Prepared/{pallet_name}", pallet_name, box_num, overhang)
                    loaded_pallets.append(loaded_pallet)
                    pbar.update(1)

        return pile_prim_paths, loaded_pallet_paths

        
    def scatter_objects(self, prim_pathes, num_for_each):
        _, camera_path = tools.scatter(prim_pathes, num_for_each, (-5.0, -4.0), (3.6, 5.0))
        return camera_path[::300]   # Takes every 10th element from the path

    async def randomize_scene(self):
        # self._random_dome_texture()
        if self._random_count_index % 2 == 0:
            self._looks_randomizer.randomize()
        if self._random_count_index % 5 == 0:
            self._light_randomizer.randomize()

        self._random_count_index += 1
        
        await common.wait_for(3)


    async def generate(self):
        # objects_prim = prims_utils.get_prim_at_path("/World/Objects")
        # noise_prim = prims_utils.get_prim_at_path("/World/Noise")

        # await common.wait_for(2)
        # pile_prim_paths, loaded_pallet_paths = await self.prepare_objects()
        # await common.wait_for(2)
        self.create_camera()
        await common.wait_for(2)

        # Collect some negtive samples
        # prims_utils.set_prim_visibility(objects_prim, False)
        # prims_utils.set_prim_visibility(noise_prim, True)
        # await common.wait_for(3)
        # camera_path = tools.path_generation.generate_lawnmower_path("/World/Noise")
        # camera_path = camera_path[::15]
        # with tqdm(total=len(camera_path), desc=f"SDG Progress for Negtive", unit=" Frames", file=sys.stdout) as pbar:
        #     for camera_pose in camera_path:
        #         self._random_dome_texture()
        #         self._light_randomizer.randomize()
        #         await common.app_update_async()
        #         self.set_camera_pose_lootat((camera_pose[0], camera_pose[1], random.uniform(0.3, 3.6)),
        #                                     lookat_target=(0.0, 0.0, 0.0))
        #         await rep.orchestrator.step_async(rt_subframes=16)
        #         pbar.update(1)
        # prims_utils.set_prim_visibility(objects_prim, True)
        # prims_utils.set_prim_visibility(noise_prim, False)
        # await common.wait_for(3)

        TARGETS = [(-5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (3.6, 0.0, 0.0), (0.0, -4.0, 0.0), (0.0, 0.0, 0.0)]
        for num_for_each in range(8, 13):
            # scatter_components = ["/World/Objects/Prepared/eu", "/World/Objects/Prepared/KKP", "/World/Objects/Prepared/aluminum",
            #                       "/World/Objects/Prepared/plastic_a", "/World/Objects/Prepared/plastic_b", "/World/Objects/Prepared/rackble"]
            scatter_components = ["/World/Objects/Prepared/plastic_a"]

            scatter_prim_path = "/World/Objects/Scatter"
            camera_path = self.scatter_objects(scatter_components, num_for_each)
            self._looks_randomizer.set_prim(scatter_prim_path)

            with tqdm(total=len(camera_path) * len(TARGETS), 
                      desc=f"SDG Progress {num_for_each}", unit=" Frames", file=sys.stdout) as pbar:
                for camera_pose in camera_path:
                    for target in TARGETS:
                        await self.randomize_scene()
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
        ("/media/avent/DATA/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", 0.02),
        ("/media/avent/DATA/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd", 0.06),
        ("/media/avent/DATA/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd", 0.12),
        ("/media/avent/DATA/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd", 0.8)
    ]

    dome_textures = common.find_files("/media/avent/DATA/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/NVIDIA/Assets/Skies", "hdr")

    sdg_train = SDG(stage_url="/media/avent/DATA/IsaacAssets/SDG-Only/warehouse_stage_front_face.usd", 
                    dome_texture_urls=dome_textures,
                    boxes_urls_and_weights=boxes_urls_and_weights,
                    save_path="/media/avent/DATA/generated_data/train")
    try:
        tools.SIMU_APP.run_coroutine(sdg_train.generate())
    except KeyboardInterrupt:
        tools.LOGGER.warning("Simulation interrupted by user (Ctrl+C). Cleaning up...")
    except Exception as e:
        tools.LOGGER.error(f"Something went wrong: {e}")
    finally:
        tools.app_update(10)
        sdg_train.evaluate_datset()
        tools.SIMU_APP.close()
        # tools.app_loop()
        

if __name__ == "__main__":
    main()
