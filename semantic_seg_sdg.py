import tools
from tools import common, traverse_path
import random, sys
from tqdm import tqdm
import omni.replicator.core as rep
from writers import BaseSDG
from randomization import LooksRandomizer, LightRandomizer
from tools import common
from datetime import datetime
import os
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils

class SDG(BaseSDG):
    def __init__(self, stage_url: str, save_path=None) -> None:
        timestamp = datetime.now().strftime("%Y.%m.%d-%H:%M")
        self._save_at = f"generated_data/{timestamp}" if save_path is None else f"{save_path}/{timestamp}"
        data_save_dir = os.path.join(os.getcwd(), self._save_at)
        self._writer = rep.WriterRegistry.get("BasicWriter")
        self._writer.initialize(output_dir=data_save_dir,
                                rgb=True,
                                semantic_segmentation=True,
                                colorize_semantic_segmentation=False,
                                # semantic_types=["class"],                     # keep class semantics only
                                # semantic_filter_predicate="class:pallet|floor")
                                semantic_filter_predicate="class:floor")

        if not stage_utils.open_stage(stage_url):
            raise RuntimeError(f"Failed to open {stage_url}")
        tools.app_update(2)
        tools.LOGGER.info("SDG is starting...")

        self._this_stage = prims_utils.get_current_stage()
        self._light_randomizer = LightRandomizer()
        self._looks_randomizer_pallets = LooksRandomizer(random_looks_count=300)
        self._looks_randomizer_floors = LooksRandomizer(random_looks_count=200)
        self._looks_randomizer_floors.set_prims(common.find_prims(stage=self._this_stage, name_start_with="SM_floor"))

        self._random_count_index = 1
        self._environment_prim = prims_utils.get_prim_at_path("/Environment")
        tools.app_update(2)

        
    def scatter_objects(self, prim_pathes, num_for_each):
        _, camera_path = tools.scatter(prim_pathes, num_for_each, (-5.0, -4.0), (3.6, 5.0))
        return camera_path[::30]   # Takes every 10th element from the path

    async def randomize_scene(self):
        if self._random_count_index % 2 == 0:
            self._looks_randomizer_pallets.randomize()
        if self._random_count_index % 4 == 0:
            self._looks_randomizer_floors.randomize()
        if self._random_count_index % 6 == 0:
            self._light_randomizer.randomize()
        self._random_count_index += 1
        
        await common.wait_for(3)


    async def generate(self):
        self.create_camera()
        await common.wait_for(2)

        camera_path = await traverse_path.get_traverse_path(lower_boundary=(-13.0, -9.0, 0.5), upper_boundary=(13.0, 40.0, 0.5),
                                                         lane_gap=2.0, sample_step=0.1)
        camera_spin_z = 0.0
        for camera_position in tqdm(camera_path, desc="SDG Progress", unit=" Frames", file=sys.stdout):
            await self.randomize_scene()
            # self.set_camera_pose_lootat((camera_pose[0], camera_pose[1], random.uniform(0.3, 2.2)), lookat_target=target)
            camera_spin_z = (camera_spin_z + 5.0) % 360.0
            self.set_camera_pose(camera_position, (0.0, 0.0, camera_spin_z))
            await rep.orchestrator.step_async(rt_subframes=16)

        await rep.orchestrator.wait_until_complete_async()
        self.detach_renderproduct()


def main() -> None:
    sdg_train = SDG(stage_url="/media/avent/DATA/IsaacAssets/SDG-Only/warehouse_stage.usd",
                    save_path="/media/avent/DATA/generated_data/train")
    try:
        tools.SIMU_APP.run_coroutine(sdg_train.generate())
    except KeyboardInterrupt:
        tools.LOGGER.warning("Simulation interrupted by user (Ctrl+C). Cleaning up...")
    except Exception as e:
        tools.LOGGER.error(f"Something went wrong: {e}")
    finally:
        tools.app_update(10)
        sdg_train.organize_basicwriter_outputs(semantic_folder_name="mask")
        tools.SIMU_APP.close()
        

if __name__ == "__main__":
    main()
