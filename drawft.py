import tools
from tools import common
import os, random, asyncio, sys, carb.settings
from tqdm import tqdm
from typing import Optional
import omni.replicator.core as rep
from writers import CocoInstanceSegWriter
from randomization.evnet_randomizer import CameraAndLightRandomizer, MaterialRandomizer
from randomization import stack_boxes_on_pallet_async, volume_stack
from sample import PermuAndCombi
from tools import common

from omni.kit.async_engine import run_coroutine
from datetime import datetime
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
from tools.path_generation import generate_rectangle_path
from omni import usd
from tools.logger import LOGGER

async def prepare_loads_with_goods(prim_paths: list[str], loads_count: int, boxes_urls_and_weights: list):
    pca_list = []
    idx = 0.0
   
    with tqdm(total=len(prim_paths) * loads_count, desc="Preparation Progress", unit="Pallet", file=sys.stdout) as pbar:
        for prim_path in prim_paths:
            pca = PermuAndCombi([prim_path])
            pca.set_pose(translation=(0.0, 999.0 + idx, 0.0), yaw=0.0)
            pca.create_columns(columns=loads_count, direction='x', gap=0.2)

            for col in pca.column_prims:
                num_boxes = random.randint(10, 50)
                await stack_boxes_on_pallet_async(pallet_prim=col, boxes_urls_and_weights=boxes_urls_and_weights,
                                                  num_boxes=num_boxes, overhang=0.1)
                pbar.update(1)

            pca_list.append(pca)
            idx += 3.6
    return pca_list


class SDG:
    # Disable capture on play and async rendering
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
    carb.settings.get_settings().set("/app/asyncRendering", False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 1) # (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)

    def __init__(self, stage_url: str, dome_texture_urls: list, 
                 obj_urls_dir: str,
                 stacking_cols: int, stacking_rows: int,
                 camera_height: float, pallet_with_goods_count: Optional[int],
                 boxes_urls_and_weights: Optional[list]=None,
                 save_path=None) -> None:
        if not stage_utils.open_stage(stage_url):
            raise RuntimeError(f"Failed to open {stage_url}")
        
        self._dome_prim_path = "/World/Lights/DomeLight"
        self._dome_texture_urls = dome_texture_urls
        self._dome_prim = prims_utils.create_prim(prim_path=self._dome_prim_path, prim_type="DomeLight",
                                            attributes={"inputs:intensity": 1000.0,
                                                        "inputs:texture:file": dome_texture_urls[0]})
        # 2. Get the specific texture attribute
        self._dome_texture = self._dome_prim.GetAttribute("inputs:texture:file")

        self._prim_paths = common.load_usds(obj_urls_dir)
        self._pac = PermuAndCombi(self._prim_paths)

        self._pac.create_columns(columns=stacking_cols, direction='x')


        self._camera_light_randomizer = CameraAndLightRandomizer(camera_path)
        self._material_randomizer = MaterialRandomizer(self._pac.prim_path)

        self._render_product = rep.create.render_product(camera=self._camera_light_randomizer.camera, 
                                                         resolution=(504, 504))

        # Set up writer
        timestamp = datetime.now().strftime("%Y.%m.%d-%H:%M")
        save_at = f"generated_data/{timestamp}" if save_path is None else f"{save_path}/{timestamp}"
        data_save_dir = os.path.join(os.getcwd(), save_at)
        self._writer = CocoInstanceSegWriter(output_dir=data_save_dir)
        self._writer.attach(self._render_product)

        self._pallet_with_goods_count = pallet_with_goods_count

        self._counts = stacking_cols * (stacking_rows - 1)
        run_coroutine(self._pac.stack(columns=stacking_cols, rows=stacking_rows))

        LOGGER.info("Preparing pallets with goods...")
        self._loads_with_goods: asyncio.Future = None if boxes_urls_and_weights is None else run_coroutine(
            prepare_loads_with_goods(self._prim_paths, pallet_with_goods_count, boxes_urls_and_weights))
        
    def _random_dome_texture(self):
        display_dome = random.choice([True, False])
        common.make_visiable(self._environment_prim_path, False)
        if not display_dome:
            texture = random.choice(self._dome_texture_urls)
            self._dome_texture.Set(texture)

    async def generate(self, sample_interval: int):
        # Step 1: Prepare loads with goods
        if self._loads_with_goods is not None:
            pca_list: list[PermuAndCombi] = await self._loads_with_goods

        this_stage = stage_utils.get_current_stage()

        # Step 2: Collect stacking pallets (without goods)
        frames_generated_total = self._camera_light_randomizer.frames_generated * (self._counts // sample_interval +
                                                                                   len(self._prim_paths) * self._pallet_with_goods_count) \
        if self._pallet_with_goods_count is not None else self._camera_light_randomizer.frames_generated * (self._counts // sample_interval)
        LOGGER.info(f"Preparation done. SDG is starting, {frames_generated_total} images will be generated.")

        with tqdm(total=frames_generated_total, desc="SDG Progress", unit=" Frames", file=sys.stdout) as pbar:
            for count in range(self._counts):
                await self._pac.run()
                if (count + 1) % sample_interval == 0:
                    for frame in range(self._camera_light_randomizer.frames_generated):
                        self._camera_light_randomizer.randomize_camera()
                        pbar.update(1)
                        if frame % 3 == 0:
                            self._material_randomizer.randomize_material()
                        if frame % 4 == 0:
                            self._random_dome_texture()
                        if frame % 6 == 0:
                            self._camera_light_randomizer.randomize_light()

                        await rep.orchestrator.step_async(rt_subframes=10)
            this_stage.RemovePrim(self._pac.prim_path)

            # Step 3: Collect data of pallets with goods, one by one
            targe_prim_path_parent = "/World/Target"
            prims_utils.create_prim(targe_prim_path_parent)
            target_prim_path = f"{targe_prim_path_parent}/obj"

            if self._loads_with_goods is not None:
                for pca in pca_list:
                    for col in pca.column_prims:
                        usd.duplicate_prim(this_stage, prim_path=str(col.GetPrimPath()), path_to=target_prim_path)
                        common.set_local_trasform(target_prim_path, [0.0, 0.0, 0.0])

                        for frame in range(self._camera_light_randomizer.frames_generated):
                            self._camera_light_randomizer.randomize_camera()
                            pbar.update(1)
                            if frame % 4 == 0:
                                self._random_dome_texture()
                            if frame % 10 == 0:
                                self._camera_light_randomizer.randomize_light()
                            await rep.orchestrator.step_async(rt_subframes=10)
                        this_stage.RemovePrim(target_prim_path)

        await rep.orchestrator.wait_until_complete_async()

        self._writer.detach()
        self._render_product.destroy()   # type: ignore


def main() -> None:
    FORK_CAMERA_HEIGHT = 0.75
    CAMERA_HEIGHT_TRAIN, CAMERA_HEIGHT_VAL = 0.75, 1.0
    CAMERA_RADIUSES_TRAIN = [2.2, 3.2, 4.2]
    CAMERA_RADIUSES_VAL = [3.0]
    PALLET_WITH_GOODS_COLUMNS = 10

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
                    obj_urls_dir="/home/avent/Desktop/pallets",
                    boxes_urls_and_weights=boxes_urls_and_weights,
                    img_resolution=(600, 600),
                    stacking_cols=6, stacking_rows=6,
                    camera_height=CAMERA_HEIGHT_TRAIN,
                    pallet_with_goods_count=5,
                    save_path="/media/avent/DATA/generated_data/train")
    try:
        tools.SIMU_APP.run_coroutine(sdg_train.generate(sample_interval=3))

        # sdg_val = SDG(environment_urls=environment_urls, dome_texture_urls=dome_textures,
        #               obj_urls_dir="/home/avent/Desktop/pallets",
        #               boxes_urls_and_weights=boxes_urls_and_weights,
        #               img_resolution=(504, 504),
        #               stacking_cols=3, stacking_rows=3,
        #               camera_height=CAMERA_HEIGHT_VAL, camera_orbit_radiuses=CAMERA_RADIUSES_VAL,
        #               pallet_with_goods_count=2,
        #               save_path="/home/avent/Desktop/generated_data/valid")
        # simu_app.run_coroutine(sdg_val.generate(sample_interval=3))
    except Exception as e:   # catches all built-in exceptions
        LOGGER.error(f"Something went wrong: {e}")
    finally:
        tools.SIMU_APP.close()

if __name__ == "__main__":
    main()
