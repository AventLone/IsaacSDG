from tools.logger import logging, logging_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging_handler)

import os, random, asyncio, sys, carb.settings
from tqdm import tqdm
from typing import Optional
import omni.replicator.core as rep
from writers import CocoInstanceSegWriter
from randomization.evnet_randomizer import CameraAndLightRandomizer, MaterialRandomizer
from randomization import stack_boxes_on_pallet_async
from sample import PermuAndCombi
from tools import common

from omni.kit.async_engine import run_coroutine
from datetime import datetime
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
from tools.path_generation import generate_rectangle_path
from omni import usd

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
    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    def __init__(self, environment_urls: list, dome_texture_urls: list, 
                 obj_urls_dir: str, img_resolution: tuple[int, int], 
                 stacking_cols: int, stacking_rows: int,
                 camera_height: float, pallet_with_goods_count: Optional[int],
                 boxes_urls_and_weights: Optional[list]=None,
                 save_path=None) -> None:
        stage_utils.create_new_stage()
        prims_utils.create_prim("/World")
        self._environment_prim_path = "/World/Environment"
        self._dome_prim_path = "/World/Lights/DomeLight"

        stage_utils.add_reference_to_stage(
            usd_path="/home/avent/Desktop/IsaacAssets/Collected_warehouse_trailer/Environments/warehouse_trailer.usd",
            prim_path=self._environment_prim_path
        )
        common.set_world_trasform(prim=self._environment_prim_path,
                                  translation=[-7.6, 4.85, 0.0], orientation=common.yaw2quat(90.0))

        self._dome_texture_urls = dome_texture_urls
        self._environment_urls = environment_urls
        self._dome_prim = prims_utils.create_prim(prim_path=self._dome_prim_path, prim_type="DomeLight",
                                            attributes={"inputs:intensity": 1000.0,
                                                        "inputs:texture:file": dome_texture_urls[0]})
        # 2. Get the specific texture attribute
        # Note: The attribute name is 'inputs:texture:file'
        self._dome_texture = self._dome_prim.GetAttribute("inputs:texture:file")

        self._prim_paths = common.load_usds(obj_urls_dir)
        self._pac = PermuAndCombi(self._prim_paths)
        self._img_resolution = img_resolution

        self._pac.create_columns(columns=stacking_cols, direction='x')
        # Randomizer
        # dimensions_x, dimensions_y, _ = common.get_dimensions(self._pac.prim_path)
        # dimensions_x = max(dimensions_x, 1.5)
        # dimensions_y = max(dimensions_y, 1.5)
        dimensions_x, dimensions_y = 1.25 * stacking_cols, 1.8
        dimensions_list = [(dimensions_x + 0.2, dimensions_y + 0.2)]
        for i in range(1, 4):
            amplifier = i * 0.4 + 1.0
            dimensions_list.append((dimensions_x * (amplifier - 0.2), dimensions_y * (amplifier + 0.2)))
        camera_path = generate_rectangle_path(camera_height, dimensions_list)
        self._camera_light_randomizer = CameraAndLightRandomizer(camera_path)
        self._material_randomizer = MaterialRandomizer(self._pac.prim_path)

        self._render_product = rep.create.render_product(
            camera=self._camera_light_randomizer.camera, resolution=img_resolution)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        save_at = f"generated_data/{timestamp}" if save_path is None else f"{save_path}/{timestamp}"
        data_save_dir = os.path.join(os.getcwd(), save_at)

        self._writer = CocoInstanceSegWriter(output_dir=data_save_dir)
        self._writer.attach(self._render_product, trigger=self._camera_light_randomizer.camera_trigger)

        self._pallet_with_goods_count = pallet_with_goods_count

        self._counts = stacking_cols * (stacking_rows - 1)
        run_coroutine(self._pac.stack(columns=stacking_cols, rows=stacking_rows))

        logger.info("Preparing pallets with goods...")
        self._loads_with_goods: asyncio.Future = None if boxes_urls_and_weights is None else run_coroutine(
            prepare_loads_with_goods(self._prim_paths, pallet_with_goods_count, boxes_urls_and_weights)) # type: ignore
        
    def _random_dome_texture(self):
        display_dome = random.choice([True, False])
        common.make_visiable(self._environment_prim_path, display_dome)
        if not display_dome:
            texture = random.choice(self._dome_texture_urls)
            self._dome_texture.Set(texture)

    async def generate(self, sample_interval: int):
        # Step 1: Prepare loads with goods
        pca_list: list[PermuAndCombi] = await self._loads_with_goods

        this_stage = stage_utils.get_current_stage()

        # Step 2: Collect stacking pallets (without goods)
        frames_generated_total = self._camera_light_randomizer.frames_generated * (self._counts // sample_interval +
                                                                                   len(self._prim_paths) * self._pallet_with_goods_count) \
        if self._pallet_with_goods_count is not None else self._camera_light_randomizer.frames_generated * (self._counts // sample_interval)
        logger.info(f"Preparation done. SDG is starting, {frames_generated_total} images will be generated.")

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
