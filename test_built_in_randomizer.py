from tools.common import SIMU_APP

# import asyncio
import inspect
import os

import isaacsim.core.experimental.utils.semantics as semantics_utils
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
from isaacsim.replicator.behavior.behaviors import (
    LightRandomizer,
    LocationRandomizer,
    LookAtBehavior,
    RotationRandomizer,
    TextureRandomizer,
    VolumeStackRandomizer
)
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from isaacsim.replicator.behavior.utils.behavior_utils import (
    add_behavior_script_with_parameters_async,
    publish_event_and_wait_for_completion_async
)
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, UsdGeom, Usd
from isaacsim.core.utils import stage as stage_utils


async def setup_and_run_stacking_simulation_async(prim: Usd.Prim, seed: int | None = None):
    # STACK_ASSETS_CSV = (
    #     "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd,"
    #     "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd,"
    #     "/Isaac/Props/KLT_Bin/small_KLT_visual.usd,"
    # )
    STACK_ASSETS_CSV = (
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd,"
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd,"
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd,"
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd"
    )

    # Add the behavior script with custom parameters
    script_path = inspect.getfile(VolumeStackRandomizer)
    parameters = {
        f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:assets:csv": STACK_ASSETS_CSV,
        f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:assets:numRange": Gf.Vec2i(2, 20),
        f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:renderSimulation": True
    }
    if seed is not None:
        parameters[f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:seed"] = seed
    await add_behavior_script_with_parameters_async(prim, script_path, parameters)

    # Helper function to handle publishing and waiting for events
    async def handle_event(action, expected_state, max_wait):
        return await publish_event_and_wait_for_completion_async(
            publish_payload={"prim_path": prim.GetPath(), "action": action},
            expected_payload={"prim_path": prim.GetPath(), "state_name": expected_state},
            publish_event_name=VolumeStackRandomizer.EVENT_NAME_IN,
            subscribe_event_name=VolumeStackRandomizer.EVENT_NAME_OUT,
            max_wait_updates=max_wait
        )

    # Define and execute the stacking simulation steps
    actions = [("reset", "RESET", 10), ("setup", "SETUP", 500), ("run", "FINISHED", 10000)]
    for action, state, wait in actions:
        print(f"Executing '{action}' and waiting for state '{state}'...")
        if not await handle_event(action, state, wait):
            print(f"Failed to complete '{action}' with state '{state}'.")
            if action == "run":
                print("Requesting reset before continuing.")
                await handle_event("reset", "RESET", 2000)
            return

    print("Stacking simulation finished.")


async def setup_texture_randomizer_async(prim, seed: int | None = None):
    TEXTURE_ASSETS_CSV = (
        "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg,"
        "/Isaac/Materials/Textures/Patterns/nv_wood_boards_brown.jpg,"
        "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg,"
    )

    script_path = inspect.getfile(TextureRandomizer)
    parameters = {
        f"{EXPOSED_ATTR_NS}:{TextureRandomizer.BEHAVIOR_NS}:interval": 5,
        f"{EXPOSED_ATTR_NS}:{TextureRandomizer.BEHAVIOR_NS}:textures:csv": TEXTURE_ASSETS_CSV,
    }
    if seed is not None:
        parameters[f"{EXPOSED_ATTR_NS}:{TextureRandomizer.BEHAVIOR_NS}:seed"] = seed
    await add_behavior_script_with_parameters_async(prim, script_path, parameters)


async def setup_light_behaviors_async(prim, light_seed: int | None = None, location_seed: int | None = None):
    # Light randomization
    light_script_path = inspect.getfile(LightRandomizer)
    light_parameters = {
        f"{EXPOSED_ATTR_NS}:{LightRandomizer.BEHAVIOR_NS}:interval": 4,
        f"{EXPOSED_ATTR_NS}:{LightRandomizer.BEHAVIOR_NS}:range:intensity": Gf.Vec2f(20000, 120000),
    }
    if light_seed is not None:
        light_parameters[f"{EXPOSED_ATTR_NS}:{LightRandomizer.BEHAVIOR_NS}:seed"] = light_seed
    await add_behavior_script_with_parameters_async(prim, light_script_path, light_parameters)

    # Location randomization
    location_script_path = inspect.getfile(LocationRandomizer)
    location_parameters = {
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:interval": 2,
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:minPosition": Gf.Vec3d(-1.25, -1.25, 0.0),
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:maxPosition": Gf.Vec3d(1.25, 1.25, 0.0),
    }
    if location_seed is not None:
        location_parameters[f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:seed"] = location_seed
    await add_behavior_script_with_parameters_async(prim, location_script_path, location_parameters)


async def setup_target_asset_behaviors_async(prim, rotation_seed: int | None = None, location_seed: int | None = None):
    # Rotation randomization
    rotation_script_path = inspect.getfile(RotationRandomizer)
    rotation_parameters = {}
    if rotation_seed is not None:
        rotation_parameters[f"{EXPOSED_ATTR_NS}:{RotationRandomizer.BEHAVIOR_NS}:seed"] = rotation_seed
    await add_behavior_script_with_parameters_async(prim, rotation_script_path, rotation_parameters)

    # Location randomization
    location_script_path = inspect.getfile(LocationRandomizer)
    location_parameters = {
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:interval": 3,
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:minPosition": Gf.Vec3d(-0.2, -0.2, -0.2),
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:maxPosition": Gf.Vec3d(0.2, 0.2, 0.2),
    }
    if location_seed is not None:
        location_parameters[f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:seed"] = location_seed
    await add_behavior_script_with_parameters_async(prim, location_script_path, location_parameters)


async def setup_camera_behaviors_async(prim, target_prim_path):
    # Look at behavior following the target asset
    script_path = inspect.getfile(LookAtBehavior)
    parameters = {
        f"{EXPOSED_ATTR_NS}:{LookAtBehavior.BEHAVIOR_NS}:targetPrimPath": target_prim_path,
    }
    await add_behavior_script_with_parameters_async(prim, script_path, parameters)


async def setup_writer_and_capture_data_async(camera_path, num_captures):
    # Create the writer and the render product
    rp = rep.create.render_product(camera_path, (512, 512))
    writer = rep.writers.get("BasicWriter")
    output_directory = os.path.join(os.getcwd(), "_out_behaviors_sdg")
    print(f"output_directory: {output_directory}")
    writer.initialize(output_dir=output_directory, rgb=True, distance_to_image_plane=True, colorize_depth=True)
    writer.attach(rp)

    # Disable capture on play, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Use the timeline to control the behavior scripts execution and frame captures
    timeline = omni.timeline.get_timeline_interface()

    # Start the SDG pipeline
    for i in range(num_captures):
        # Advance the timeline with one update and then pause it to avoid triggering the behavior scripts by the step_async internal updates
        timeline.play()
        await omni.kit.app.get_app().next_update_async()
        timeline.pause()
        timeline.commit()

        # Capture the frame
        print(f"Capturing frame {i} at time {timeline.get_current_time():.4f}")
        await rep.orchestrator.step_async(rt_subframes=32, delta_time=0.0)

    # Stop the timeline (and trigger the behavior scripts to stop)
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()

    # Make sure all the frames are written from the backend queue and free the rendering resources
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()


async def run_example_async(num_captures, seed: int | None = None):
    STAGE_URL = "/Isaac/Samples/Replicator/Stage/warehouse_pallets_behavior_scripts.usd"
    PALLETS_ROOT_PATH = "/Root/Pallets"
    LIGHTS_ROOT_PATH = "/Root/Lights"
    CAMERA_PATH = "/Root/Camera_01"
    TARGET_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd"
    TARGET_ASSET_PATH = "/Root/Target"
    TARGET_ASSET_LABEL = "power_drill"
    TARGET_ASSET_LOCATION = (-1.5, 5.5, 1.5)

    # Generate unique seeds per behavior instance to ensure determinism regardless of execution order
    if seed is not None:
        seed_rng = np.random.default_rng(seed)
        stacking_seed = int(seed_rng.integers(0, 2**31))
        texture_seed = int(seed_rng.integers(0, 2**31))
        light_intensity_seed = int(seed_rng.integers(0, 2**31))
        light_location_seed = int(seed_rng.integers(0, 2**31))
        target_rotation_seed = int(seed_rng.integers(0, 2**31))
        target_location_seed = int(seed_rng.integers(0, 2**31))
    else:
        stacking_seed = texture_seed = None
        light_intensity_seed = light_location_seed = None
        target_rotation_seed = target_location_seed = None

    # Open stage
    # assets_root_path = await get_assets_root_path_async()
    assets_root_path = "/home/avent/Desktop/IsaacAssets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1"
    print(f"Opening stage from {assets_root_path + STAGE_URL}")
    await stage_utils.open_stage_async(assets_root_path + STAGE_URL)
    stage = stage_utils.get_current_stage()

    # Check if all required prims exist in the stage
    pallets_root_prim = stage.GetPrimAtPath(PALLETS_ROOT_PATH)
    lights_root_prim = stage.GetPrimAtPath(LIGHTS_ROOT_PATH)
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not all([pallets_root_prim.IsValid(), lights_root_prim.IsValid(), camera_prim.IsValid()]):
        print(f"Not all required prims exist in the stage.")
        return

    # Spawn the target asset at the requested location, label it with the target asset label
    target_prim = stage.DefinePrim(TARGET_ASSET_PATH, "Xform")
    target_prim.GetReferences().AddReference(assets_root_path + TARGET_ASSET_URL)
    if not target_prim.HasAttribute("xformOp:translate"):
        UsdGeom.Xformable(target_prim).AddTranslateOp()
    target_prim.GetAttribute("xformOp:translate").Set(TARGET_ASSET_LOCATION)
    semantics_utils.remove_all_labels(target_prim, include_descendants=True)
    semantics_utils.add_labels(target_prim, labels=[TARGET_ASSET_LABEL], taxonomy="class")

    # Setup and run the stacking simulation before capturing the data
    # Note: Physics simulation is non-deterministic, final positions may vary
    await setup_and_run_stacking_simulation_async(pallets_root_prim, seed=stacking_seed)

    # Setup texture randomizer
    await setup_texture_randomizer_async(pallets_root_prim, seed=texture_seed)

    # Setup the light behaviors
    await setup_light_behaviors_async(
        lights_root_prim, light_seed=light_intensity_seed, location_seed=light_location_seed
    )

    # Setup the target asset behaviors
    await setup_target_asset_behaviors_async(
        target_prim, rotation_seed=target_rotation_seed, location_seed=target_location_seed
    )

    # Setup the camera behaviors
    await setup_camera_behaviors_async(camera_prim, str(target_prim.GetPath()))

    # Setup the writer and capture the data, behavior scripts are triggered by running the timeline
    await setup_writer_and_capture_data_async(camera_path=camera_prim.GetPath(), num_captures=num_captures)


# asyncio.ensure_future(run_example_async(num_captures=6))

SIMU_APP.run_coroutine(run_example_async(num_captures=6))

while SIMU_APP.is_running():
    SIMU_APP.update()

SIMU_APP.close()