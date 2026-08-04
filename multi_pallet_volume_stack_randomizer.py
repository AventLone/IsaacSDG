#!/usr/bin/env python3
"""
Standalone Isaac Sim 6.0 script: stack random boxes on multiple pallets at the same time
using isaacsim.replicator.behavior.behaviors.VolumeStackRandomizer.

Run:
  ./python.sh /path/to/multi_pallet_volume_stack_randomizer.py --headless \
      --num-pallets 6 --num-captures 8 --save-usd /tmp/multi_pallet_stacks.usd

Notes:
- This uses one VolumeStackRandomizer behavior per pallet stack surface.
- The reset/setup/run actions are published for all surfaces with asyncio.gather(), so each
  pallet builds its own stack in the same physics scene.
"""

import argparse
import asyncio
import inspect
import os
from pathlib import Path
from typing import Iterable

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--num-pallets", type=int, default=6)
parser.add_argument("--grid-cols", type=int, default=3)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--min-boxes", type=int, default=8)
parser.add_argument("--max-boxes", type=int, default=24)
parser.add_argument("--drop-height", type=float, default=2.0)
parser.add_argument("--num-captures", type=int, default=6)
parser.add_argument("--output-dir", type=str, default="_out_multi_pallet_vsr")
parser.add_argument("--save-usd", type=str, default="")
parser.add_argument(
    "--pallet-usd",
    type=str,
    default="/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
)
parser.add_argument(
    "--box-usds-csv",
    type=str,
    default=(
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd,"
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd,"
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd"
    ),
)
args, unknown = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import carb.settings  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.replicator.behavior.behaviors import VolumeStackRandomizer  # noqa: E402
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS  # noqa: E402
from isaacsim.replicator.behavior.utils.behavior_utils import (  # noqa: E402
    add_behavior_script_with_parameters_async,
    publish_event_and_wait_for_completion_async,
)
from isaacsim.storage.native import get_assets_root_path_async  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402


# Make SDG deterministic and standalone-friendly.
settings = carb.settings.get_settings()
settings.set("/omni/replicator/captureOnPlay", False)
settings.set("/omni/replicator/asyncRendering", False)
settings.set("/app/asyncRendering", False)
settings.set("rtx/post/dlss/execMode", 2)  # DLSS Quality

# The UI extension is not needed for headless; the core behavior extension is.
enable_extension("omni.kit.scripting")
enable_extension("isaacsim.replicator.behavior")
for _ in range(5):
    simulation_app.update()


def abs_asset(root: str, maybe_relative: str) -> str:
    if maybe_relative.startswith("omniverse://") or maybe_relative.startswith("file://"):
        return maybe_relative
    if os.path.isabs(maybe_relative) and maybe_relative.endswith(".usd") and Path(maybe_relative).exists():
        return maybe_relative
    return root + maybe_relative


def xform_set_translation(prim: Usd.Prim, xyz: Iterable[float]) -> None:
    xform = UsdGeom.Xformable(prim)
    if not prim.HasAttribute("xformOp:translate"):
        xform.AddTranslateOp()
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*xyz))


def xform_set_scale(prim: Usd.Prim, xyz: Iterable[float]) -> None:
    xform = UsdGeom.Xformable(prim)
    if not prim.HasAttribute("xformOp:scale"):
        xform.AddScaleOp()
    prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*xyz))


def create_physics_scene(stage: Usd.Stage) -> None:
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)


def create_invisible_stack_material(stage: Usd.Stage) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, "/World/Materials/InvisibleStackSurface")
    shader: UsdShade.Shader = UsdShade.Shader.Define(stage, "/World/Materials/InvisibleStackSurface/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.8, 1.0))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.01)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def create_stack_surface(
    stage: Usd.Stage,
    path: str,
    center_xy: tuple[float, float],
    top_z: float = 0.155,
    footprint_xy: tuple[float, float] = (1.20, 1.00),
    thickness: float = 0.02,
    material: UsdShade.Material | None = None,
) -> Usd.Prim:
    """A thin Cube Gprim used as the behavior's stacking area and collision support."""
    cube = UsdGeom.Cube.Define(stage, path)
    prim = cube.GetPrim()
    cube.CreateSizeAttr(1.0)
    xform_set_translation(prim, (center_xy[0], center_xy[1], top_z - thickness / 2.0))
    xform_set_scale(prim, (footprint_xy[0], footprint_xy[1], thickness))

    # Let the stack surface physically support dropped boxes.
    UsdPhysics.CollisionAPI.Apply(prim)
    if material is not None:
        UsdShade.MaterialBindingAPI(prim).Bind(material)
    return prim


def create_pallet(stage: Usd.Stage, path: str, usd_path: str, pos: tuple[float, float, float]) -> Usd.Prim:
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(usd_path)
    xform_set_translation(prim, pos)
    return prim


def create_camera_and_lights(stage: Usd.Stage) -> str:
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(600.0)

    key = UsdLux.SphereLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(65000.0)
    key.CreateRadiusAttr(1.5)
    xform_set_translation(key.GetPrim(), (0.0, -4.0, 6.0))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam_prim = camera.GetPrim()
    xform_set_translation(cam_prim, (3.8, -6.5, 3.3))
    # Aim roughly at the pallet grid center.
    xform = UsdGeom.Xformable(cam_prim)
    if not cam_prim.HasAttribute("xformOp:orient"):
        xform.AddOrientOp()
    # Quaternion found by a rough manual front/top view; adjust as needed for your warehouse.
    cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(0.73, 0.49, 0.38, 0.29))
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateFocusDistanceAttr(7.0)
    return str(cam_prim.GetPath())


async def attach_volume_stack_randomizer_async(
    surface_prim: Usd.Prim,
    box_assets_csv: str,
    num_range: tuple[int, int],
    drop_height: float,
    seed: int | None,
) -> None:
    script_path = inspect.getfile(VolumeStackRandomizer)
    ns = f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}"
    params = {
        f"{ns}:includeChildren": False,
        f"{ns}:assets:csv": box_assets_csv,
        f"{ns}:assets:numRange": Gf.Vec2i(num_range[0], num_range[1]),
        f"{ns}:dropHeight": float(drop_height),
        f"{ns}:renderSimulation": False,
        f"{ns}:preserveSimulationState": True,
        f"{ns}:removeRigidBodyDynamics": True,
    }
    if seed is not None:
        params[f"{ns}:seed"] = int(seed)
    await add_behavior_script_with_parameters_async(surface_prim, script_path, params)


async def run_one_action_async(surface_prim: Usd.Prim, action: str, expected_state: str, max_wait: int) -> bool:
    return await publish_event_and_wait_for_completion_async(
        publish_payload={"prim_path": surface_prim.GetPath(), "action": action},
        expected_payload={"prim_path": surface_prim.GetPath(), "state_name": expected_state},
        publish_event_name=VolumeStackRandomizer.EVENT_NAME_IN,
        subscribe_event_name=VolumeStackRandomizer.EVENT_NAME_OUT,
        max_wait_updates=max_wait,
    )


async def run_all_pallet_stacks_async(surface_prims: list[Usd.Prim]) -> None:
    # Barrier-style orchestration: all reset, then all setup, then all simulate/run.
    for action, state, wait in [
        ("reset", "RESET", 100),
        ("setup", "SETUP", 2000),
        ("run", "FINISHED", 20000),
    ]:
        print(f"[VSR] {action}: {len(surface_prims)} pallets")
        results = await asyncio.gather(
            *(run_one_action_async(p, action, state, wait) for p in surface_prims),
            return_exceptions=True,
        )
        bad = [str(p.GetPath()) for p, ok in zip(surface_prims, results) if ok is not True]
        if bad:
            raise RuntimeError(f"VolumeStackRandomizer action '{action}' failed for: {bad}. Results: {results}")


async def capture_async(camera_path: str, num_captures: int, output_dir: str) -> None:
    rp = rep.create.render_product(camera_path, (1280, 720))
    writer = rep.writers.get("BasicWriter")
    writer.initialize(output_dir=os.path.abspath(output_dir), rgb=True, distance_to_image_plane=True, colorize_depth=True)
    writer.attach(rp)
    rep.orchestrator.set_capture_on_play(False)

    timeline = omni.timeline.get_timeline_interface()
    for i in range(num_captures):
        print(f"[Capture] frame {i}")
        timeline.play()
        await omni.kit.app.get_app().next_update_async()
        timeline.pause()
        timeline.commit()
        await rep.orchestrator.step_async(rt_subframes=32, delta_time=0.0)

    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()


async def main_async() -> None:
    rng = np.random.default_rng(args.seed)
    assets_root = await get_assets_root_path_async()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root. Check your Isaac asset/Nucleus setup.")

    pallet_usd = abs_asset(assets_root, args.pallet_usd)
    box_csv = ",".join(abs_asset(assets_root, p.strip()) for p in args.box_usds_csv.split(",") if p.strip())

    await omni.usd.get_context().new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Pallets", "Xform")
    stage.DefinePrim("/World/StackSurfaces", "Xform")
    stage.DefinePrim("/World/Lights", "Xform")
    stage.DefinePrim("/World/Materials", "Scope")
    create_physics_scene(stage)
    mat = create_invisible_stack_material(stage)

    spacing_x, spacing_y = 1.65, 1.55
    rows = int(np.ceil(args.num_pallets / args.grid_cols))
    grid_w = (args.grid_cols - 1) * spacing_x
    grid_h = (rows - 1) * spacing_y

    surfaces = []
    for i in range(args.num_pallets):
        r, c = divmod(i, args.grid_cols)
        x = c * spacing_x - grid_w / 2.0
        y = r * spacing_y - grid_h / 2.0
        create_pallet(stage, f"/World/Pallets/Pallet_{i:02d}", pallet_usd, (x, y, 0.0))
        surface = create_stack_surface(
            stage,
            f"/World/StackSurfaces/Surface_{i:02d}",
            center_xy=(x, y),
            top_z=0.155,
            footprint_xy=(1.18, 0.95),
            material=mat,
        )
        surfaces.append(surface)

    camera_path = create_camera_and_lights(stage)
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

    # Attach one behavior instance per surface, with independent seed per pallet.
    await asyncio.gather(*(attach_volume_stack_randomizer_async(
        surface,
        box_assets_csv=box_csv,
        num_range=(args.min_boxes, args.max_boxes),
        drop_height=args.drop_height,
        seed=int(rng.integers(0, 2**31)))
        for surface in surfaces)
    )

    await run_all_pallet_stacks_async(surfaces)

    if args.save_usd:
        stage.GetRootLayer().Export(args.save_usd)
        print(f"[USD] saved to {args.save_usd}")

    if args.num_captures > 0:
        await capture_async(camera_path, args.num_captures, args.output_dir)
        print(f"[Writer] images saved to {os.path.abspath(args.output_dir)}")


try:
    asyncio.get_event_loop().run_until_complete(main_async())
finally:
    simulation_app.close()
