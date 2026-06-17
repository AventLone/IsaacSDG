import numpy as np
from tools import common
import random
from isaacsim.core.utils import prims as prims_utils
from omni import usd
from pxr import UsdGeom
from tools import path_generation
import cv2
# import isaacsim.core.experimental.utils.semantics as semantics_utils
import isaacsim.core.utils.semantics as semantics_utils
from isaacsim.core.utils import bounds as bounds_utils

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysicsSchemaTools, PhysxSchema
import omni.usd, omni.physx, carb
from itertools import chain

def can_place_rect(grid: np.ndarray, x, y, w, h, margin=0):
    """
    x, y: top-left pixel coordinate
    w, h: rectangle size in pixels
    margin: extra safety distance around the rectangle
    """
    H, W = grid.shape[:2]

    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(W, x + w + margin)
    y1 = min(H, y + h + margin)

    if x < 0 or y < 0 or x + w > W or y + h > H:
        return False

    roi = grid[y0:y1, x0:x1]
    return np.all(roi == 0)    # Valid only if the whole region is free


def scatter(prim_pathes: list[str], num_for_each, lower_bound, upper_bound, origin=(0, 0, 0)) -> np.ndarray:
    this_stage = prims_utils.get_current_stage()
    scatter_prim_path = "/World/Objects/Scatter"
    if not prims_utils.get_prim_at_path(scatter_prim_path).IsValid():
        prims_utils.create_prim(scatter_prim_path)

    resolution = 0.02

    rect_sizes = []
    for prim in prim_pathes:
        dim_x, dim_y, _ = common.get_dimensions(prim)
        rect_sizes.append((round(dim_x / resolution), round(dim_y / resolution)))
    rect_sizes = rect_sizes * num_for_each

    H = round((upper_bound[1] - lower_bound[1]) / resolution)
    W = round((upper_bound[0] - lower_bound[0]) / resolution)

    grid = np.zeros((H, W), dtype=np.uint8)
    for i, (w, h) in enumerate(rect_sizes):
        for _ in range(666):
            x = random.randint(0, W - w)
            y = random.randint(0, H - h)

            if can_place_rect(grid, x, y, w, h):
                grid[y: y + h, x: x + w] = 255   # place rect on the grid

                # convert pixel coordinate to real-world coordinate
                cx_px = float(x) + float(w) / 2.0
                cy_px = float(y) + float(h) / 2.0

                world_x = lower_bound[0] + cx_px * resolution + origin[0]
                world_y = upper_bound[1] - cy_px * resolution + origin[1]
                world_z = origin[2]

                # Copy prim
                dst_prim_path = f"{scatter_prim_path}/no_{i:04d}"
                target_prim_path = prim_pathes[i % len(prim_pathes)]
                usd.duplicate_prim(this_stage, prim_path=target_prim_path, path_to=dst_prim_path)
                common.set_world_trasform(dst_prim_path, translation=(world_x, world_y, world_z))
                UsdGeom.Imageable(prims_utils.get_prim_at_path(dst_prim_path)).MakeVisible()
                
                break

    free, inflated = path_generation.inflate_obstacles(grid, resolution=resolution, camera_radius=0.2)
    waypoints = path_generation.generate_lawnmower_waypoints(free, spacing_px=30, margin_px=3, min_segment_px=20)
    path_px = path_generation.connect_waypoints_with_astar(free, waypoints)

    cv2.imwrite("images/camera_path.png", path_generation.visualize_path(grid, inflated, path_px, waypoints=waypoints))

    return grid, path_generation.pixel_path_to_world_poses(path_px,
                                                           (origin[0] + lower_bound[0], origin[1] + upper_bound[1]),
                                                           resolution)


class Pile:
    index = 0

    @staticmethod
    def generate(prim_path: str, parent_prim_path: str, num: int):
        this_stage = prims_utils.get_current_stage()
        _, _, height = common.get_dimensions(prim_path)
        piles_prim_path = f"{parent_prim_path}/piles_{Pile.index}"
        prims_utils.create_prim(piles_prim_path, position=(100.0, 100.0, 0.0))
        for i in range(num):
            component_prim_path = f"{piles_prim_path}/componnet_{i}"
            usd.duplicate_prim(stage=this_stage, prim_path=prim_path, path_to=component_prim_path)
            new_component_pos = (random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03), height * i)
            common.set_local_trasform(component_prim_path, new_component_pos, common.yaw2quat(random.uniform(-10.0, 10.0)))
            # common.make_visible(component_prim_path, True)
            # semantics_utils.add_labels(component_prim_path, labels=[category_label])
        # common.make_visible(piles_prim_path, False)
        Pile.index += 1
        return piles_prim_path

    
class LoadedPallet:
    index = 0
    assets_urls_and_weights = None
    box_prim_paths_and_weights: list[tuple[str, float]] = []
    default_material: UsdShade.Material = None
    physics_material: UsdPhysics.MaterialAPI = None
    
    @staticmethod
    def set_assets(assets_urls_and_weights: list[tuple[str, float]]):
        LoadedPallet.assets_urls_and_weights = assets_urls_and_weights

        assets_prim_path = "/Assets"
        assets_boxes_path = f"{assets_prim_path}/Boxes"
        assets_piles_path = f"{assets_prim_path}/Piles"
        assets_loaded_path = f"{assets_prim_path}/Loaded"
        if not prims_utils.is_prim_path_valid(assets_prim_path):
            prims_utils.create_prim(assets_prim_path)

        if not prims_utils.is_prim_path_valid(assets_boxes_path):
            prims_utils.create_prim(assets_boxes_path, prim_type="Scope")

        if not prims_utils.is_prim_path_valid(assets_piles_path):
            prims_utils.create_prim(assets_piles_path, prim_type="Scope")

        if not prims_utils.is_prim_path_valid(assets_loaded_path):
            prims_utils.create_prim(assets_loaded_path, prim_type="Scope")

        # for i, (box_url, weight) in enumerate(assets_urls_and_weights):
        #     prim_path = f"{assets_boxes_path}/box_{i}"
        #     prims_utils.add_reference_to_stage(usd_path=box_url, prim_path=prim_path)
        #     LoadedPallet.box_prim_paths_and_weights.append((prim_path, weight))
        
        # Create a custom physics material to allow the boxes to easily slide into stacking positions
        this_stage = prims_utils.get_current_stage()
        physics_material_prim_path = "/VolumeStackLooks"
        material_path = "/VolumeStackLooks/PhysicsMaterial"
        if not prims_utils.is_prim_path_valid(physics_material_prim_path):
            prims_utils.create_prim("/VolumeStackLooks", prim_type="Scope")
            LoadedPallet.default_material = UsdShade.Material.Define(this_stage, material_path)
        else:
            material_prim = prims_utils.get_prim_at_path(material_path)
            LoadedPallet.default_material = UsdShade.Material(material_prim)
        LoadedPallet.physics_material = UsdPhysics.MaterialAPI.Apply(LoadedPallet.default_material.GetPrim())
        LoadedPallet.physics_material.CreateRestitutionAttr().Set(0.0)        # Inelastic collision (no bouncing)
        LoadedPallet.physics_material.CreateStaticFrictionAttr().Set(0.001)   # Small friction to allow sliding of stationary boxes
        LoadedPallet.physics_material.CreateDynamicFrictionAttr().Set(0.001)  # Small friction to allow sliding of moving boxes


    @staticmethod
    async def generate(prim_path: str, pallet_name: str, num_boxes: int, overhang=0.1):
        if LoadedPallet.assets_urls_and_weights is None:
            raise ValueError("Please set assets_urls_and_weights before call `generate`!")
        this_stage = prims_utils.get_current_stage()

        LoadedPallet.physics_material.CreateStaticFrictionAttr().Set(0.001)
        LoadedPallet.physics_material.CreateDynamicFrictionAttr().Set(0.001)

        if not prims_utils.is_prim_path_valid(f"/Assets/Loaded/{pallet_name}"):
            prims_utils.create_prim(f"/Assets/Loaded/{pallet_name}")
        loaded_pallet_prim_path = f"/Assets/Loaded/{pallet_name}/loaded_{LoadedPallet.index}"
        # prims_utils.create_prim(loaded_pallet_prim_path, position=(150.0 + LoadedPallet.index * 3.0, 150.0, 0.0))
        prims_utils.create_prim(loaded_pallet_prim_path)
        omni.usd.duplicate_prim(this_stage, prim_path=prim_path, path_to=loaded_pallet_prim_path)
        common.set_local_trasform(loaded_pallet_prim_path, translation=(150.0 + LoadedPallet.index * 3.0, 150.0, 0.0))
       
        pallet_prim = prims_utils.get_prim_at_path(loaded_pallet_prim_path)
        # Apply the physics material to the pallet
        common.add_colliders(pallet_prim, approx_type="convexDecomposition")
        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(pallet_prim)
        mat_binding_api.Bind(LoadedPallet.default_material, UsdShade.Tokens.weakerThanDescendants, "physics")

        drop_height= 3.6
        drop_margin = 0.4
        # Create collision walls around the top of the pallet and apply the physics material to them
        collision_walls = LoadedPallet.create_collision_walls(pallet_prim,
                                                height=drop_height + drop_margin,
                                                material=LoadedPallet.default_material)

        # Create the random boxes (without physics) with the specified weights and sort them by size (volume)
        box_urls, box_weights = zip(*LoadedPallet.assets_urls_and_weights)
        rand_boxes_urls = random.choices(box_urls, weights=box_weights, k=num_boxes)
        boxes_prim = prims_utils.create_prim(f"{loaded_pallet_prim_path}/Boxes")
        box_prims = [prims_utils.add_reference_to_stage(usd_path=box_url, prim_path=f"{loaded_pallet_prim_path}/Boxes/Box_{i}")
             for i, box_url in enumerate(rand_boxes_urls)]

        box_prims.sort(key=lambda box: common.bbox_cache.ComputeLocalBound(box).GetVolume(), reverse=True)
        pallet_dimensions_x, pallet_dimensions_y, _ = common.get_dimensions(pallet_prim)

        # Simulate dropping the boxes from random poses on the pallet
        random_range_x = pallet_dimensions_x / 3.0
        random_range_y = pallet_dimensions_y / 3.0
    
        for box_prim in box_prims:
            
            common.set_local_trasform(box_prim, [random.uniform(-random_range_x, random_range_x), 
                                        random.uniform(-random_range_y, random_range_y), drop_height])
            common.add_colliders(box_prim, approx_type="convexHull")
            common.add_rigid_body_dynamics(box_prim, angular_damping=0.9)
            
            # Bind the physics material to the box (allow frictionless sliding)
            mat_binding_api = UsdShade.MaterialBindingAPI.Apply(box_prim)
            mat_binding_api.Bind(LoadedPallet.default_material, UsdShade.Tokens.weakerThanDescendants, "physics")
            await common.app_update_async()   # Wait for an app update to load the new attributes

            # Play simulation for a few frames for each box
            common.timeline.play()
            await common.wait_for(30)
            common.timeline.pause()

        # Iteratively apply forces to the boxes to move them around then pull them all together towards the pallet center
        await LoadedPallet.apply_forces_async(box_prims, pallet_prim, strength=1000)

        # Remove rigid body dynamics of the boxes until all other scenarios are completed
        for box in box_prims:
            UsdPhysics.RigidBodyAPI(box).GetRigidBodyEnabledAttr().Set(False)

        # Increase the friction to prevent sliding of the boxes on the pallet before removing the collision walls
        LoadedPallet.physics_material.CreateStaticFrictionAttr().Set(0.999)
        LoadedPallet.physics_material.CreateDynamicFrictionAttr().Set(0.999)

        # Remove collision walls
        for wall in collision_walls:
            this_stage.RemovePrim(wall.GetPath())

        # semantics_utils.remove_all_labels(boxes_prim, include_descendants=True)
        # semantics_utils.remove_labels(boxes_prim, instance_name="class", include_descendants=True)
        semantics_utils.remove_all_semantics(boxes_prim, recursive=True)
        semantics_utils.add_labels(boxes_prim, labels=["goods"])
        overhang = abs(overhang)
        if overhang > 0.0:
            common.set_local_trasform(boxes_prim, translation=[random.uniform(-overhang, overhang),
                                                        random.uniform(-overhang, overhang), 0.0])
            
        LoadedPallet.index += 1
        return loaded_pallet_prim_path
    
    @staticmethod
    def create_collision_walls(prim: Usd.Prim, height=4.6, thickness=0.1, material=None, visible=False) -> list[Usd.Prim]:
        dimensions_x, dimensions_y, _ = common.get_dimensions(prim)

        # Define the walls (name, location, size) with the specified thickness added externally to the surface and height
        walls = [("ceiling", (0.0, 0.0, height + thickness / 2), (dimensions_x, dimensions_y, thickness)),
                ("left_wall", (0.0, -(dimensions_y + thickness) / 2.0, height / 2.0), (dimensions_x, thickness, height)),
                ("right_wall", (0.0, (dimensions_y + thickness) / 2.0, height / 2.0), (dimensions_x, thickness, height)),
                ("front_wall", ((dimensions_x + thickness) / 2.0, 0.0, height / 2.0), (thickness, dimensions_y, height)),
                ("back_wall", (-(dimensions_x + thickness) / 2.0, 0.0, height / 2.0), (thickness, dimensions_y, height))]

        # Use the parent prim path to create the walls as children (use local coordinates)
        prim_path = prim.GetPrimPath()
        collision_walls = []
        for name, location, size in walls:
            scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
            cube_prim = prims_utils.create_prim(f"{prim_path}/{name}", prim_type="Cube")
            common.set_local_trasform(prim=cube_prim, translation=location, scale=scale)
            common.add_colliders(cube_prim, approx_type="convexHull")
            if not visible:
                UsdGeom.Imageable(cube_prim).MakeInvisible()
            if material is not None:
                mat_binding_api = UsdShade.MaterialBindingAPI.Apply(cube_prim)
                mat_binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
            collision_walls.append(cube_prim)
        return collision_walls


    # Slide the assets independently in perpendicular directions and then pull them all together towards the given center
    @staticmethod
    async def apply_forces_async(boxes: list[Usd.Prim], pallet, strength=550, strength_center_multiplier=2):
        common.timeline.play()
        # Get the pallet center and forward vector to apply forces in the perpendicular directions and towards the center
        pallet_tf: Gf.Matrix4d = UsdGeom.Xformable(pallet).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pallet_center = pallet_tf.ExtractTranslation()
        pallet_rot: Gf.Rotation = pallet_tf.ExtractRotation()
        force_forward = Gf.Vec3d(pallet_rot.TransformDir(Gf.Vec3d(1, 0, 0))) * strength
        force_right = Gf.Vec3d(pallet_rot.TransformDir(Gf.Vec3d(0, 1, 0))) * strength

        physx_api = omni.physx.get_physx_simulation_interface()
        stage_id = prims_utils.get_current_stage_id()
        for box_prim in boxes:
            body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
            forces = [force_forward, force_right, -force_forward, -force_right]
            for force in chain(forces, forces):
                box_tf: Gf.Matrix4d = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                box_position = carb.Float3(*box_tf.ExtractTranslation())
                physx_api.apply_force_at_pos(stage_id, body_path, carb.Float3(force), box_position, "Force")
                await common.wait_for(3)

        # Pull all box at once to the pallet center
        for box_prim in boxes:
            body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
            box_tf: Gf.Matrix4d = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            box_location = box_tf.ExtractTranslation()
            force_to_center = (pallet_center - box_location) * strength * strength_center_multiplier
            physx_api.apply_force_at_pos(stage_id, body_path, carb.Float3(*force_to_center), carb.Float3(*box_location))

        await common.wait_for(20)
        common.timeline.pause()

