"""
Physics-based Randomized Volume Filling
"""
import random, omni.timeline, omni.kit.app
from itertools import chain
from tools.common import set_local_trasform, get_dimensions, bbox_cache
import isaacsim.core.utils.semantics as semantics_utils
# import isaacsim.core.experimental.utils.semantics as semantics_utils
from semantics.schema.editor import remove_prim_semantics

import carb
import omni.kit.app, omni.physx
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysicsSchemaTools, PhysxSchema
from typing import Literal

timeline = omni.timeline.get_timeline_interface()
app_interface = omni.kit.app.get_app()

async def wait_for(frames: int): 
    for _ in range(frames):
        await app_interface.next_update_async()


# Enables collisions with the asset (without rigid body dynamics the asset will be static)
def add_colliders(prim, approx_type: Literal["convexHull", "convexDecomposition"]):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)  # type: ignore
            collision_api.CreateCollisionEnabledAttr(True)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set(approx_type)


# Enables rigid body dynamics (physics simulation) on the prim (having valid colliders is recommended)
def add_rigid_body_dynamics(prim: Usd.Prim, disable_gravity=False, angular_damping=None):
    # Physics
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim) 
    else:
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    # PhysX
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    else:
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
    physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
    if angular_damping is not None:
        physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)

def create_collision_walls(prim: Usd.Prim, height=4.6, thickness=0.1, material=None, visible=False) -> list[Usd.Prim]:
    dimensions_x, dimensions_y, _ = get_dimensions(prim)

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
        set_local_trasform(prim=cube_prim, translation=location, scale=scale)
        add_colliders(cube_prim, approx_type="convexHull")
        if not visible:
            UsdGeom.Imageable(cube_prim).MakeInvisible()
        if material is not None:
            mat_binding_api = UsdShade.MaterialBindingAPI.Apply(cube_prim)
            mat_binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        collision_walls.append(cube_prim)
    return collision_walls


# Slide the assets independently in perpendicular directions and then pull them all together towards the given center
async def apply_forces_async(boxes: list[Usd.Prim], pallet, strength=550, strength_center_multiplier=2):
    timeline.play()
    # Get the pallet center and forward vector to apply forces in the perpendicular directions and towards the center
    pallet_tf: Gf.Matrix4d = UsdGeom.Xformable(pallet).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pallet_center = pallet_tf.ExtractTranslation()
    pallet_rot: Gf.Rotation = pallet_tf.ExtractRotation()
    force_forward = Gf.Vec3d(pallet_rot.TransformDir(Gf.Vec3d(1, 0, 0))) * strength
    force_right = Gf.Vec3d(pallet_rot.TransformDir(Gf.Vec3d(0, 1, 0))) * strength

    physx_api = omni.physx.get_physx_simulation_interface()
    stage_id = stage_utils.get_current_stage_id()
    for box_prim in boxes:
        body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
        forces = [force_forward, force_right, -force_forward, -force_right]
        for force in chain(forces, forces):
            box_tf: Gf.Matrix4d = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            box_position = carb.Float3(*box_tf.ExtractTranslation())
            physx_api.apply_force_at_pos(stage_id, body_path, carb.Float3(force), box_position, "Force")
            
            # for _ in range(10):
            await wait_for(2)

    # Pull all box at once to the pallet center
    for box_prim in boxes:
        body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
        box_tf: Gf.Matrix4d = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        box_location = box_tf.ExtractTranslation()
        force_to_center = (pallet_center - box_location) * strength * strength_center_multiplier
        physx_api.apply_force_at_pos(stage_id, body_path, carb.Float3(*force_to_center), carb.Float3(*box_location))

    await wait_for(20)
    timeline.pause()

async def apply_multiforces_async(boxes_list: list[list[Usd.Prim]], pallets: list[Usd.Prim],
                             strength=550, strength_center_multiplier=2):
    timeline.play()
    # Get the pallet center and forward vector to apply forces in the perpendicular directions and towards the center
    contexts = []
    for boxes, pallet in zip(boxes_list, pallets):
        pallet_tf: Gf.Matrix4d = UsdGeom.Xformable(pallet).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pallet_center = pallet_tf.ExtractTranslation()
        pallet_rot: Gf.Rotation = pallet_tf.ExtractRotation()
        force_forward = Gf.Vec3d(pallet_rot.TransformDir(Gf.Vec3d(1, 0, 0))) * strength
        force_right = Gf.Vec3d(pallet_rot.TransformDir(Gf.Vec3d(0, 1, 0))) * strength

        contexts.append({"pallet": pallet, "boxes": boxes, "center": pallet_center,
                         "forces": [force_forward, force_right, -force_forward, -force_right]})

    physx_api = omni.physx.get_physx_simulation_interface()
    stage_id = stage_utils.get_current_stage_id()

    # Slide boxes on all pallets in shared simulation steps.
    for force_idx in range(4):
        for ctx in contexts:
            force = ctx["forces"][force_idx]
            for box_prim in ctx["boxes"]:
                body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
                box_tf: Gf.Matrix4d = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                box_position = carb.Float3(*box_tf.ExtractTranslation())
                physx_api.apply_force_at_pos(stage_id, body_path, carb.Float3(force), box_position, "Force")
        await wait_for(6)

    # Pull every box toward its own pallet center.
    for ctx in contexts:
        pallet_center = ctx["center"]

        for box_prim in ctx["boxes"]:
            body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
            box_tf: Gf.Matrix4d = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            box_location = box_tf.ExtractTranslation()
            force_to_center = (pallet_center - box_location) * strength * strength_center_multiplier

            physx_api.apply_force_at_pos(stage_id, body_path, carb.Float3(*force_to_center),
                                         carb.Float3(*box_location), "Force")

    await wait_for(20)
    timeline.pause()

async def volume_stack(pallet_prims: list[Usd.Prim], boxes_urls_and_weights: list[tuple[str, float]],
                       boxes_nums: list[int], overhang=0.0, drop_height=3.6, drop_margin=0.4) -> None:
    this_stage = stage_utils.get_current_stage()
    # Create a custom physics material to allow the boxes to easily slide into stacking positions
    physics_material_prim_path = "/VolumeStackLooks"
    material_path = "/VolumeStackLooks/PhysicsMaterial"
    if not prims_utils.is_prim_path_valid(physics_material_prim_path):
        prims_utils.create_prim("/VolumeStackLooks", prim_type="Scope")
        default_material = UsdShade.Material.Define(this_stage, material_path)
    else:
        # default_material = prims_utils.get_prim_at_path(material_path)
        material_prim = prims_utils.get_prim_at_path(material_path)
        default_material = UsdShade.Material(material_prim)
    physics_material = UsdPhysics.MaterialAPI.Apply(default_material.GetPrim())
    physics_material.CreateRestitutionAttr().Set(0.0)  # Inelastic collision (no bouncing)
    physics_material.CreateStaticFrictionAttr().Set(0.001)  # Small friction to allow sliding of stationary boxes
    physics_material.CreateDynamicFrictionAttr().Set(0.001)  # Small friction to allow sliding of moving boxes
    
    collision_walls_list: list[list[Usd.Prim]] = []
    boxes_prim_list = []
    box_prims_list = []
    box_urls, box_weights = zip(*boxes_urls_and_weights)
    for i, pallet_prim in enumerate(pallet_prims):
        # Apply the physics material to pallets
        add_colliders(pallet_prim, approx_type="convexDecomposition")
        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(pallet_prim)
        mat_binding_api.Bind(default_material, UsdShade.Tokens.weakerThanDescendants, "physics")
        # mat_binding_api.Bind(default_material, UsdShade.Tokens.weakerThanDescendants, UsdShade.Tokens.physics)

        # Create collision walls around the top of the pallet and apply the physics material to them
        collision_walls_list.append(create_collision_walls(pallet_prim, 
                                                           height=drop_height + drop_margin,
                                                           material=default_material))
        
        # Prepare boxes for the pallet
        pallet_prim_path = pallet_prim.GetPrimPath()
        boxes_prim_list.append(prims_utils.create_prim(f"{pallet_prim_path}/Boxes"))
        rand_boxes_urls = random.choices(box_urls, weights=box_weights, k=boxes_nums[i])
        box_prims = [stage_utils.add_reference_to_stage(usd_path=box_url, prim_path=f"{pallet_prim_path}/Boxes/Box_{i}")
             for i, box_url in enumerate(rand_boxes_urls)]
        box_prims.sort(key=lambda box: bbox_cache.ComputeLocalBound(box).GetVolume(), reverse=True)
        box_prims_list.append(box_prims)

    for idx in range(max(boxes_nums)):
        for box_prims, pallet_prim in zip(box_prims_list, pallet_prims):
            if idx >= len(box_prims):
                continue

            pallet_dimensions_x, pallet_dimensions_y, _ = get_dimensions(pallet_prim)
            # Simulate dropping the boxes from random poses on the pallet
            random_range_x = pallet_dimensions_x / 3.0
            random_range_y = pallet_dimensions_y / 3.0

            set_local_trasform(box_prims[idx], [random.uniform(-random_range_x, random_range_x), 
                                        random.uniform(-random_range_y, random_range_y), drop_height])
            add_colliders(box_prims[idx], approx_type="convexHull")
            add_rigid_body_dynamics(box_prims[idx], angular_damping=0.9)
            # Bind the physics material to the box (allow frictionless sliding)
            mat_binding_api = UsdShade.MaterialBindingAPI.Apply(box_prims[idx])
            mat_binding_api.Bind(default_material, UsdShade.Tokens.weakerThanDescendants, "physics")

        await wait_for(2)  # Wait for an app update to load the new attributes

        # Play simulation for a few frames for each box
        timeline.play()
        await wait_for(20)
        timeline.pause()

    # Iteratively apply forces to the boxes to move them around then pull them all together towards the pallet center
    await apply_multiforces_async(box_prims_list, pallet_prims, strength=1000)

    # Remove rigid body dynamics of the boxes until all other scenarios are completed
    for box_prims in box_prims_list:
        for box in box_prims:
            UsdPhysics.RigidBodyAPI(box).GetRigidBodyEnabledAttr().Set(False)

    # Increase the friction to prevent sliding of the boxes on the pallet before removing the collision walls
    physics_material.CreateStaticFrictionAttr().Set(0.999)
    physics_material.CreateDynamicFrictionAttr().Set(0.999)

    # Remove collision walls
    for collision_walls in collision_walls_list:
        for wall in collision_walls:
            this_stage.RemovePrim(wall.GetPath())
    overhang = abs(overhang)
    
    for boxes_prim in boxes_prim_list:
        semantics_utils.remove_all_labels(boxes_prim, include_descendants=True)
        semantics_utils.add_labels(boxes_prim, labels=["goods"])
        if overhang > 0.0:
            set_local_trasform(boxes_prim, translation=[random.uniform(-overhang, overhang),
                                                        random.uniform(-overhang, overhang), 0.0])

async def stack_boxes_on_pallet_async(pallet_prim: Usd.Prim, boxes_urls_and_weights: list[tuple[str, float]], 
                                      num_boxes: int, overhang=0.0, drop_height=3.6, drop_margin=0.4) -> None:
    pallet_path = pallet_prim.GetPrimPath()
    this_stage = stage_utils.get_current_stage()

    # Create a custom physics material to allow the boxes to easily slide into stacking positions
    physics_material_prim_path = "/VolumeStackLooks"
    material_path = "/VolumeStackLooks/PhysicsMaterial"
    if not prims_utils.is_prim_path_valid(physics_material_prim_path):
        prims_utils.create_prim("/VolumeStackLooks", prim_type="Scope")
        default_material = UsdShade.Material.Define(this_stage, material_path)
    else:
        material_prim = prims_utils.get_prim_at_path(material_path)
        default_material = UsdShade.Material(material_prim)
    physics_material = UsdPhysics.MaterialAPI.Apply(default_material.GetPrim())
    physics_material.CreateRestitutionAttr().Set(0.0)  # Inelastic collision (no bouncing)
    physics_material.CreateStaticFrictionAttr().Set(0.01)  # Small friction to allow sliding of stationary boxes
    physics_material.CreateDynamicFrictionAttr().Set(0.01)  # Small friction to allow sliding of moving boxes

    # Apply the physics material to the pallet
    add_colliders(pallet_prim, approx_type="convexDecomposition")
    mat_binding_api = UsdShade.MaterialBindingAPI.Apply(pallet_prim)
    mat_binding_api.Bind(default_material, UsdShade.Tokens.weakerThanDescendants, "physics")

    # Create collision walls around the top of the pallet and apply the physics material to them
    collision_walls = create_collision_walls(pallet_prim,
                                              height=drop_height + drop_margin,
                                              material=default_material)

    # Create the random boxes (without physics) with the specified weights and sort them by size (volume)
    box_urls, box_weights = zip(*boxes_urls_and_weights)
    rand_boxes_urls = random.choices(box_urls, weights=box_weights, k=num_boxes)
    box_prims = list()
    boxes_prim = prims_utils.create_prim(f"{pallet_path}/Boxes")

    box_prims = [stage_utils.add_reference_to_stage(usd_path=box_url, prim_path=f"{pallet_path}/Boxes/Box_{i}")
             for i, box_url in enumerate(rand_boxes_urls)]
    box_prims.sort(key=lambda box: bbox_cache.ComputeLocalBound(box).GetVolume(), reverse=True)

    pallet_dimensions_x, pallet_dimensions_y, _ = get_dimensions(pallet_prim)

    # Simulate dropping the boxes from random poses on the pallet
    random_range_x = pallet_dimensions_x / 3.0
    random_range_y = pallet_dimensions_y / 3.0
   
    for box_prim in box_prims:
        set_local_trasform(box_prim, [random.uniform(-random_range_x, random_range_x), 
                                      random.uniform(-random_range_y, random_range_y), drop_height])
        add_colliders(box_prim, approx_type="convexHull")
        add_rigid_body_dynamics(box_prim, angular_damping=0.9)
        
        # Bind the physics material to the box (allow frictionless sliding)
        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(box_prim)
        mat_binding_api.Bind(default_material, UsdShade.Tokens.weakerThanDescendants, "physics")
        # Wait for an app update to load the new attributes
        await app_interface.next_update_async()

        # Play simulation for a few frames for each box
        timeline.play()
        await wait_for(20)
        timeline.pause()

    # Iteratively apply forces to the boxes to move them around then pull them all together towards the pallet center
    await apply_forces_async(box_prims, pallet_prim, strength=1000)

    # Remove rigid body dynamics of the boxes until all other scenarios are completed
    for box in box_prims:
        UsdPhysics.RigidBodyAPI(box).GetRigidBodyEnabledAttr().Set(False)

    # Increase the friction to prevent sliding of the boxes on the pallet before removing the collision walls
    physics_material.CreateStaticFrictionAttr().Set(0.999)
    physics_material.CreateDynamicFrictionAttr().Set(0.999)

    # Remove collision walls
    for wall in collision_walls:
        this_stage.RemovePrim(wall.GetPath())

    semantics_utils.remove_all_labels(boxes_prim, include_descendants=True)
    semantics_utils.add_labels(boxes_prim, labels=["goods"])
    overhang = abs(overhang)
    if overhang > 0.0:
        set_local_trasform(boxes_prim, translation=[random.uniform(-overhang, overhang),
                                                    random.uniform(-overhang, overhang), 0.0])