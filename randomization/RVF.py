"""
Physics-based Randomized Volume Filling
"""
import random, omni.timeline, omni.kit.app
from itertools import chain
from tools.common import set_local_trasform, get_dimensions, bbox_cache
# from omni.physx.scripts.utils import setCollider
from isaacsim.core.utils import semantics as semantics_utils

import carb
import omni.kit.app, omni.physx
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysicsSchemaTools, PhysxSchema # type: ignore

timeline = omni.timeline.get_timeline_interface()
app_interface = omni.kit.app.get_app()


async def wait_for(frames: int):
    for _ in range(frames):
        await app_interface.next_update_async()  # type: ignore


# Enables collisions with the asset (without rigid body dynamics the asset will be static)
def add_colliders(prim):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):  # type: ignore
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)  # type: ignore
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)  # type: ignore
            collision_api.CreateCollisionEnabledAttr(True)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):  # type: ignore
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)  # type: ignore
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)  # type: ignore
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")


# Enables rigid body dynamics (physics simulation) on the prim (having valid colliders is recommended)
def add_rigid_body_dynamics(prim: Usd.Prim, disable_gravity=False, angular_damping=None):
    # Physics
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):   # type: ignore
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)   # type: ignore   
    else:
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)   # type: ignore
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    # PhysX
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    else:
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
    physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
    if angular_damping is not None:
        physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)

def create_collision_walls(prim: Usd.Prim, height=4.6, thickness=0.1, material=None, visible=False):
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
        prim = prims_utils.create_prim(f"{prim_path}/{name}", prim_type="Cube")
        set_local_trasform(prim=prim, translation=location, scale=scale)
        add_colliders(prim)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()
        if material is not None:
            mat_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            mat_binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        collision_walls.append(prim)
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


async def stack_boxes_on_pallet_async(pallet_prim: Usd.Prim, boxes_urls_and_weights: list[tuple[str, float]], 
                                      num_boxes: int, overhang=0.0, drop_height=3.6, drop_margin=0.4) -> None:
    pallet_path = pallet_prim.GetPrimPath()
    this_stage = stage_utils.get_current_stage()

    # Create a custom physics material to allow the boxes to easily slide into stacking positions
    prims_utils.create_prim(f"{pallet_path}/Looks", prim_type="Scope")
    material_path = f"{pallet_path}/Looks/PhysicsMaterial"
    default_material = UsdShade.Material.Define(this_stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(default_material.GetPrim()) # type: ignore
    physics_material.CreateRestitutionAttr().Set(0.0)  # Inelastic collision (no bouncing)
    physics_material.CreateStaticFrictionAttr().Set(0.01)  # Small friction to allow sliding of stationary boxes
    physics_material.CreateDynamicFrictionAttr().Set(0.01)  # Small friction to allow sliding of moving boxes

    # Apply the physics material to the pallet
    # add_colliders(pallet_prim)
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
        add_colliders(box_prim)
        add_rigid_body_dynamics(box_prim, angular_damping=0.9)
        
        # Bind the physics material to the box (allow frictionless sliding)
        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(box_prim)
        mat_binding_api.Bind(default_material, UsdShade.Tokens.weakerThanDescendants, "physics")
        # Wait for an app update to load the new attributes
        await app_interface.next_update_async()   # type: ignore

        # Play simulation for a few frames for each box
        timeline.play()
        await wait_for(20)
        timeline.pause()

    # Iteratively apply forces to the boxes to move them around then pull them all together towards the pallet center
    await apply_forces_async(box_prims, pallet_prim, strength=1000)

    # Remove rigid body dynamics of the boxes until all other scenarios are completed
    for box in box_prims:
        UsdPhysics.RigidBodyAPI(box).GetRigidBodyEnabledAttr().Set(False)   # type: ignore

    # Increase the friction to prevent sliding of the boxes on the pallet before removing the collision walls
    physics_material.CreateStaticFrictionAttr().Set(0.99)
    physics_material.CreateDynamicFrictionAttr().Set(0.99)

    # Remove collision walls
    for wall in collision_walls:
        this_stage.RemovePrim(wall.GetPath())

    semantics_utils.remove_all_semantics(boxes_prim, recursive=True)
    semantics_utils.add_labels(boxes_prim, labels=["goods"])
    overhang = abs(overhang)
    if overhang > 0.0:
        set_local_trasform(boxes_prim, translation=[random.uniform(-overhang, overhang),
                                                    random.uniform(-overhang, overhang), 0.0])