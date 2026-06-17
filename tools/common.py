from typing import Sequence, Literal
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils import bounds as bounds_utils, prims as prims_utils, stage as stage_utils
# from pxr import Usd, Gf, UsdGeom, UsdPhysics
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysicsSchemaTools, PhysxSchema
import pathlib, glob, os, omni.timeline, omni.kit.app

timeline = omni.timeline.get_timeline_interface()
app_interface = omni.kit.app.get_app()

async def wait_for(frames: int):
    for _ in range(frames):
        await app_interface.next_update_async()

async def app_update_async():
    await app_interface.next_update_async()

def find_files(dir: str, extension: str, recursive=True):
    # 拼接匹配模式，** 表示递归匹配任意层级的子文件夹
    search_pattern = os.path.join(dir, "**", f"*{extension}")

    # recursive=True 激活多层子文件夹的查找
    return glob.glob(search_pattern, recursive=recursive)

def find_usds(dir: str) -> list[str]:
    folder = pathlib.Path(dir)
    usd_files = []
    for usd_file in folder.rglob("*.usd"):
        usd_files.append(str(usd_file))
    return usd_files

def load_usds(dir: str | list[str], objs_prim_path="/World/Objs") -> list[str]:
    """
    Load USDs from a folder into the stage
    """
    prims_utils.create_prim(objs_prim_path)
    usd_file_paths = find_usds(dir) if type(dir) is str else dir
    obj_prim_paths = []
    idx = 0
    for file_path in usd_file_paths:
        idx += 1
        obj_prim_path = f"{objs_prim_path}/obj{idx}"
        obj_prim_paths.append(obj_prim_path)
        stage_utils.add_reference_to_stage(usd_path=file_path, prim_path=obj_prim_path)
    return obj_prim_paths


def yaw2quat(yaw: float) -> Sequence[float]:
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()
    # Convert Gf.Quat to a format Isaac Sim understands (w, x, y, z)
    # GetReal() is 'w', GetImaginary() is (x, y, z)
    return rotation.GetReal(), *rotation.GetImaginary()

def make_visible(prim: str | Usd.Prim, visible: bool = True):
    prim: Usd.Prim = prim if isinstance(prim, Usd.Prim) else prims_utils.get_prim_at_path(prim)
    if visible:
        UsdGeom.Imageable(prim).MakeVisible()
    else:
        UsdGeom.Imageable(prim).MakeInvisible()

bbox_cache = bounds_utils.create_bbox_cache()

def get_dimensions(prim: str | Usd.Prim):
    """
    Calculate dimensions (x, y, z)
    """
    prim_path = str(prim.GetPrimPath()) if isinstance(prim, Usd.Prim) else prim
    aabb = bounds_utils.compute_aabb(bbox_cache, prim_path) # [min x, min y, min z, max x, max y, max z]
    return float(aabb[3] - aabb[0]), float(aabb[4] - aabb[1]), float(aabb[5] - aabb[2])

def set_local_trasform(prim: str | Usd.Prim, 
                       translation: Sequence[float], 
                       orientation: Sequence[float] = [1.0, 0.0, 0.0, 0.0],
                       scale: Sequence[float] = [1.0, 1.0, 1.0]) -> None:
    prim_path = str(prim.GetPrimPath()) if type(prim) is Usd.Prim else prim
    xform_prim = SingleXFormPrim(prim_path)
    xform_prim.initialize()
    xform_prim.set_local_pose(translation, orientation)
    xform_prim.set_local_scale(scale)

def set_world_trasform(prim: str | Usd.Prim, 
                       translation: Sequence[float], 
                       orientation: Sequence[float] = [1.0, 0.0, 0.0, 0.0],
                       scale: Sequence[float]=None) -> None:
    prim_path = str(prim.GetPrimPath()) if type(prim) is Usd.Prim else prim
    xform_prim = SingleXFormPrim(prim_path)
    xform_prim.initialize()
    xform_prim.set_world_pose(translation, orientation)
    if scale is not None:
        xform_prim.set_local_scale(scale)

# def add_colliders(prim):
#     prim = prim if type(prim) is Usd.Prim else prims_utils.get_prim_at_path(prim)
#     # Iterate descendant prims (including root) and add colliders to mesh or primitive types
#     for desc_prim in Usd.PrimRange(prim):
#         if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
#             # Physics
#             if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
#                 collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
#             else:
#                 collision_api = UsdPhysics.CollisionAPI(desc_prim)
#             collision_api.CreateCollisionEnabledAttr(True)

#         # Add mesh specific collision properties only to mesh types
#         if desc_prim.IsA(UsdGeom.Mesh):
#             if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
#                 mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
#             else:
#                 mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
#             # mesh_collision_api.CreateApproximationAttr().Set("triangleMesh")
#             mesh_collision_api.CreateApproximationAttr().Set("convexHull")

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
