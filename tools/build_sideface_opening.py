from tools.common import SIMU_APP

from pxr import Usd, UsdGeom, UsdShade, Sdf
from tools import common
from isaacsim.core.utils import prims as prims_utils, stage as stage_utils
from isaacsim.core.utils import semantics as semantics_utils
import omni.kit.commands
# from 


def create_invisible_material(material_path="/World/Materials/side_face_invisible_mat"):
    """
    创建一个完全透明、无折射、无反射的完全隐形材质。
    """
    current_stage = stage_utils.get_current_stage()
    material = UsdShade.Material.Define(current_stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(current_stage, Sdf.Path(f"{material_path}/PreviewSurface"))

    shader.CreateIdAttr("UsdPreviewSurface")
    
    # 核心修改项
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.0)      # 完全透明
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)          # 折射率设为1.0（与空气一致，消除折射扭曲）
    
    # 辅助修改项：消除任何可能残留的微弱反光
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    return material

def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material)

def create_sidefaces(prim_path: str, thickness=0.002):
    transparent_material = create_invisible_material()
    prim_label = semantics_utils.get_labels(prims_utils.get_prim_at_path(prim_path))
    print(prim_label)
    dimensions_x, dimensions_y, height = common.get_dimensions(prim_path)

    # Define the walls (name, location, size) with the specified thickness added externally to the surface and height
    faces = [("left", (0.0, -(dimensions_y + thickness) / 2.0, height / 2.0), (dimensions_x, thickness, height)),
             ("right", (0.0, (dimensions_y + thickness) / 2.0, height / 2.0), (dimensions_x, thickness, height)),
             ("front", ((dimensions_x + thickness) / 2.0, 0.0, height / 2.0), (thickness, dimensions_y, height)),
             ("back", (-(dimensions_x + thickness) / 2.0, 0.0, height / 2.0), (thickness, dimensions_y, height))]
    
    sides_faces_prim_path = f"{prim_path}/side_faces"
    prims_utils.create_prim(sides_faces_prim_path)
    for name, location, size in faces:
        scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        prim = prims_utils.create_prim(f"{sides_faces_prim_path}/{name}", prim_type="Cube")
        common.set_local_trasform(prim=prim, translation=location, scale=scale)
        bind_material(prim, transparent_material)
        semantics_utils.add_labels(prim, labels=[f"side_face"])


stage_utils.create_new_stage()

stage_utils.add_reference_to_stage(
            usd_path="/home/avent/Desktop/IsaacAssets/Isaac/5.1/Isaac/Props/Pallet/pallet.usd",
            prim_path="/Pallet"
        )

create_sidefaces("/Pallet")

while SIMU_APP.is_running():
    SIMU_APP.update()

SIMU_APP.close()