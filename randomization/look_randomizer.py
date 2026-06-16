from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
from isaacsim.core.utils import prims as prims_utils
import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
from pathlib import Path
import sys
from tqdm import trange

def random_color_texture(path: str, size: int = 512, blur_radius: float = 8.0):
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)   # Start with random RGB noise
    img = Image.fromarray(arr, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))   # Blur it to make smooth mixed-color regions instead of pixel noise
    img.save(path)
    return path

def random_mixed_color_texture(path: str, size: int = 512, grid_size: int = 16, 
                               saturation: float = 1.8, contrast: float = 1.4, blur_radius: float = 2.0):
    # Generate low-res random color map
    arr = np.random.randint(30, 256, (grid_size, grid_size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # Upscale to texture size, creating large smooth color regions
    img = img.resize((size, size), Image.Resampling.BICUBIC)

    # Optional blur
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Increase saturation and contrast
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    img.save(path)
    return path


class LooksRandomizer:
    def __init__(self, random_looks_count=100):
        temp_texture_img_dir = "images/temp"
        Path(temp_texture_img_dir).mkdir(parents=True, exist_ok=True)
        this_stage = prims_utils.get_current_stage()

        material_prim_path = "/World/Materials"
        if not prims_utils.get_prim_at_path(material_prim_path).IsValid():
            prims_utils.create_prim(material_prim_path)

        self._materials = []
        # for i in range(random_looks_count):
        for i in trange(random_looks_count, desc="Creating random materials", unit="mat", file=sys.stdout):
            material_path = f"{material_prim_path}/material_{i}"
            material = UsdShade.Material.Define(this_stage, material_path)

            shader = UsdShade.Shader.Define(this_stage, f"{material_path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            # if random.random() <= 0.7:
            #     tex = UsdShade.Shader.Define(this_stage, material_path + "/DiffuseTexture")
            #     tex.CreateIdAttr("UsdUVTexture")
            #     tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(random_mixed_color_texture(path=f"{temp_texture_img_dir}/texture{i}.png"))
            #     tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            #     shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            # else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)))
            
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(random.uniform(0.01, 1.0))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(random.uniform(0.01, 1.0))
            # shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(random.uniform(0.9, 1.0))      # 完全透明
            # shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(random.uniform(0.1, 1.0))          # 折射率设为1.0（与空气一致，消除折射扭曲）

            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self._materials.append(material)

    def set_prim(self, prim_path: str):
        prim = prims_utils.get_prim_at_path(prim_path)

        if not prim or not prim.IsValid():
            raise RuntimeError(f"Invalid prim path: {prim_path}")
        
        self._sub_prims = []

        for sub_prim in Usd.PrimRange(prim):
            if sub_prim.IsA(UsdGeom.Mesh) or sub_prim.IsA(UsdGeom.Gprim):
                self._sub_prims.append(sub_prim)

    def randomize(self):
        for subprim in self._sub_prims:
            UsdShade.MaterialBindingAPI.Apply(subprim).Bind(random.choice(self._materials))