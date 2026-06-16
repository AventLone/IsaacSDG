import os, carb.settings
import omni.replicator.core as rep
from writers import CocoInstanceSegWriter
from datetime import datetime
from isaacsim.core.utils import stage as stage_utils, prims as prims_utils
from pxr import UsdGeom, Gf

from tools import audit_coco, LOGGER
import glob




class BaseSDG:
    # Disable capture on play and async rendering
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
    carb.settings.get_settings().set("/app/asyncRendering", False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 1) # (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)

    def __init__(self, writer_type=CocoInstanceSegWriter, save_path=None) -> None:
        # Set up writer
        timestamp = datetime.now().strftime("%Y.%m.%d-%H:%M")
        self._save_at = f"generated_data/{timestamp}" if save_path is None else f"{save_path}/{timestamp}"
        data_save_dir = os.path.join(os.getcwd(), self._save_at)
        self._writer = writer_type(output_dir=data_save_dir)

    def create_camera(self, resolution=(504, 504), focus_distance=400.0, 
                       focal_length=15.0, horizontal_aperture=36.0, clipping_range=(0.1, 10000.0)):
        stage = stage_utils.get_current_stage()
        camera_path = "/World/Camera"

        if not prims_utils.get_prim_at_path(camera_path).IsValid():
            camera = UsdGeom.Camera.Define(stage, camera_path)
        else:
            camera = UsdGeom.Camera(prims_utils.get_prim_at_path(camera_path))

        camera.GetFocusDistanceAttr().Set(focus_distance)
        camera.GetFocalLengthAttr().Set(focal_length)
        camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(*clipping_range))
        self._render_product = rep.create.render_product(camera_path,resolution=resolution)
        self._writer.attach(self._render_product)

        self._camera_xformable = UsdGeom.Xformable(camera)
        # self._camera_xform_api = UsdGeom.XformCommonAPI(camera)

    def detach_renderproduct(self):
        self._writer.detach()
        if self._render_product is not None:
            self._render_product.destroy()


    def set_camera_pose_lootat(self, position, lookat_target=(0.0, 0.0, 0.0)):
        """
        Move camera to `position` and make it look at `lookat_target`.

        position: tuple/list, (x, y, z)
        lookat_target: tuple/list, (x, y, z)
        """
        # eye: Gf.Vec3d = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        # target: Gf.Vec3d = Gf.Vec3d(float(lookat_target[0]), float(lookat_target[1]), float(lookat_target[2]))
        eye: Gf.Vec3d = Gf.Vec3d(*position)
        target: Gf.Vec3d = Gf.Vec3d(*lookat_target)

        if (eye - target).GetLength() < 1e-6:
            raise ValueError("Camera position and look-at target cannot be the same.")

        up = Gf.Vec3d(0.0, 0.0, 1.0)

        # If camera direction is almost parallel to Z-up, use Y-up instead
        direction: Gf.Vec3d = (target - eye).GetNormalized()
        if abs(Gf.Dot(direction, up)) > 0.99:
            up = Gf.Vec3d(0.0, 1.0, 0.0)

        # SetLookAt gives a view matrix, so invert it to get camera world transform
        view_mat = Gf.Matrix4d().SetLookAt(eye, target, up)
        camera_world_mat = view_mat.GetInverse()

        # Apply full transform directly, no Euler decomposition needed
        self._camera_xformable.ClearXformOpOrder()
        self._camera_xformable.AddTransformOp().Set(camera_world_mat)


    def evaluate_datset(self):
        # 1. Search only the immediate folder
        json_files = glob.glob(f"{self._save_at}/*.json")
        if len(json_files) != 1:
            LOGGER.error("JSON files in dataset are more than one!")
            return
        audit_coco(json_files[0])


    # def set_camera_pose_rpy(self, position, rpy_deg):
    #     """
    #     position: (x, y, z)
    #     rpy_deg:  (roll, pitch, yaw) in degrees
    #     """
    #     x, y, z = position
    #     roll, pitch, yaw = rpy_deg

    #     self._camera_xform_api.SetTranslate((float(x), float(y), float(z)))
    #     # RotationOrderXYZ means: X = roll, Y = pitch, Z = yaw
    #     self._camera_xform_api.SetRotate((float(roll), float(pitch), float(yaw)), UsdGeom.XformCommonAPI.RotationOrderXYZ)