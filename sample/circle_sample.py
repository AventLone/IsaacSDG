import omni.replicator.core as rep
from isaacsim.core.utils import xforms, bounds
import numpy as np

def generate_orbit_positions(origin: np.ndarray, radius: float, count: int):
    # 在 0 到 2pi 之间均匀生成角度
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    # 计算对应的 X, Y 坐标
    return [(float(radius * np.cos(angle) + origin[0]), 
             float(radius * np.sin(angle) + origin[1]), 
             float(origin[2])) for angle in angles]
    
class CircleSampler:
    def __init__(self, prim_path: str) -> None:
        self.obj_prim_path = prim_path
        self.obj_prim = rep.get.prim_at_path(prim_path)
        self.camera = rep.create.camera(focus_distance=400.0, focal_length=15.0,
                                        clipping_range=(0.1, 1000000.0), name="PickupCam")
        
        self.materials = rep.create.material_omnipbr(
            metallic=rep.distribution.uniform(0.0, 1.0),
            roughness=rep.distribution.uniform(0.0, 1.0),
            diffuse=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            count=300
        )

        rep.randomizer.register(self._randomize_obj_pose)
        rep.randomizer.register(self._randomize_obj_apperance)
        rep.randomizer.register(self._randomize_camera_pose)
        rep.randomizer.register(self._randomize_light)

        self._obj_poses = [(-6.0, -14.38727644649423, 0.0),
                           (-6.0, -5.485066445180405, 0.0),
                           (-13.4, -2.3, 0.0)]
        self._camera_poses = self._get_orbit_points(origins=self._obj_poses,
                                                    heights=[0.5, 0.8, 1.2], 
                                                    radiuses=[1.7, 2.2, 2.5, 3.0])
        frames_required = len(self._camera_poses)
        print(f"{frames_required} images will be generated.")

        self.camera_trigger = rep.trigger.on_frame(max_execs=frames_required, interval=1, rt_subframes=8)

        cam_change_count = int(len(self._camera_poses) / len(self._obj_poses))
        self.obj_pose_trigger = rep.trigger.on_frame(max_execs=frames_required // cam_change_count, interval=cam_change_count, rt_subframes=8)
        self.obj_apperance_trigger = rep.trigger.on_frame(max_execs=frames_required // 2, interval=2, rt_subframes=8)
        self.light_trigger = rep.trigger.on_frame(max_execs=frames_required // 15, interval=15, rt_subframes=8)
       
    @property
    def obj_position(self):
        position, _ = xforms.get_world_pose(self.obj_prim_path)
        return position

    def trigger_camera(self):
        with self.camera_trigger:
            rep.randomizer._randomize_camera_pose()   # type: ignore

    def trigger_obj_pose(self):
        with self.obj_pose_trigger:
            rep.randomizer._randomize_obj_pose()      # type: ignore

    def trigger_obj_apperance(self):
        with self.obj_apperance_trigger:
            rep.randomizer._randomize_obj_apperance() # type: ignore

    def trigger_light(self):
        with self.light_trigger:
            rep.randomizer._randomize_light()   # type: ignore
    
    def _randomize_obj_pose(self) -> rep.scripts.utils.ReplicatorItem:
        with self.obj_prim:
            rep.modify.pose(position=rep.distribution.sequence(self._obj_poses))
        return self.obj_prim.node # type: ignore
    
    def _randomize_obj_apperance(self) -> rep.scripts.utils.ReplicatorItem:
        meshes = rep.get.prims(path_pattern=f"{self.obj_prim_path}/*", prim_types=["Mesh", "GeomSubset"])
        with meshes:
            rep.randomizer.materials(self.materials)
        return meshes.node   # type: ignore

    def _randomize_camera_pose(self) -> rep.scripts.utils.ReplicatorItem:
        with self.camera:
            rep.modify.pose(
                position=rep.distribution.sequence(self._camera_poses),
                look_at=self.obj_prim
            )
        return self.camera.node # type: ignore

    def _randomize_light(self) -> rep.scripts.utils.ReplicatorItem:
        lights = rep.get.prims(prim_types=["RectLight", "SphereLight", "DomeLight"])
        with lights:
            rep.modify.attribute("intensity", rep.distribution.uniform(1000, 80000))
            rep.modify.attribute("color", rep.distribution.uniform((0.3, 0.3, 0.3), (1.0, 1.0, 1.0)))
        return lights.node # type: ignore
        
    def _get_orbit_points(self, origins: list,  heights: list, radiuses: list) -> list:
        positions = list()
        # position = self.obj_position
        count = 8
        for origin in origins:
            origin = np.array(origin, dtype=np.float32)
            for height in heights:
                origin[2] = height
                for radius in radiuses:
                    positions.extend(generate_orbit_positions(origin, radius, count))

        return positions
        