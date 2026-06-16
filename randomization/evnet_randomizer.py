import omni.replicator.core as rep

class CameraAndLightRandomizer:
    def __init__(self, camera_path: list[tuple[float, float, float]]) -> None:
        self._camera_poses = camera_path
        self.frames_generated = len(camera_path)
        
        self.camera = rep.create.camera(focus_distance=400.0, focal_length=15.0,
                                        horizontal_aperture=36.0,  # Increase this value to widen the FOV
                                        clipping_range=(0.1, 1000000.0), name="PickupCam")
        
        self._trigger_camera_event = "randomize_camera_pose"
        self._trigger_light_event = "randomize_light"

        rep.randomizer.register(self._randomize_camera_pose)
        rep.randomizer.register(self._randomize_light)

        # self.camera_trigger = rep.trigger.on_custom_event(self._trigger_camera_event)
    
        with rep.trigger.on_custom_event(self._trigger_camera_event):
            rep.randomizer._randomize_camera_pose()
        with rep.trigger.on_custom_event(self._trigger_light_event):
            rep.randomizer._randomize_light()

    def randomize_camera(self):
        rep.utils.send_og_event(self._trigger_camera_event)

    def randomize_light(self):
        rep.utils.send_og_event(self._trigger_light_event)

    def _randomize_camera_pose(self) -> rep.scripts.utils.ReplicatorItem:
        with self.camera:
            rep.modify.pose(
                position=rep.distribution.sequence(self._camera_poses),
                look_at=rep.distribution.uniform((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2))
            )
        return self.camera.node

    def _randomize_light(self) -> rep.scripts.utils.ReplicatorItem:
        lights = rep.get.prims(prim_types=["RectLight", "SphereLight", "DomeLight"])
        with lights:
            rep.modify.attribute("intensity", rep.distribution.choice([1000, 5000, 10000, 20000, 40000, 50000]))
            rep.modify.attribute("color", rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        return lights.node
    

class LightRandomizer:
    def __init__(self):
        self._trigger_light_event = "randomize_light"
        rep.randomizer.register(self._randomize_light)
        with rep.trigger.on_custom_event(self._trigger_light_event):
            rep.randomizer._randomize_light()

    def randomize(self):
        rep.utils.send_og_event(self._trigger_light_event)

    def _randomize_light(self) -> rep.scripts.utils.ReplicatorItem:
        lights = rep.get.prims(prim_types=["RectLight", "SphereLight", "DomeLight"])
        with lights:
            # rep.modify.attribute("intensity", rep.distribution.choice([1000, 5000, 10000, 20000, 40000, 50000]))
            rep.modify.attribute("intensity", rep.distribution.uniform(2000.0, 50000.0))
            rep.modify.attribute("color", rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        return lights.node
    

class MaterialRandomizer:
    def __init__(self, prim_path: str, material_count=120) -> None:
        self.obj_prim_path = prim_path     
        self._materials = rep.create.material_omnipbr(
            metallic=rep.distribution.uniform(0.0, 1.0),
            roughness=rep.distribution.uniform(0.0, 1.0),
            diffuse=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            count=material_count
        )

        # 3. 定义事件名称
        self._trigger_material_event = "randomize_material"

        rep.randomizer.register(self._randomize_material)

        with rep.trigger.on_custom_event(self._trigger_material_event):
            rep.randomizer._randomize_material()   # type: ignore

    def randomize_material(self) -> None:
        """外部调用：触发材质随机化"""
        rep.utils.send_og_event(self._trigger_material_event)

    def _randomize_material(self) -> rep.scripts.utils.ReplicatorItem:
        meshes = rep.get.prims(path_pattern=f"{self.obj_prim_path}/*", prim_types=["Mesh", "GeomSubset"])
        with meshes:
            rep.randomizer.materials(self._materials)
        return meshes.node   # type: ignore
