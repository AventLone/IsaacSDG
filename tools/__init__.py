VERSION = "0.1"

#----------------------------------- Core ------------------------------------------#
from isaacsim.simulation_app import SimulationApp
HEADLESS = True
PHYSICS_STEP_SIZE = 1.0 / 50.0 # Physics step size (seconds). Example: 1/60 = 0.016666...
SIMU_APP = SimulationApp({"renderer": "RayTracedLighting",
                          "headless": HEADLESS, "disable_viewport_updates": HEADLESS,
                          "physics_dt": PHYSICS_STEP_SIZE})

import warp.config
warp.config.quiet = True

import carb.settings
settings = carb.settings.get_settings()
settings.set("/log/level", "error")
settings.set("/log/channels/omni.replicator.core", "error")
settings.set("/log/channels/omni.replicator.core.*", "error")

# Enable PhysX GPU dynamics
settings.set("/physics/useGpuDynamics", True)
settings.set("/physics/cudaDevice", 0)
#-----------------------------------------------------------------------------------#

def app_update(frames: int):
    for _ in range(frames):
        SIMU_APP.update()

def app_loop():
    while SIMU_APP.is_running():
        SIMU_APP.update()
    SIMU_APP.close()

from .path_generation import generate_lawnmower_path
from .scatter_objects import scatter, Pile, LoadedPallet
from .evaluate_coco_dataset import audit_coco
from .logger import LOGGER