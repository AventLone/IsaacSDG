VERSION="0.1"

#----------------------------------- Core ------------------------------------------#
from isaacsim.simulation_app import SimulationApp
SIMU_APP = SimulationApp({"renderer": "RayTracedLighting", "headless": True})

import warp.config
warp.config.quiet = True

import carb.settings
settings = carb.settings.get_settings()
settings.set("/log/level", "error")
settings.set("/log/channels/omni.replicator.core", "error")
settings.set("/log/channels/omni.replicator.core.*", "error")
#-----------------------------------------------------------------------------------#

def app_update(frames: int):
    for _ in range(frames):
        SIMU_APP.update()

def app_loop():
    while SIMU_APP.is_running():
        SIMU_APP.update()
    SIMU_APP.close()

from .path_generation import generate_lawnmower_path
from .scatter_objects import scatter, Pile

from .evaluate_coco_dataset import audit_coco

from .logger import LOGGER