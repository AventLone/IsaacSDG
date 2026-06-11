VERSION="0.1"

#----------------------------------- Core ------------------------------------------#
from isaacsim.simulation_app import SimulationApp
SIMU_APP = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

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

from .path_generation import generate_lawnmower_path
from .scatter_objects import scatter