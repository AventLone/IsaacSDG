"""
Permutations and Combinations
"""
import asyncio, random
from typing import Literal
from isaacsim.core.utils import stage, prims as prims_utils, xforms
from isaacsim.core.prims import SingleXFormPrim
from omni import usd
from tools.common import make_visiable, get_dimensions, set_local_trasform, yaw2quat
from pxr import Usd

class PermuAndCombi:
    # PRIM_PATH = "/World/CombiPrim"
    _instance_id = 0

    def __init__(self, prim_paths: list[str]) -> None:
        PermuAndCombi._instance_id += 1
        self.prim_path = f"/World/CombiPrim_{PermuAndCombi._instance_id}"
        prims_utils.create_prim(self.prim_path)

        self.prim = SingleXFormPrim(self.prim_path)
        self.prim.initialize()  # needed if operating on an existing prim in a scene

        self._component_prim_paths = prim_paths
        self._stage = stage.get_current_stage()

        self.column_prims: list[Usd.Prim] = []

        for prim_path in prim_paths:
            make_visiable(prim_path, False)

        self._trigger = asyncio.Event()
        self._finished = asyncio.Event()

    async def run(self):
        self._trigger.set()
        await self._finished.wait()
        self._finished.clear()

    def set_pose(self, translation: tuple[float, float, float], yaw: float):
        self.prim.set_world_pose(position=translation, orientation=yaw2quat(yaw)) # type: ignore

    def _pile_on(self, colomn_idx: int, row: int, xy_range=0.01, yaw_range=5.0):
        """
        Pile a component on top of the specified colomn
        """
        component_prim_path = f"{self.prim_path}/Col{colomn_idx}/component{row}"
        col_component_prim_path = f"{self.prim_path}/Col{colomn_idx}/component0"
        usd.duplicate_prim(stage=self._stage,
                           prim_path=col_component_prim_path,
                           path_to=component_prim_path)
        _, _, component_height = get_dimensions(col_component_prim_path)
        new_component_pos = (random.uniform(-xy_range, xy_range), 
                             random.uniform(-xy_range, xy_range), 
                             component_height * row)
        set_local_trasform(component_prim_path, new_component_pos, yaw2quat(random.uniform(-yaw_range, yaw_range)))
    
    def _add_colcomn(self, col_idx, direction: Literal['x', 'y'], gap=0.036):
        """
        Add an extra colomn
        """
        col_prim_path = f"{self.prim_path}/Col{col_idx}"
        self.column_prims.append(prims_utils.create_prim(col_prim_path))
        component_path = f"{col_prim_path}/component1"
        random_component = random.choice(self._component_prim_paths)
        usd.duplicate_prim(self._stage, prim_path=random_component, path_to=component_path)
        make_visiable(component_path)

        if col_idx > 1:
            pre_col_prim_path = f"{self.prim_path}/Col{col_idx - 1}"
            dimensions_x, dimensions_y, _ = get_dimensions(pre_col_prim_path)
            pre_col_position, _ = xforms.get_local_pose(pre_col_prim_path)
            new_col_position = (pre_col_position[0] + dimensions_x + gap, pre_col_position[1],
                                pre_col_position[2]) if direction == 'x' else (pre_col_position[0],
                                                                               pre_col_position[1] + dimensions_y + gap,
                                                                               pre_col_position[2])
            set_local_trasform(col_prim_path, new_col_position)

    def create_columns(self, columns: int, direction: Literal['x', 'y'], gap=0.036):
        """
        Create all colomns
        """
        dimensions_x_list, dimensions_y_list = [], []
        random.shuffle(self._component_prim_paths)
        component_count = len(self._component_prim_paths)
        for col_idx in range(columns):
            col_prim_path = f"{self.prim_path}/Col{col_idx}"
            self.column_prims.append(prims_utils.create_prim(col_prim_path))
            component_path = f"{col_prim_path}/component0"
            # random_component = random.choice(self._component_prim_paths)
            random_component = self._component_prim_paths[col_idx] if col_idx +1 <= component_count else random.choice(self._component_prim_paths)
            usd.duplicate_prim(self._stage, prim_path=random_component, path_to=component_path)
            make_visiable(component_path)
            dimensions_x, dimensions_y, _ = get_dimensions(component_path)
            dimensions_x_list.append(dimensions_x)
            dimensions_y_list.append(dimensions_y)

        whole_dimensions_x_half = (sum(dimensions_x_list) + gap * (columns - 1)) / 2.0
        whole_dimensions_y_half = (sum(dimensions_y_list) + gap * (columns - 1)) / 2.0

        col_0_position = (-whole_dimensions_x_half + dimensions_x_list[0] / 2.0, 
                            0.0, 0.0) if  direction == 'x' else (0.0, 
                                                                 -whole_dimensions_y_half + dimensions_y_list[0] / 2.0,
                                                                 0.0)
        set_local_trasform(f"{self.prim_path}/Col0", col_0_position)
        
        for col_idx in range(1, columns):
            pre_col_prim_path = f"{self.prim_path}/Col{col_idx - 1}"
            col_prim_path = f"{self.prim_path}/Col{col_idx}"
            pre_dimensions_x, pre_dimensions_y = dimensions_x_list[col_idx - 1], dimensions_y_list[col_idx - 1]
            dimensions_x, dimensions_y = dimensions_x_list[col_idx], dimensions_y_list[col_idx]
            pre_col_position, _ = xforms.get_local_pose(pre_col_prim_path)
            gap_x = (pre_dimensions_x + dimensions_x) / 2.0 + gap
            gap_y = (pre_dimensions_y + dimensions_y) / 2.0 + gap
            new_col_position = (pre_col_position[0] + gap_x, pre_col_position[1],
                                pre_col_position[2]) if direction == 'x' else (pre_col_position[0],
                                                                               pre_col_position[1] + gap_y,
                                                                               pre_col_position[2])
            set_local_trasform(col_prim_path, new_col_position)
        

    async def stack(self, columns: int, rows=1):
        col_list = list(range(1, columns + 1))
        random.shuffle(col_list)

        # self.create_columns(columns=columns, direction=direction, gap=gap)

        if rows <= 1:
            return
        
        for col in range(columns):              
                for row in range(1, rows + 1):
                    await self._trigger.wait()
                    self._trigger.clear()
                    self._pile_on(col, row)
                    self._finished.set()