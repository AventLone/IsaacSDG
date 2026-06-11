from tools import SIMU_APP

from tools.orthographic_views import OrthographicProject
import cv2
import numpy as np
from isaacsim.core.utils import stage as stage_utils

pallet_prim_path = "/Pallet"

# stage_utils.create_new_stage()
# stage_utils.add_reference_to_stage(
#             usd_path="/home/avent/Desktop/IsaacAssets/Props/KKP.usd",
#             prim_path="/Pallet"
#         )

if not stage_utils.open_stage("/home/avent/Desktop/IsaacAssets/SDG-Only/warehouse_stage.usd"):
    raise RuntimeError("Failed to ope usd.")

def get_prim_opening_img(prim_path: str, cell_size: float):
    projector = OrthographicProject(prim_path, cell_size, padding=0.0)
    face_on_x, _ = projector.projection_on_yz
    face_on_y, _ = projector.projection_on_xz

    return find_opening(face_on_x), find_opening(face_on_y)

def find_opening(img: np.ndarray):
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)
    inverted_img = cv2.bitwise_not(img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_img)

    # 1. Isolate the area column, skipping index 0 (which is the background label)
    # stats[1:, 4] gives you a 1D array of all cluster areas
    cluster_areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = np.max(cluster_areas)

    # Loop starts at 1 to skip the background (label 0)
    opening_rects = []
    

    for i in range(1, num_labels):
        # 1. This slice contains exactly [x, y, width, height] -> Your cv.Rect equivalent
        rect = stats[i, 0:4]
        x, y, w, h = rect
        
        # 2. Filter out tiny noise if needed
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 2:

            if float(area - max_area) / max_area < 0.02:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # opening_rects.append()
            else:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return debug_img


projector = OrthographicProject(prim_path="/World/Objects", cell_size=0.02, padding=0.1)
occ_grid, _ = projector.projection_on_xy
# face_x_opening, face_y_opening = get_prim_opening_img("/Pallet", cell_size=0.005)


cv2.imwrite('images/occ_grid.png', occ_grid)
# cv2.imwrite('images/y_opeing.png', face_y_opening)

SIMU_APP.close()

