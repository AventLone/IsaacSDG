from tools.common import SIMU_APP

from tools.orthographic_views import OrthographicProject
import cv2
import numpy as np
from isaacsim.core.utils import stage as stage_utils

pallet_prim_path = "/Pallet"

stage_utils.create_new_stage()
stage_utils.add_reference_to_stage(
            usd_path="/home/avent/Desktop/IsaacAssets/SimReadyExplorer/Warehouse/02/common_assets/props/heavydutynestablepallet_a01/heavydutynestablepallet_a01_inst_base.usd",
            prim_path="/Pallet"
        )

projector = OrthographicProject(pallet_prim_path, cell_size=0.002, padding=0.0)
face_on_x, _ = projector.projection_on_yz

cv2.imwrite("face_on_x.png", face_on_x)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img = cv2.morphologyEx(face_on_x, cv2.MORPH_OPEN, kernel)
# cv2.imwrite("face_on_x_open.png", face_on_x_open)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("noiseless_img.png", img)

inverted_img = cv2.bitwise_not(img)
cv2.imwrite("inverted_img.png", inverted_img)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_img)
print(f"num_labels: {num_labels}")

debug_img = cv2.cvtColor(face_on_x, cv2.COLOR_GRAY2BGR)



# 1. Isolate the area column, skipping index 0 (which is the background label)
# stats[1:, 4] gives you a 1D array of all cluster areas
cluster_areas = stats[1:, cv2.CC_STAT_AREA]
max_area = np.max(cluster_areas)
# if len(cluster_areas) > 0:
#     # 2. Find the maximum area value among all clusters
#     max_area = np.max(cluster_areas)

#     # 3. Find all indices where the area equals the maximum area
#     # We add 1 to the indices because we skipped the background index 0 earlier
#     max_indices = np.where(cluster_areas == max_area)[0] + 1

#     # 4. Extract the standard cv.Rect (x, y, w, h) for each matching index
#     biggest_rectangles = []
#     for idx in max_indices:
#         x = stats[idx, cv2.CC_STAT_LEFT]
#         y = stats[idx, cv2.CC_STAT_TOP]
#         w = stats[idx, cv2.CC_STAT_WIDTH]
#         h = stats[idx, cv2.CC_STAT_HEIGHT]
        
#         biggest_rectangles.append((x, y, w, h))

#     print(f"Maximum Pixel Area: {max_area}")
#     print(f"Biggest Rectangle(s) (x, y, w, h): {biggest_rectangles}")
# else:
#     biggest_rectangles = []
#     print("No foreground clusters found.")

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

            opening_rects.append()
        else:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imwrite('images/fitted_rectangles.png', debug_img)

SIMU_APP.close()

