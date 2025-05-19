import cv2
import numpy as np

def preprocess_ir_data(ir_data, max_data):
    # Convert IR data to float for processing
    ir_data = ir_data.astype(np.float32)

    # Apply logarithmic compression to reduce brightness extremes
    ir_data = np.log1p(ir_data) / np.log1p(max_data) * 255
    ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)

    # Apply CLAHE to enhance dim regions and suppress bright ones
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
    ir_data = clahe.apply(ir_data)

    # Apply sharpening (Unsharp Masking)
    blurred = cv2.GaussianBlur(ir_data, (0, 0), 2)  # Gaussian blur
    ir_data = cv2.addWeighted(ir_data, 1.5, blurred, -0.5, 0)  # Sharpening formula

    return ir_data

def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    return undistorted

def detect_aruco(image,type=cv2.aruco.DICT_6X6_250):
    aruco_dict = cv2.aruco.getPredefinedDictionary(type)
    arucoParams = cv2.aruco.DetectorParameters()
    (CORNERS, ids,_) = cv2.aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)
    marker_positions = {}

    if len(CORNERS) > 0:
        ids = ids.flatten()
        for marker_corner, marker_id in zip(CORNERS, ids):
            corners = marker_corner.reshape((4, 2))
            cX, cY = np.mean(corners, axis=0).astype(int)
            marker_positions[marker_id] = (cX, cY)
    return marker_positions, CORNERS, ids

def draw_aruco(image, corners, ids):
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

        cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(markerID))
    return image

def detect_blob(image, min_thresh=200, max_thresh=255,min_area=20,min_circularity=0.1,min_convexity=0.1,min_inertia=0.01):
    params = cv2.SimpleBlobDetector_Params()
    # Change Color
    params.filterByColor, params.blobColor = True, 255
    # Change thresholds
    params.minThreshold, params.maxThreshold = min_thresh, max_thresh
    # Filter by Area.
    params.filterByArea, params.minArea = True, min_area
    # Filter by Circularity
    params.filterByCircularity, params.minCircularity = True, min_circularity
    # Filter by Convexity
    params.filterByConvexity, params.minConvexity = True, min_convexity
    # Filter by Inertia
    params.filterByInertia, params.minInertiaRatio= True, min_inertia
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(ir_image)
    return keypoints

def draw_blob(image, keypoints):
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(len(keypoints)):
        cv2.circle(im_with_keypoints, ir_ref_pts[i], 2, (0, 255, 0), -1)
        cv2.putText(im_with_keypoints, str(i),(ir_ref_pts[i][0], ir_ref_pts[i][1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
    return im_with_keypoints

def keypoints_classifier(keypoints):
    # Convert keypoints to readable format
    readable_keypts = cv2.KeyPoint_convert(keypoints).astype('int16')
    # print(readable_keypts)
    top_left = min(readable_keypts, key=lambda p: (p[0] + p[1]))  # Smallest x + y (close to top-left)
    top_right = max(readable_keypts, key=lambda p: (p[0] - p[1]))  # Largest x - y (close to top-right)
    bottom_left = min(readable_keypts, key=lambda p: (p[0] - p[1]))  # Smallest x - y (close to bottom-left)
    bottom_right = max(readable_keypts, key=lambda p: (p[0] + p[1]))  # Largest x + y (close to bottom-right)

    
    ir_ref_pts = np.array([top_left, bottom_left, bottom_right, top_right])
    # print(ir_ref_pts)
    body_points = np.array([x for x in readable_keypts if x not in ir_ref_pts])
    return ir_ref_pts,body_points

def homography_transform(image1, image2, ref_pts1, ref_pts2,overlay_perc=0.3):
    # Find homography matrix
    M, _ = cv2.findHomography(ref_pts1, ref_pts2, cv2.RANSAC)

    # Warp image2 to align with image1
    aligned_img = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

    # Overlay images
    overlay = cv2.addWeighted(image2, overlay_perc, aligned_img, 1-overlay_perc, 0)

    return aligned_img, overlay
        
################################################################################

col_image = cv2.imread(r"Custom_Results\sample1_Color.png")
ir_image = cv2.imread(r"Custom_Results\sample1_IR.png", cv2.IMREAD_GRAYSCALE)

marked_positions,corners,ids = detect_aruco(col_image)
print(marked_positions)

aruco_ref_pts = []
for i in range(4):
	aruco_ref_pts.append(marked_positions[i])
aruco_ref_pts = np.array(aruco_ref_pts)

# col_image = draw_aruco(col_image, corners, ids)
# cv2.imshow("Image", col_image)
# cv2.waitKey(0)

camera_matrix = np.array([[1.80376226e+03,0.00000000e+00,3.16914171e+02],
 [0.00000000e+00,4.31043121e+03,2.87652674e+02],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
 )
dist_coeffs = np.array([[-3.49027957e+00,3.13640856e+01,8.12342618e-02,3.06527531e-02,-3.86407135e+02]])
ir_image = undistort_image(ir_image, camera_matrix, dist_coeffs)

keypoints = detect_blob(ir_image)

for i in keypoints:
    cv2.circle(ir_image, (int(i.pt[0]), int(i.pt[1])), 2, (0, 255, 0), -1)

ir_ref_pts,body_points = keypoints_classifier(keypoints)
# print(ir_ref_pts)

ir_image = preprocess_ir_data(ir_image, 255)
ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

# for i in body_points:
#     cv2.drawMarker(ir_image, (i[0], i[1]), (0, 0, 255), markerType=cv2.MARKER_STAR, thickness=1,markerSize=10)
#     print(i)

# im_with_keypoints = draw_blob(ir_image, keypoints)
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)


rgbtoir_aligned_img, rgbtoir_overlay = homography_transform(col_image, ir_image, aruco_ref_pts, ir_ref_pts, overlay_perc=0.5)
# irtorgb_aligned_img, irtorgb_overlay = homography_transform(ir_image, col_image, ir_ref_pts, aruco_ref_pts)

# Show results
# cv2.imshow("IR scaled to RGB",irtorgb_overlay)
cv2.imshow("RGB scaled to IR",rgbtoir_overlay)
# cv2.imwrite("Custom_Results/rgb_scaled_to_ir_image.png", rgbtoir_overlay)
# cv2.imwrite("Custom_Results/ir_scaled_to_rgb_image.png", irtorgb_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
