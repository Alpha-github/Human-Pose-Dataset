import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Initialize the blob detector
params = cv2.SimpleBlobDetector_Params()

# Change Color
params.filterByColor = True
params.blobColor = 255
 
# Change thresholds
params.minThreshold = 185;
params.maxThreshold = 255;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 20
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.01
 
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.01
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Tracking history
tracked_points = {}
next_id = 0  # Unique ID for new points

# Function to compute cost matrix (Euclidean distance)
def compute_cost_matrix(previous_points, current_points):
    cost_matrix = np.zeros((len(previous_points), len(current_points)))
    for i, (px, py) in enumerate(previous_points):
        for j, (cx, cy) in enumerate(current_points):
            cost_matrix[i, j] = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return cost_matrix

# Read video
# cap = cv2.VideoCapture(r'Custom_Results\infrared_recording_1738923613.avi')
cap = cv2.VideoCapture(r'Custom_Results\infrared_recording_1738923613.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(gray)
    current_points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    
    new_tracked_points = {}
    
    if tracked_points:
        previous_ids = list(tracked_points.keys())
        previous_points = [tracked_points[i] for i in previous_ids]
        
        cost_matrix = compute_cost_matrix(previous_points, current_points)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 50:  # Threshold for assignment
                new_tracked_points[previous_ids[r]] = current_points[c]
                assigned.add(c)
        
        for i, (cx, cy) in enumerate(current_points):
            if i not in assigned:
                new_tracked_points[next_id] = (cx, cy)
                next_id += 1
    else:
        for cx, cy in current_points:
            new_tracked_points[next_id] = (cx, cy)
            next_id += 1
    
    tracked_points = new_tracked_points
    
    for tid, (x, y) in tracked_points.items():
        # cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(frame, str(tid), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
