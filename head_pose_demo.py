"""
Head Pose Estimation Demo
Demonstrates how head orientation is calculated using facial landmarks and PnP algorithm
Uses the EXACT implementation from MonitorTest1/head_pose.py
"""

import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import time


class HeadPoseEstimator:
    """
    Estimates head pose (orientation) using facial landmarks and PnP (Perspective-n-Point) algorithm
    This is the EXACT implementation from MonitorTest1
    
    The algorithm:
    1. Uses 6 key facial landmarks as 2D reference points (indices: 1, 152, 263, 33, 287, 57)
    2. Maps them to a 3D face model
    3. Uses cv2.solvePnP to find rotation and translation vectors
    4. Uses cv2.decomposeProjectionMatrix to extract Euler angles (yaw, pitch, roll)
    """
    
    def __init__(self, yaw_threshold=20, pitch_threshold=20, hold_seconds=3):
        self.face_detector = FaceMeshDetector(maxFaces=1)
        
        # Thresholds from MonitorTest1
        self.YAW_THRESHOLD = yaw_threshold
        self.PITCH_THRESHOLD = pitch_threshold
        self.HEAD_TURN_SECONDS = hold_seconds
        self.head_turned = False
        self.turned_start_time = 0.0
        
        # 3D model points - EXACT values from MonitorTest1
        # Landmarks: [1, 152, 263, 33, 287, 57]
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip (landmark 1)
            (0.0, -63.6, -12.5),       # Chin (landmark 152)
            (-43.3, 32.7, -26.0),      # Right eye right corner (landmark 263)
            (43.3, 32.7, -26.0),       # Left eye left corner (landmark 33)
            (-28.9, -28.9, -24.1),     # Right mouth corner (landmark 287)
            (28.9, -28.9, -24.1)       # Left mouth corner (landmark 57)
        ])
        
        # Corresponding 2D landmark indices in MediaPipe Face Mesh
        self.landmark_indices = [1, 152, 263, 33, 287, 57]
        
        # Head direction tracking
        self.yaw_history = []
        self.pitch_history = []
        self.history_size = 10
        
        # Direction state
        self.current_direction = "Forward"
        self.direction_start_time = None
        self.direction_durations = {
            "Forward": 0,
            "Left": 0,
            "Right": 0,
            "Up": 0,
            "Down": 0
        }
    
    def check(self, img, face, now, on_warning=None):
        """
        EXACT implementation from MonitorTest1
        Check head pose and return yaw, pitch angles
        """
        # Extract 2D image points - EXACT order from MonitorTest1
        image_points = np.array([
            face[1], face[152], face[263], face[33], face[287], face[57]
        ], dtype="double")

        # Get frame dimensions
        h, w = img.shape[:2]
        
        # Camera matrix - same as MonitorTest1
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            return None

        # Convert to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # IMPORTANT: MonitorTest1 uses decomposeProjectionMatrix, not manual Euler conversion
        proj = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        pitch, yaw, roll = [angle[0] for angle in euler]

        # Display angles on frame (as in MonitorTest1)
        cv2.putText(img, f"Yaw: {int(yaw)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, f"Pitch: {int(pitch)}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Check if head is turned (warning logic from MonitorTest1)
        if abs(yaw) > self.YAW_THRESHOLD or abs(pitch) > self.PITCH_THRESHOLD:
            if not self.head_turned:
                self.turned_start_time = now
                self.head_turned = True
            if now - self.turned_start_time > self.HEAD_TURN_SECONDS:
                cv2.putText(img, "WARNING: LOOKING AWAY!", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if on_warning:
                    on_warning(now)
        else:
            self.head_turned = False

        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'rvec': rvec,
            'tvec': tvec,
            'image_points': image_points,
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs
        }
    
    def estimate_pose(self, landmarks, frame, now):
        """
        Wrapper to use MonitorTest1's check method
        Returns enhanced data for visualization
        """
        result = self.check(frame, landmarks, now)
        if result is None:
            return None
        
        yaw = result['yaw']
        pitch = result['pitch']
        roll = result['roll']
        
        # Smooth angles with history
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)
        
        if len(self.yaw_history) > self.history_size:
            self.yaw_history.pop(0)
            self.pitch_history.pop(0)
        
        smooth_yaw = np.mean(self.yaw_history)
        smooth_pitch = np.mean(self.pitch_history)
        
        # Project 3D axis points for visualization
        axis_points_3d = np.float32([
            [0, 0, 0],      # Origin (nose)
            [50, 0, 0],     # X-axis (red, pointing right)
            [0, 50, 0],     # Y-axis (green, pointing down)
            [0, 0, 50]      # Z-axis (blue, pointing forward)
        ])
        
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d,
            result['rvec'],
            result['tvec'],
            result['camera_matrix'],
            result['dist_coeffs']
        )
        
        # Get nose tip for drawing
        nose_tip = tuple(map(int, result['image_points'][0]))
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'smooth_yaw': smooth_yaw,
            'smooth_pitch': smooth_pitch,
            'rotation_vector': result['rvec'],
            'translation_vector': result['tvec'],
            'axis_points_2d': axis_points_2d,
            'nose_tip': nose_tip,
            'image_points': result['image_points']
        }
    
    def get_head_direction(self, yaw, pitch, yaw_threshold=20, pitch_threshold=20):
        """
        Determine head direction based on yaw and pitch angles
        
        Args:
            yaw: horizontal rotation (negative = left, positive = right)
            pitch: vertical rotation (negative = up, positive = down)
            yaw_threshold: threshold in degrees for left/right detection
            pitch_threshold: threshold in degrees for up/down detection
        """
        direction = "Forward"
        
        # Check vertical direction first (prioritize up/down)
        if pitch < -pitch_threshold:
            direction = "Up"
        elif pitch > pitch_threshold:
            direction = "Down"
        # Then check horizontal direction
        elif yaw < -yaw_threshold:
            direction = "Left"
        elif yaw > yaw_threshold:
            direction = "Right"
        
        return direction
    
    def update_direction_tracking(self, direction):
        """Track duration of each head direction"""
        current_time = time.time()
        
        if self.direction_start_time is None:
            self.direction_start_time = current_time
            self.current_direction = direction
            return
        
        # If direction changed
        if direction != self.current_direction:
            # Update duration for previous direction
            duration = current_time - self.direction_start_time
            self.direction_durations[self.current_direction] += duration
            
            # Start tracking new direction
            self.current_direction = direction
            self.direction_start_time = current_time
        else:
            # Update current direction duration
            duration = current_time - self.direction_start_time
    
    def draw_axis(self, frame, pose_data):
        """Draw 3D coordinate axes on the face"""
        if not pose_data:
            return
        
        nose_tip = pose_data['nose_tip']
        axis_points = pose_data['axis_points_2d']
        
        # Draw axes (origin to each axis endpoint)
        origin = tuple(map(int, axis_points[0].ravel()))
        
        # X-axis (RED) - points to the right
        x_axis = tuple(map(int, axis_points[1].ravel()))
        cv2.line(frame, origin, x_axis, (0, 0, 255), 3)
        cv2.putText(frame, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Y-axis (GREEN) - points down
        y_axis = tuple(map(int, axis_points[2].ravel()))
        cv2.line(frame, origin, y_axis, (0, 255, 0), 3)
        cv2.putText(frame, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Z-axis (BLUE) - points forward
        z_axis = tuple(map(int, axis_points[3].ravel()))
        cv2.line(frame, origin, z_axis, (255, 0, 0), 3)
        cv2.putText(frame, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    def draw_landmarks(self, frame, pose_data):
        """Draw the 6 key landmarks used for PnP"""
        if not pose_data:
            return
        
        image_points = pose_data['image_points']
        
        for i, point in enumerate(image_points):
            pt = tuple(map(int, point))
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)
            cv2.putText(frame, str(i), (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


def draw_angle_bars(frame, yaw, pitch, roll):
    """Draw angle indicator bars"""
    x_start = frame.shape[1] - 250
    y_start = 50
    bar_width = 200
    bar_height = 15
    
    # Helper function to draw a bar
    def draw_bar(y_pos, angle, label, max_angle=90):
        # Background bar
        cv2.rectangle(frame, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height),
                     (50, 50, 50), -1)
        
        # Filled bar (proportional to angle)
        fill_width = int(bar_width * (angle / (2 * max_angle)) + bar_width / 2)
        fill_width = max(0, min(bar_width, fill_width))
        
        color = (0, 255, 0) if abs(angle) < 20 else (0, 165, 255) if abs(angle) < 40 else (0, 0, 255)
        cv2.rectangle(frame, (x_start, y_pos), (x_start + fill_width, y_pos + bar_height),
                     color, -1)
        
        # Center line
        center_x = x_start + bar_width // 2
        cv2.line(frame, (center_x, y_pos), (center_x, y_pos + bar_height), (255, 255, 255), 1)
        
        # Label and value
        cv2.putText(frame, f"{label}: {angle:.1f}°", (x_start, y_pos - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw bars for each angle
    draw_bar(y_start, yaw, "Yaw (L/R)")
    draw_bar(y_start + 40, pitch, "Pitch (U/D)")
    draw_bar(y_start + 80, roll, "Roll")


def draw_info_panel(frame, pose_data, direction, fps, warning_active):
    """Draw comprehensive information panel"""
    if not pose_data:
        cv2.putText(frame, "No face detected", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (550, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "MonitorTest1 Head Pose Implementation", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset = 70
    line_height = 22
    
    # Method info
    cv2.putText(frame, "Method: cv2.solvePnP + decomposeProjectionMatrix",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += line_height
    
    # Landmarks used
    cv2.putText(frame, "Landmarks: [1, 152, 263, 33, 287, 57]",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += line_height
    
    # Raw angles
    cv2.putText(frame, f"Yaw: {pose_data['yaw']:.2f}° (Left/Right turn)",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame, f"Pitch: {pose_data['pitch']:.2f}° (Up/Down tilt)",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame, f"Roll: {pose_data['roll']:.2f}° (Shoulder tilt)",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
    y_offset += line_height
    
    # Smoothed angles (for direction classification)
    cv2.putText(frame, f"Smooth Yaw: {pose_data['smooth_yaw']:.2f}°",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame, f"Smooth Pitch: {pose_data['smooth_pitch']:.2f}°",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += line_height
    
    # Direction with color coding
    dir_color = (0, 255, 0) if direction == "Forward" else (0, 165, 255)
    cv2.putText(frame, f"Head Direction: {direction}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, dir_color, 2)
    y_offset += line_height
    
    # Warning state
    if warning_active:
        cv2.putText(frame, "Warning Active: Looking away > 3s", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 220),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_direction_compass(frame, direction):
    """Draw a compass showing current head direction"""
    center_x = frame.shape[1] - 100
    center_y = frame.shape[0] - 100
    radius = 60
    
    # Draw compass circle
    cv2.circle(frame, (center_x, center_y), radius, (200, 200, 200), 2)
    
    # Draw cardinal directions
    directions = {
        "Up": (center_x, center_y - radius + 10),
        "Down": (center_x, center_y + radius + 20),
        "Left": (center_x - radius - 30, center_y + 5),
        "Right": (center_x + radius + 10, center_y + 5),
        "Forward": (center_x - 25, center_y + 5)
    }
    
    for dir_name, pos in directions.items():
        color = (0, 255, 0) if dir_name == direction else (150, 150, 150)
        size = 0.6 if dir_name == direction else 0.4
        thickness = 2 if dir_name == direction else 1
        cv2.putText(frame, dir_name[0], pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    
    # Draw center indicator
    indicator_color = (0, 255, 0) if direction == "Forward" else (0, 165, 255)
    cv2.circle(frame, (center_x, center_y), 8, indicator_color, -1)


def main():
    """Main function to run the head pose demo"""
    print("=" * 80)
    print("Head Pose Estimation Demo - MonitorTest1 Implementation")
    print("=" * 80)
    print("\nThis demo uses the EXACT implementation from MonitorTest1/head_pose.py")
    print("\nTechnical Details:")
    print("  - Uses 6 facial landmarks: [1, 152, 263, 33, 287, 57]")
    print("  - 3D model points (exact values from MonitorTest1):")
    print("    * Nose tip (1): (0.0, 0.0, 0.0)")
    print("    * Chin (152): (0.0, -63.6, -12.5)")
    print("    * Right eye (263): (-43.3, 32.7, -26.0)")
    print("    * Left eye (33): (43.3, 32.7, -26.0)")
    print("    * Right mouth (287): (-28.9, -28.9, -24.1)")
    print("    * Left mouth (57): (28.9, -28.9, -24.1)")
    print("\nAlgorithm Steps:")
    print("  1. cv2.solvePnP: Maps 2D landmarks → 3D model → rotation & translation")
    print("  2. cv2.Rodrigues: Converts rotation vector → rotation matrix")
    print("  3. cv2.decomposeProjectionMatrix: Extracts Euler angles")
    print("\nAngle Meanings:")
    print("  - Yaw: Left/Right head turn")
    print("  - Pitch: Up/Down head tilt")
    print("  - Roll: Shoulder tilt")
    print("\nWarning Logic (from MonitorTest1):")
    print("  - Triggers when |yaw| > 20° OR |pitch| > 20°")
    print("  - Warning displays after 3 seconds of continuous head turn")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset direction tracking")
    print("=" * 80)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize estimator with MonitorTest1 defaults
    estimator = HeadPoseEstimator(yaw_threshold=20, pitch_threshold=20, hold_seconds=3)
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    
    print("\nStarting camera feed...")
    print("Try moving your head in different directions!\n")
    
    warning_active = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get current time
        curr_time = time.time()
        
        # Detect face mesh
        frame_mesh, faces = estimator.face_detector.findFaceMesh(frame, draw=False)
        
        pose_data = None
        direction = "No Face"
        warning_active = False
        
        if faces:
            landmarks = faces[0]
            
            # Use MonitorTest1's estimate_pose which wraps the check method
            pose_data = estimator.estimate_pose(landmarks, frame, curr_time)
            
            if pose_data:
                # Get direction
                direction = estimator.get_head_direction(
                    pose_data['smooth_yaw'],
                    pose_data['smooth_pitch']
                )
                
                # Update tracking
                estimator.update_direction_tracking(direction)
                
                # Check if warning is active
                warning_active = estimator.head_turned and (curr_time - estimator.turned_start_time > estimator.HEAD_TURN_SECONDS)
                
                # Draw visualizations
                estimator.draw_landmarks(frame, pose_data)
                estimator.draw_axis(frame, pose_data)
                draw_angle_bars(frame, pose_data['smooth_yaw'], 
                              pose_data['smooth_pitch'], pose_data['roll'])
                draw_direction_compass(frame, direction)
        
        # Calculate FPS
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Draw info panel
        draw_info_panel(frame, pose_data, direction, fps, warning_active)
        
        # Draw legend
        legend_y = frame.shape[0] - 180
        cv2.rectangle(frame, (5, legend_y - 25), (400, frame.shape[0] - 5), (0, 0, 0), -1)
        cv2.putText(frame, "MonitorTest1 Implementation:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "YELLOW circles: 6 landmarks [1,152,263,33,287,57]", (10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "RED (X): Right, GREEN (Y): Down, BLUE (Z): Forward", (10, legend_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "Thresholds: |yaw|>20 or |pitch|>20 for 3s = WARNING", (10, legend_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "Yellow text (150, 180): Yaw & Pitch from MonitorTest1", (10, legend_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Display frame
        cv2.imshow("Head Pose - MonitorTest1 Implementation", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            estimator.direction_durations = {k: 0 for k in estimator.direction_durations}
            estimator.direction_start_time = None
            estimator.head_turned = False
            estimator.turned_start_time = 0.0
            print("Direction tracking reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 80)
    print("Session Summary - Direction Durations:")
    print("=" * 80)
    for direction, duration in estimator.direction_durations.items():
        print(f"  {direction:10s}: {duration:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
