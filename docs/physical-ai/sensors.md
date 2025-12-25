# Robot Sensor Systems: Cameras, LiDAR, IMU, and Force Sensors

## Concept

Robot sensor systems form the foundation of perception in Physical AI, providing the raw data that enables robots to understand and interact with their environment. Think of these sensors as the robot's "senses" - just as humans use vision, touch, balance, and proprioception to navigate the world, robots rely on cameras, LiDAR, IMU, and force sensors to perceive their surroundings and their own state.

Each sensor type provides unique information:
- **Cameras** provide rich visual information including color, texture, and shape
- **LiDAR** provides precise distance measurements in 3D space
- **IMU (Inertial Measurement Unit)** provides orientation, acceleration, and angular velocity
- **Force sensors** provide information about physical interaction with objects

These sensors matter because they form the critical interface between the robot's digital intelligence and the physical world. Without accurate sensor data, even the most sophisticated AI algorithms would be unable to operate effectively in real-world environments. The fusion of data from multiple sensors enables robots to build comprehensive models of their environment and their own state, which is essential for safe and effective operation.

If you're familiar with how humans integrate information from multiple senses, robot sensor fusion works similarly. Just as you might use visual information, balance, and touch to navigate a dark room, robots combine data from cameras, LiDAR, IMU, and force sensors to create a coherent understanding of their environment and their interactions with it.

## ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   ROBOT SENSOR ECOSYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │   ENVIRONMENT   │    │        SENSOR FUSION             │   │
│  │                 │    │                                  │   │
│  │  ┌───────────┐  │    │  ┌────────────────────────────┐  │   │
│  │  │   OBJECTS │  │    │  │     PERCEPTION SYSTEM      │  │   │
│  │  │  Recognition│ ◄───────►                            │  │   │
│  │  └───────────┘  │    │  │  • 3D Environment Model   │  │   │
│  │                 │    │  │  • Object Detection       │  │   │
│  │  ┌───────────┐  │    │  │  • State Estimation       │  │   │
│  │  │  OBSTACLES│  │    │  │  • Motion Tracking        │  │   │
│  │  │  Detection│ ◄───────►                            │  │   │
│  │  └───────────┘  │    │  └────────────────────────────┘  │   │
│  │                 │    │                                  │   │
│  │  ┌───────────┐  │    │  ┌────────────────────────────┐  │   │
│  │  │  HUMANS   │  │    │  │      CONTROL SYSTEM        │  │   │
│  │  │  Tracking│ ◄───────►                            │  │   │
│  │  └───────────┘  │    │  │  • Balance Control        │  │   │
│  │                 │    │  │  • Navigation Planning     │  │   │
│  │  ┌───────────┐  │    │  │  • Manipulation Control   │  │   │
│  │  │  SURFACES │  │    │  │  • Safety Monitoring      │  │   │
│  │  │  Mapping  │ ◄───────►                            │  │   │
│  │  └───────────┘  │    │  └────────────────────────────┘  │   │
│  └─────────────────┘    └──────────────────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                   INDIVIDUAL SENSORS                            │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   CAMERA    │    │    LIDAR    │    │     IMU     │        │
│  │             │    │             │    │             │        │
│  │ • RGB Data  │    │ • Distance  │    │ • Orientation│        │
│  │ • Texture   │    │ • 3D Points │    │ • Acceleration│       │
│  │ • Color     │    │ • Obstacles │    │ • Angular Vel│        │
│  │ • Shape     │    │ • Mapping   │    │ • Stability │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────────────┤
│  │                SENSOR FUSION PROCESSING                     │
│  └─────────────────────────────────────────────────────────────┤
│                                    │                           │
│                                    ▼                           │
│                    ┌─────────────────────────────────────────┐ │
│                    │           FORCE SENSORS                 │ │
│                    │                                         │ │
│                    │ • Contact Detection                     │ │
│                    │ • Grasp Force Monitoring                │ │
│                    │ • Interaction Feedback                  │ │
│                    │ • Safety Force Limiting                 │ │
│                    └─────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This diagram illustrates the robot sensor ecosystem with individual sensors feeding into sensor fusion processing, which then provides information to perception and control systems.

## Real-world Analogy

Think of robot sensor systems like a team of specialized detectives, each with their own expertise, working together to solve the case of "what's happening in the robot's environment."

The camera detective is excellent at recognizing visual patterns, colors, and textures - they can spot a red cup on a table or identify a person's face. The LiDAR detective is the master of precise measurements - they can tell you exactly how far away that cup is and map out the entire room in 3D. The IMU detective is the balance and motion expert - they know if the robot is tilting, accelerating, or spinning. The force sensor detective is the tactile expert - they know when the robot is touching something and how hard.

Just as a detective team combines their different expertise to build a complete picture of a case, the robot combines data from all these sensors to understand its world. No single sensor can provide a complete picture, but together they create a rich, comprehensive understanding that enables intelligent behavior.

Just as you might use your eyes to see an object, your sense of balance to stay upright, and your sense of touch to feel texture, a robot uses its sensors to build a complete picture of its environment and its interactions with it.

## Pseudo-code (ROS 2 / Python style)

```python
# Example Robot Sensor System Implementation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, JointState, CameraInfo
from geometry_msgs.msg import Vector3, Quaternion, Point, PoseStamped
from std_msgs.msg import Float32MultiArray, Header
from builtin_interfaces.msg import Time
import numpy as np
import cv2
from cv_bridge import CvBridge
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
from collections import deque
import math
from scipy.spatial.transform import Rotation as R

class SensorData:
    """Container for sensor data with timestamps"""
    def __init__(self):
        self.camera_image: Optional[np.ndarray] = None
        self.camera_timestamp: Optional[Time] = None
        self.lidar_data: Optional[np.ndarray] = None
        self.lidar_timestamp: Optional[Time] = None
        self.imu_data: Optional[Dict] = None
        self.imu_timestamp: Optional[Time] = None
        self.force_data: Optional[Dict] = None
        self.force_timestamp: Optional[Time] = None
        self.joint_data: Optional[Dict] = None
        self.joint_timestamp: Optional[Time] = None

class CameraSensorNode(Node):
    """Handles camera sensor data processing"""
    def __init__(self):
        super().__init__('camera_sensor')

        # Create subscriber for camera image
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create subscriber for camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)

        # Publishers for processed camera data
        self.object_detection_pub = self.create_publisher(
            Float32MultiArray, '/camera/object_detections', 10)
        self.feature_pub = self.create_publisher(
            Float32MultiArray, '/camera/features', 10)

        # Internal state
        self.cv_bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        self.image_callback_count = 0

        # Timer for processing loop
        self.process_timer = self.create_timer(0.1, self.process_camera_data)

    def image_callback(self, msg: Image):
        """Process incoming camera image"""
        try:
            # Convert ROS Image message to OpenCV format
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_callback_count += 1

            # Log every 10th image to avoid spam
            if self.image_callback_count % 10 == 0:
                self.get_logger().info(f'Received camera image: {self.latest_image.shape}')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def camera_info_callback(self, msg: CameraInfo):
        """Process camera calibration information"""
        self.camera_info = msg
        self.get_logger().debug(f'Camera info updated: {msg.width}x{msg.height}')

    def process_camera_data(self):
        """Process camera data and publish results"""
        if self.latest_image is not None:
            # Perform object detection
            objects = self.detect_objects(self.latest_image)

            # Extract features
            features = self.extract_features(self.latest_image)

            # Publish object detections
            if objects:
                obj_msg = Float32MultiArray()
                obj_msg.data = self.pack_object_data(objects)
                self.object_detection_pub.publish(obj_msg)

            # Publish features
            if features:
                feat_msg = Float32MultiArray()
                feat_msg.data = features
                self.feature_pub.publish(feat_msg)

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in the camera image"""
        # In a real implementation, this would use a trained object detection model
        # For this example, we'll simulate object detection using simple color-based detection

        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for common objects (red, blue, green)
        color_ranges = [
            {'name': 'red_object', 'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            {'name': 'blue_object', 'lower': np.array([100, 50, 50]), 'upper': np.array([130, 255, 255])},
            {'name': 'green_object', 'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])}
        ]

        objects = []
        for color_range in color_ranges:
            # Create mask for the color
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter out small contours
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2

                    objects.append({
                        'name': color_range['name'],
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(0.9, area / 10000)  # Simple confidence based on size
                    })

        return objects

    def extract_features(self, image: np.ndarray) -> List[float]:
        """Extract visual features from the image"""
        # In a real implementation, this would use SIFT, ORB, or deep features
        # For this example, we'll use simple statistical features

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate basic statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        # Calculate color histogram (simplified)
        hist_b = cv2.calcHist([image], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [8], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [8], [0, 256])

        # Flatten and normalize histograms
        color_features = []
        for hist in [hist_b, hist_g, hist_r]:
            hist_flat = hist.flatten()
            hist_norm = hist_flat / np.sum(hist_flat) if np.sum(hist_flat) > 0 else hist_flat
            color_features.extend(hist_norm)

        # Combine all features
        features = [mean_intensity, std_intensity, edge_density]
        features.extend(color_features)

        return features

    def pack_object_data(self, objects: List[Dict]) -> List[float]:
        """Pack object detection data into a flat list"""
        packed = []
        packed.append(len(objects))  # Number of objects

        for obj in objects:
            packed.append(obj['center'][0])  # x center
            packed.append(obj['center'][1])  # y center
            packed.append(obj['bbox'][2])    # width
            packed.append(obj['bbox'][3])    # height
            packed.append(obj['area'])       # area
            packed.append(obj['confidence']) # confidence
            # Add more features as needed
            packed.extend([0.0, 0.0, 0.0, 0.0])  # Padding for future features

        # Pad to fixed size for consistency
        while len(packed) < 100:
            packed.append(0.0)

        return packed[:100]

class LidarSensorNode(Node):
    """Handles LiDAR sensor data processing"""
    def __init__(self):
        super().__init__('lidar_sensor')

        # Create subscriber for LiDAR point cloud
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/scan', self.lidar_callback, 10)

        # Publishers for processed LiDAR data
        self.obstacle_pub = self.create_publisher(
            Float32MultiArray, '/lidar/obstacles', 10)
        self.map_pub = self.create_publisher(
            Float32MultiArray, '/lidar/map', 10)

        # Internal state
        self.latest_lidar: Optional[PointCloud2] = None
        self.lidar_callback_count = 0

        # Timer for processing loop
        self.process_timer = self.create_timer(0.05, self.process_lidar_data)

    def lidar_callback(self, msg: PointCloud2):
        """Process incoming LiDAR data"""
        self.latest_lidar = msg
        self.lidar_callback_count += 1

        if self.lidar_callback_count % 20 == 0:  # Log every 20th message
            self.get_logger().info(f'Received LiDAR data with {msg.height * msg.width} points')

    def process_lidar_data(self):
        """Process LiDAR data and publish results"""
        if self.latest_lidar is not None:
            # Parse point cloud (in a real implementation, use proper parsing)
            # For this example, we'll simulate processing
            obstacles = self.detect_obstacles_from_lidar()
            map_data = self.create_2d_map_from_lidar()

            # Publish obstacles
            if obstacles:
                obs_msg = Float32MultiArray()
                obs_msg.data = self.pack_obstacle_data(obstacles)
                self.obstacle_pub.publish(obs_msg)

            # Publish map
            if map_data:
                map_msg = Float32MultiArray()
                map_msg.data = map_data
                self.map_pub.publish(map_msg)

    def detect_obstacles_from_lidar(self) -> List[Dict]:
        """Detect obstacles from LiDAR data"""
        # In a real implementation, this would parse the PointCloud2 message properly
        # For this example, we'll simulate obstacle detection

        # Simulate 5 obstacles at various positions
        obstacles = []
        for i in range(5):
            angle = i * 2 * math.pi / 5  # Evenly distributed
            distance = 1.0 + i * 0.5  # Increasing distance

            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            obstacles.append({
                'position': (x, y, 0.0),
                'distance': distance,
                'size': 0.3,  # Estimated size
                'confidence': 0.9
            })

        return obstacles

    def create_2d_map_from_lidar(self) -> List[float]:
        """Create a 2D occupancy grid from LiDAR data"""
        # In a real implementation, this would create a proper occupancy grid
        # For this example, we'll simulate a simple grid

        # Create a 20x20 grid (400 values)
        grid_size = 20
        grid = [0.0] * (grid_size * grid_size)  # Initialize as free space

        # Add some obstacles based on simulated detection
        obstacles = self.detect_obstacles_from_lidar()
        for obs in obstacles:
            # Convert world coordinates to grid coordinates
            x, y, _ = obs['position']
            grid_x = int((x + 5.0) * grid_size / 10.0)  # Map -5m to 5m to 0-20
            grid_y = int((y + 5.0) * grid_size / 10.0)  # Map -5m to 5m to 0-20

            # Ensure within bounds
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                grid[grid_x * grid_size + grid_y] = 1.0  # Mark as occupied

        return grid

    def pack_obstacle_data(self, obstacles: List[Dict]) -> List[float]:
        """Pack obstacle data into a flat list"""
        packed = []
        packed.append(len(obstacles))  # Number of obstacles

        for obs in obstacles:
            packed.extend(obs['position'])  # x, y, z
            packed.append(obs['distance'])
            packed.append(obs['size'])
            packed.append(obs['confidence'])

        # Pad to fixed size
        while len(packed) < 50:
            packed.append(0.0)

        return packed[:50]

class IMUSensorNode(Node):
    """Handles IMU sensor data processing"""
    def __init__(self):
        super().__init__('imu_sensor')

        # Create subscriber for IMU data
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Publishers for processed IMU data
        self.orientation_pub = self.create_publisher(
            Float32MultiArray, '/imu/orientation', 10)
        self.motion_pub = self.create_publisher(
            Float32MultiArray, '/imu/motion', 10)

        # Internal state
        self.latest_imu: Optional[Imu] = None
        self.orientation_history = deque(maxlen=10)
        self.imu_callback_count = 0

        # Timer for processing loop
        self.process_timer = self.create_timer(0.01, self.process_imu_data)

    def imu_callback(self, msg: Imu):
        """Process incoming IMU data"""
        self.latest_imu = msg
        self.imu_callback_count += 1

        # Store orientation for history
        if self.latest_imu:
            orientation = (
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w
            )
            self.orientation_history.append(orientation)

    def process_imu_data(self):
        """Process IMU data and publish results"""
        if self.latest_imu is not None:
            # Extract orientation
            orientation = self.extract_orientation()

            # Extract motion data
            motion = self.extract_motion()

            # Publish orientation
            if orientation:
                orient_msg = Float32MultiArray()
                orient_msg.data = orientation
                self.orientation_pub.publish(orient_msg)

            # Publish motion
            if motion:
                motion_msg = Float32MultiArray()
                motion_msg.data = motion
                self.motion_pub.publish(motion_msg)

    def extract_orientation(self) -> List[float]:
        """Extract orientation data from IMU"""
        if not self.latest_imu:
            return [0.0, 0.0, 0.0, 1.0]  # Default quaternion (no rotation)

        # Extract quaternion
        q = self.latest_imu.orientation
        quaternion = [q.x, q.y, q.z, q.w]

        # Convert to Euler angles for additional information
        euler = self.quaternion_to_euler((q.x, q.y, q.z, q.w))

        # Combine quaternion and Euler angles
        result = quaternion
        result.extend(euler)

        return result

    def extract_motion(self) -> List[float]:
        """Extract motion data from IMU"""
        if not self.latest_imu:
            return [0.0] * 6  # Default: no motion

        # Extract angular velocity
        av = self.latest_imu.angular_velocity
        angular_vel = [av.x, av.y, av.z]

        # Extract linear acceleration
        la = self.latest_imu.linear_acceleration
        linear_acc = [la.x, la.y, la.z]

        # Combine
        result = angular_vel
        result.extend(linear_acc)

        return result

    def quaternion_to_euler(self, quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]

class ForceSensorNode(Node):
    """Handles force/torque sensor data processing"""
    def __init__(self):
        super().__init__('force_sensor')

        # Create subscriber for force/torque data
        # Using a generic message for force data
        self.force_sub = self.create_subscription(
            Float32MultiArray, '/wrist_force', self.force_callback, 10)

        # Publishers for processed force data
        self.contact_pub = self.create_publisher(
            Float32MultiArray, '/force/contact', 10)
        self.grasp_pub = self.create_publisher(
            Float32MultiArray, '/force/grasp', 10)

        # Internal state
        self.latest_force: Optional[List[float]] = None
        self.force_callback_count = 0

        # Timer for processing loop
        self.process_timer = self.create_timer(0.005, self.process_force_data)

    def force_callback(self, msg: Float32MultiArray):
        """Process incoming force data"""
        self.latest_force = list(msg.data)
        self.force_callback_count += 1

        if self.force_callback_count % 100 == 0:  # Log every 100th message
            self.get_logger().info(f'Received force data: {self.latest_force[:6]}...')

    def process_force_data(self):
        """Process force data and publish results"""
        if self.latest_force and len(self.latest_force) >= 6:
            # Extract force and torque components
            force = self.latest_force[0:3]
            torque = self.latest_force[3:6]

            # Detect contact
            contact_info = self.detect_contact(force, torque)

            # Evaluate grasp quality
            grasp_info = self.evaluate_grasp(force, torque)

            # Publish contact information
            contact_msg = Float32MultiArray()
            contact_msg.data = contact_info
            self.contact_pub.publish(contact_msg)

            # Publish grasp information
            grasp_msg = Float32MultiArray()
            grasp_msg.data = grasp_info
            self.grasp_pub.publish(grasp_msg)

    def detect_contact(self, force: List[float], torque: List[float]) -> List[float]:
        """Detect contact based on force/torque data"""
        # Calculate force magnitude
        force_magnitude = math.sqrt(sum(f**2 for f in force))

        # Calculate torque magnitude
        torque_magnitude = math.sqrt(sum(t**2 for t in torque))

        # Threshold for contact detection (in Newtons and Newton-meters)
        force_threshold = 2.0
        torque_threshold = 0.5

        # Determine contact state
        is_contact = 1.0 if force_magnitude > force_threshold or torque_magnitude > torque_threshold else 0.0

        # Pack results
        result = [
            force_magnitude,
            torque_magnitude,
            is_contact,
            force[0], force[1], force[2],  # Individual force components
            torque[0], torque[1], torque[2]  # Individual torque components
        ]

        return result

    def evaluate_grasp(self, force: List[float], torque: List[float]) -> List[float]:
        """Evaluate grasp quality based on force/torque data"""
        # Calculate grasp stability metrics
        force_magnitude = math.sqrt(sum(f**2 for f in force))
        torque_magnitude = math.sqrt(sum(t**2 for t in torque))

        # Calculate force distribution (how evenly force is distributed across fingers)
        force_distribution = 1.0 - abs(force[0] - force[1]) / (force[0] + force[1] + 0.001)  # Avoid division by zero
        force_distribution = max(0.0, min(1.0, force_distribution))

        # Calculate grasp stability score
        # A stable grasp typically has moderate force and low torque
        stability_score = max(0.0, min(1.0, (5.0 - torque_magnitude) / 5.0)) * \
                         max(0.0, min(1.0, force_magnitude / 10.0))

        # Pack results
        result = [
            stability_score,
            force_distribution,
            force_magnitude,
            torque_magnitude,
            force[0], force[1], force[2],  # Individual force components
            torque[0], torque[1], torque[2]  # Individual torque components
        ]

        return result

class SensorFusionNode(Node):
    """Fuses data from multiple sensors to create comprehensive perception"""
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscriptions for all sensor data
        self.camera_sub = self.create_subscription(
            Float32MultiArray, '/camera/object_detections', self.camera_callback, 10)
        self.lidar_sub = self.create_subscription(
            Float32MultiArray, '/lidar/obstacles', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Float32MultiArray, '/imu/orientation', self.imu_callback, 10)
        self.force_sub = self.create_subscription(
            Float32MultiArray, '/force/contact', self.force_callback, 10)

        # Publishers for fused data
        self.perception_pub = self.create_publisher(
            Float32MultiArray, '/sensor_fusion/perception', 10)
        self.state_pub = self.create_publisher(
            Float32MultiArray, '/sensor_fusion/state', 10)

        # Internal state
        self.sensor_data = SensorData()
        self.fusion_callback_count = 0

        # Timer for fusion processing
        self.fusion_timer = self.create_timer(0.02, self.fuse_sensor_data)

    def camera_callback(self, msg: Float32MultiArray):
        """Process camera data"""
        # For fusion, we store the raw data and process it in the fusion loop
        pass

    def lidar_callback(self, msg: Float32MultiArray):
        """Process LiDAR data"""
        # For fusion, we store the raw data and process it in the fusion loop
        pass

    def imu_callback(self, msg: Float32MultiArray):
        """Process IMU data"""
        # For fusion, we store the raw data and process it in the fusion loop
        pass

    def force_callback(self, msg: Float32MultiArray):
        """Process force data"""
        # For fusion, we store the raw data and process it in the fusion loop
        pass

    def fuse_sensor_data(self):
        """Fuse data from multiple sensors"""
        # In a real implementation, this would perform sophisticated sensor fusion
        # using techniques like Kalman filtering, particle filtering, or neural networks
        # For this example, we'll simulate a simple fusion approach

        # Create a fused perception vector
        fused_data = []

        # Add camera-based object information (first 10 values from camera data)
        camera_objects = 3  # Simulated number of objects detected
        fused_data.extend([camera_objects, 1.0, 2.0, 0.5, 0.8, 1.5, 2.5, 0.6, 0.0, 0.0])

        # Add LiDAR-based obstacle information (next 10 values)
        lidar_obstacles = 2  # Simulated number of obstacles detected
        fused_data.extend([lidar_obstacles, 1.2, 0.8, 0.3, 0.9, 2.1, -0.5, 0.4, 0.0, 0.0])

        # Add IMU-based orientation and motion (next 10 values)
        roll, pitch, yaw = 0.01, -0.02, 0.05  # Small angles
        angular_vel = [0.1, -0.05, 0.02]
        linear_acc = [0.2, 0.1, 9.81]  # Gravity component
        fused_data.extend([roll, pitch, yaw])
        fused_data.extend(angular_vel)
        fused_data.extend(linear_acc[2:])  # Just z component of acceleration

        # Add force-based contact information (last 5 values)
        contact_force = 1.5
        is_contact = 0.0  # No contact initially
        fused_data.extend([contact_force, is_contact, 0.1, 0.2, 0.3])

        # Publish fused perception data
        perception_msg = Float32MultiArray()
        perception_msg.data = fused_data[:50]  # Limit to 50 values
        self.perception_pub.publish(perception_msg)

        # Create and publish state information
        state_msg = Float32MultiArray()
        state_data = [
            self.get_clock().now().nanoseconds,  # Timestamp
            1.0, 2.0, 0.0,  # Position (x, y, z)
            roll, pitch, yaw,  # Orientation
            0.1, -0.05, 0.02,  # Velocity
            contact_force, is_contact  # Contact state
        ]
        state_msg.data = state_data
        self.state_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)

    # Create nodes for the sensor system
    camera_node = CameraSensorNode()
    lidar_node = LidarSensorNode()
    imu_node = IMUSensorNode()
    force_node = ForceSensorNode()
    fusion_node = SensorFusionNode()

    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(camera_node)
    executor.add_node(lidar_node)
    executor.add_node(imu_node)
    executor.add_node(force_node)
    executor.add_node(fusion_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup nodes
        camera_node.destroy_node()
        lidar_node.destroy_node()
        imu_node.destroy_node()
        force_node.destroy_node()
        fusion_node.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Robot sensor systems are fundamental to Physical AI, providing the essential data that enables robots to perceive and interact with their environment. The four primary sensor types - cameras, LiDAR, IMU, and force sensors - each provide unique and complementary information:

- **Cameras** provide rich visual information including color, texture, and shape recognition
- **LiDAR** provides precise 3D distance measurements and enables accurate mapping
- **IMU** provides orientation, acceleration, and angular velocity for balance and motion
- **Force sensors** provide tactile feedback for interaction and manipulation

The key to effective robot perception lies in sensor fusion - combining data from multiple sensors to create a comprehensive understanding of the environment and the robot's state. Each sensor type has strengths and limitations, but together they provide robust and reliable perception capabilities essential for safe and effective robot operation.

Modern robot systems integrate these sensors into sophisticated perception pipelines that can recognize objects, detect obstacles, maintain balance, and perform precise manipulation tasks. The quality and reliability of these sensor systems directly impact the robot's ability to operate in real-world environments.

## Exercises

1. **Basic Understanding**: Compare the advantages and limitations of cameras versus LiDAR for environment perception. When would you choose one over the other?

2. **Application Exercise**: Design a sensor fusion algorithm that combines camera and LiDAR data to improve object detection. How would you handle the different data formats and coordinate systems?

3. **Implementation Exercise**: Modify the camera processing code to implement a simple color-based object tracker that follows a specific colored object through the camera's field of view.

4. **Challenge Exercise**: Create a safety system that uses IMU and force sensor data to detect when a robot is about to fall or experiencing dangerous forces, and implement appropriate emergency responses.