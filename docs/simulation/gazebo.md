# Robot Simulation with Gazebo: Physics, Sensors, and Environments

## Concept

Gazebo is a powerful 3D simulation environment that provides realistic physics simulation, high-fidelity graphics, and support for various robot sensors. Think of it as a virtual laboratory where you can test and develop robot behaviors without the risks and costs associated with real-world experimentation.

In robotics, simulation is crucial because it allows developers to:
- Test algorithms in a safe, repeatable environment
- Debug complex behaviors without physical robot damage
- Evaluate robot performance across various scenarios
- Train machine learning models before real-world deployment
- Prototype new robot designs and configurations

Gazebo matters in Physical AI because it bridges the gap between pure simulation and reality. Unlike simple 2D simulators, Gazebo provides realistic physics simulation with properties like friction, gravity, and collision detection that closely match real-world behavior. This is especially important for humanoid robots that need to maintain balance, manipulate objects, and navigate complex environments.

If you're familiar with game engines like Unity or Unreal Engine, Gazebo provides similar capabilities but specifically designed for robotics applications. It includes realistic physics simulation, sensor models, and tools specifically designed for robot development and testing.

## ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAZEBO SIMULATION ENVIRONMENT                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐      PHYSICS ENGINE      ┌─────────────────┐   │
│  │   ROBOT     │ ────────────────────────▶│   ODE/BULLET    │   │
│  │  MODEL      │                          │   SIMULATION    │   │
│  │  (URDF)     │ ◀─────────────────────── │   CORE          │   │
│  └─────────────┘      JOINT FORCES       └─────────────────┘   │
│         │                                           │           │
│         ▼                                           ▼           │
│  ┌─────────────┐                              ┌─────────────┐   │
│  │  SENSORS    │                              │  COLLISION  │   │
│  │  (Cameras,  │                              │  DETECTION  │   │
│  │  LiDAR,     │                              │             │   │
│  │  IMU, etc.) │                              └─────────────┘   │
│  └─────────────┘                                    │             │
│         │                                           │             │
│         ▼                                           ▼             │
│  ┌─────────────┐    DATA FLOW        ┌─────────────────────────┐ │
│  │  ROS 2      │ ───────────────────▶│    SIMULATION         │ │
│  │  INTERFACE  │                     │    WORLD              │ │
│  │             │ ◀────────────────── │    (Environment,      │ │
│  └─────────────┘   CONTROL COMMANDS  │     Objects, etc.)    │ │
│                                      └─────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                GAZEBO PLUGIN ARCHITECTURE                       │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  SENSOR         │    │  PHYSICS        │    │  RENDERING  │ │
│  │  PLUGINS        │    │  PLUGINS        │    │  PLUGINS    │ │
│  │                 │    │                 │    │             │ │
│  │ • Camera        │    │ • Joint Control │    │ • OpenGL    │ │
│  │ • LiDAR         │    │ • Collision     │    │ • Lighting  │ │
│  │ • IMU           │    │ • Dynamics      │    │ • Shadows   │ │
│  │ • Force/Torque  │    │ • Contacts      │    │ • Textures  │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│         │                       │                       │       │
│         └───────────────────────┼───────────────────────┘       │
│                                 │                               │
│                    ┌─────────────────────────────────────────┐  │
│                    │        GAZEBO SERVER                  │  │
│                    │                                       │  │
│                    │  ┌──────────────────────────────────┐ │  │
│                    │  │        SIMULATION LOOP           │ │  │
│                    │  │                                  │ │  │
│                    │  │  1. Update Physics               │ │  │
│                    │  │  2. Process Sensor Data          │ │  │
│                    │  │  3. Handle ROS 2 Communication   │ │  │
│                    │  │  4. Update Graphics              │ │  │
│                    │  └──────────────────────────────────┘ │  │
│                    └─────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This diagram illustrates the Gazebo simulation environment with its core components: the physics engine, sensor simulation, collision detection, and ROS 2 interface, all working together through the simulation loop.

## Real-world Analogy

Think of Gazebo like a flight simulator for pilots, but designed specifically for robots. Just as pilots use flight simulators to practice flying in various conditions (storms, equipment failures, different airports) without the risks of real flight, roboticists use Gazebo to test robot behaviors in various environments and conditions without risking physical robots.

A flight simulator needs to accurately model:
- Aerodynamics and physics of flight
- Weather conditions and environmental factors
- Aircraft controls and responses
- Visual representation of the world

Similarly, Gazebo models:
- Physics of robot movement and interaction
- Environmental factors like friction and gravity
- Sensor responses and limitations
- Visual and geometric representation of the world

Just as flight simulators allow pilots to practice thousands of scenarios safely, Gazebo allows roboticists to test robot behaviors across countless scenarios, from normal operation to edge cases, all in a safe virtual environment.

## Pseudo-code (ROS 2 / Python style)

```xml
<!-- Example Gazebo world file with physics properties and models -->
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="robotics_lab">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.0</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.4 0.2 -1.0</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Example robot model with Gazebo plugins -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

    <!-- Example objects in the environment -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.5</mu>
                <mu2>0.5</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>20.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Example obstacle -->
    <model name="obstacle">
      <pose>-1 1 0.2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>0.4</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>0.4</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>1.0 0.0 0.0 1</ambient>
            <diffuse>1.0 0.0 0.0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.1</iyy>
            <iyz>0.0</iyz>
            <izz>0.08</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

```python
# Gazebo plugin example - Camera sensor plugin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2

class GazeboCameraPlugin(Node):
    """Simulated camera plugin for Gazebo"""
    def __init__(self):
        super().__init__('gazebo_camera_plugin')

        # Create publisher for camera images
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Camera parameters (simulated)
        self.camera_info = CameraInfo()
        self.camera_info.width = 640
        self.camera_info.height = 480
        self.camera_info.k = [640.0, 0.0, 320.0, 0.0, 640.0, 240.0, 0.0, 0.0, 1.0]  # Camera matrix
        self.camera_info.distortion_model = 'plumb_bob'
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients

        # Timer to simulate camera capture rate
        self.timer = self.create_timer(0.1, self.capture_image)

        # Simulated environment state
        self.simulation_time = 0.0

    def capture_image(self):
        """Simulate capturing an image from the virtual camera"""
        # In a real Gazebo plugin, this would receive image data from Gazebo
        # For simulation, we'll create a synthetic image

        # Create a synthetic image with some geometric shapes
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some colored shapes to simulate environment
        cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)  # Green circle
        cv2.line(image, (0, 400), (640, 400), (255, 255, 255), 2)  # White line (floor)

        # Add some noise to make it more realistic
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Convert to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera_link'

        # Publish image and camera info
        self.image_pub.publish(ros_image)
        self.camera_info.header.stamp = ros_image.header.stamp
        self.info_pub.publish(self.camera_info)

        self.simulation_time += 0.1

class GazeboLidarPlugin(Node):
    """Simulated LiDAR plugin for Gazebo"""
    def __init__(self):
        super().__init__('gazebo_lidar_plugin')

        # Create publisher for LiDAR data
        from sensor_msgs.msg import LaserScan
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # LiDAR parameters
        self.angle_min = -np.pi / 2
        self.angle_max = np.pi / 2
        self.angle_increment = np.pi / 180  # 1 degree
        self.scan_time = 0.1
        self.range_min = 0.1
        self.range_max = 10.0

        # Timer for LiDAR updates
        self.timer = self.create_timer(0.05, self.publish_scan)

    def publish_scan(self):
        """Simulate LiDAR scan data"""
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'laser_link'

        # Set scan parameters
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.time_increment = 0.0
        scan.scan_time = self.scan_time
        scan.range_min = self.range_min
        scan.range_max = self.range_max

        # Calculate number of ranges
        num_ranges = int((self.angle_max - self.angle_min) / self.angle_increment) + 1
        scan.ranges = [float('inf')] * num_ranges

        # Simulate some obstacles in the environment
        for i in range(num_ranges):
            angle = self.angle_min + i * self.angle_increment

            # Simulate a wall at 3 meters in front
            if -0.2 < angle < 0.2:
                scan.ranges[i] = 3.0 + np.random.normal(0, 0.05)  # Add some noise

            # Simulate an obstacle to the right
            elif 0.3 < angle < 0.5:
                scan.ranges[i] = 2.0 + np.random.normal(0, 0.03)

            # Simulate an obstacle to the left
            elif -0.5 < angle < -0.3:
                scan.ranges[i] = 2.5 + np.random.normal(0, 0.04)

        # Publish the scan
        self.scan_pub.publish(scan)

class GazeboIMUPlugin(Node):
    """Simulated IMU plugin for Gazebo"""
    def __init__(self):
        super().__init__('gazebo_imu_plugin')

        # Create publisher for IMU data
        from sensor_msgs.msg import Imu
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # Timer for IMU updates
        self.timer = self.create_timer(0.01, self.publish_imu)

        # Simulated robot state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.linear_acceleration = [0.0, 0.0, 9.81]  # Gravity

    def publish_imu(self):
        """Simulate IMU data"""
        from geometry_msgs.msg import Vector3, Quaternion
        import math

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Simulate small oscillations (e.g., from walking)
        self.roll = 0.01 * math.sin(self.get_clock().now().nanoseconds * 1e-9 * 2)
        self.pitch = 0.015 * math.cos(self.get_clock().now().nanoseconds * 1e-9 * 1.5)

        # Convert Euler angles to quaternion
        cy = math.cos(self.yaw * 0.5)
        sy = math.sin(self.yaw * 0.5)
        cp = math.cos(self.pitch * 0.5)
        sp = math.sin(self.pitch * 0.5)
        cr = math.cos(self.roll * 0.5)
        sr = math.sin(self.roll * 0.5)

        imu_msg.orientation.w = cr * cp * cy + sr * sp * sy
        imu_msg.orientation.x = sr * cp * cy - cr * sp * sy
        imu_msg.orientation.y = cr * sp * cy + sr * cp * sy
        imu_msg.orientation.z = cr * cp * sy - sr * sp * cy

        # Simulate angular velocity (small random movements)
        self.angular_velocity[0] = 0.1 * math.cos(self.get_clock().now().nanoseconds * 1e-9 * 3)
        self.angular_velocity[1] = 0.08 * math.sin(self.get_clock().now().nanoseconds * 1e-9 * 2.5)
        self.angular_velocity[2] = 0.05 * math.sin(self.get_clock().now().nanoseconds * 1e-9 * 4)

        imu_msg.angular_velocity.x = self.angular_velocity[0] + np.random.normal(0, 0.001)
        imu_msg.angular_velocity.y = self.angular_velocity[1] + np.random.normal(0, 0.001)
        imu_msg.angular_velocity.z = self.angular_velocity[2] + np.random.normal(0, 0.001)

        # Simulate linear acceleration (with gravity and small movements)
        self.linear_acceleration[0] = 0.2 * math.sin(self.get_clock().now().nanoseconds * 1e-9 * 5)
        self.linear_acceleration[1] = 0.15 * math.cos(self.get_clock().now().nanoseconds * 1e-9 * 4)

        imu_msg.linear_acceleration.x = self.linear_acceleration[0] + np.random.normal(0, 0.01)
        imu_msg.linear_acceleration.y = self.linear_acceleration[1] + np.random.normal(0, 0.01)
        imu_msg.linear_acceleration.z = self.linear_acceleration[2] + np.random.normal(0, 0.01)

        # Covariance matrices (set to realistic values)
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        imu_msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        imu_msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]

        self.imu_pub.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)

    # Create simulation nodes
    camera_plugin = GazeboCameraPlugin()
    lidar_plugin = GazeboLidarPlugin()
    imu_plugin = GazeboIMUPlugin()

    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(camera_plugin)
    executor.add_node(lidar_plugin)
    executor.add_node(imu_plugin)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        camera_plugin.destroy_node()
        lidar_plugin.destroy_node()
        imu_plugin.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# Gazebo simulation control and interaction example
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState, GetEntityState, SpawnEntity
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
import math

class GazeboSimulationController(Node):
    """Controller for Gazebo simulation environment"""
    def __init__(self):
        super().__init__('gazebo_simulation_controller')

        # Create clients for Gazebo services
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        self.pause_simulation_client = self.create_client(Empty, '/pause_physics')
        self.unpause_simulation_client = self.create_client(Empty, '/unpause_physics')
        self.set_entity_state_client = self.create_client(SetEntityState, '/set_entity_state')
        self.get_entity_state_client = self.create_client(GetEntityState, '/get_entity_state')
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')

        # Wait for services to be available
        self.get_logger().info('Waiting for Gazebo services...')
        self.reset_simulation_client.wait_for_service()
        self.pause_simulation_client.wait_for_service()
        self.unpause_simulation_client.wait_for_service()
        self.set_entity_state_client.wait_for_service()
        self.get_entity_state_client.wait_for_service()
        self.spawn_entity_client.wait_for_service()
        self.get_logger().info('Gazebo services ready!')

        # Timer to periodically check robot state
        self.timer = self.create_timer(1.0, self.check_robot_state)

    def reset_simulation(self):
        """Reset the entire simulation"""
        request = Empty.Request()
        future = self.reset_simulation_client.call_async(request)
        return future

    def pause_simulation(self):
        """Pause physics simulation"""
        request = Empty.Request()
        future = self.pause_simulation_client.call_async(request)
        return future

    def unpause_simulation(self):
        """Resume physics simulation"""
        request = Empty.Request()
        future = self.unpause_simulation_client.call_async(request)
        return future

    def move_robot_to_pose(self, entity_name, x, y, z, roll=0, pitch=0, yaw=0):
        """Move a robot/entity to a specific pose"""
        request = SetEntityState.Request()

        # Set entity name
        request.state.name = entity_name

        # Set position
        request.state.pose.position.x = x
        request.state.pose.position.y = y
        request.state.pose.position.z = z

        # Convert Euler to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x_q = sr * cp * cy - cr * sp * sy
        y_q = cr * sp * cy + sr * cp * sy
        z_q = cr * cp * sy - sr * sp * cy

        request.state.pose.orientation.w = w
        request.state.pose.orientation.x = x_q
        request.state.pose.orientation.y = y_q
        request.state.pose.orientation.z = z_q

        # Set zero velocities
        request.state.twist.linear.x = 0.0
        request.state.twist.linear.y = 0.0
        request.state.twist.linear.z = 0.0
        request.state.twist.angular.x = 0.0
        request.state.twist.angular.y = 0.0
        request.state.twist.angular.z = 0.0

        future = self.set_entity_state_client.call_async(request)
        return future

    def get_robot_state(self, entity_name, reference_frame=''):
        """Get the current state of a robot/entity"""
        request = GetEntityState.Request()
        request.name = entity_name
        request.reference_frame = reference_frame

        future = self.get_entity_state_client.call_async(request)
        return future

    def spawn_object(self, name, xml, initial_pose, reference_frame=''):
        """Spawn a new object in the simulation"""
        request = SpawnEntity.Request()
        request.name = name
        request.xml = xml
        request.initial_pose = initial_pose
        request.reference_frame = reference_frame

        future = self.spawn_entity_client.call_async(request)
        return future

    def check_robot_state(self):
        """Periodically check robot state"""
        future = self.get_robot_state('simple_humanoid')
        # Note: In a real implementation, you would handle the future response
        # For this example, we'll just log that we're checking
        self.get_logger().info('Checking robot state...')

class SimulationScenarioManager(Node):
    """Manager for running different simulation scenarios"""
    def __init__(self):
        super().__init__('simulation_scenario_manager')

        # Create controller
        self.controller = GazeboSimulationController()

        # Timer to run scenarios
        self.scenario_timer = self.create_timer(5.0, self.run_next_scenario)
        self.scenario_count = 0

    def run_next_scenario(self):
        """Run the next simulation scenario"""
        scenarios = [
            self.run_balancing_scenario,
            self.run_navigation_scenario,
            self.run_manipulation_scenario,
            self.run_obstacle_avoidance_scenario
        ]

        if self.scenario_count < len(scenarios):
            scenario_func = scenarios[self.scenario_count]
            self.get_logger().info(f'Running scenario {self.scenario_count + 1}: {scenario_func.__name__}')
            scenario_func()
            self.scenario_count += 1
        else:
            self.scenario_count = 0  # Reset to first scenario

    def run_balancing_scenario(self):
        """Scenario: Test robot balancing"""
        self.get_logger().info('Running balancing scenario...')
        # Move robot to position where balancing is challenged
        self.controller.move_robot_to_pose('simple_humanoid', 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    def run_navigation_scenario(self):
        """Scenario: Test robot navigation"""
        self.get_logger().info('Running navigation scenario...')
        # Move robot to starting position for navigation
        self.controller.move_robot_to_pose('simple_humanoid', -2.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    def run_manipulation_scenario(self):
        """Scenario: Test robot manipulation"""
        self.get_logger().info('Running manipulation scenario...')
        # Move robot near an object for manipulation
        self.controller.move_robot_to_pose('simple_humanoid', 1.5, 0.0, 1.0, 0.0, 0.0, 1.57)

    def run_obstacle_avoidance_scenario(self):
        """Scenario: Test obstacle avoidance"""
        self.get_logger().info('Running obstacle avoidance scenario...')
        # Move robot to position with obstacles
        self.controller.move_robot_to_pose('simple_humanoid', 0.0, -2.0, 1.0, 0.0, 0.0, 0.0)

def main(args=None):
    rclpy.init(args=args)

    scenario_manager = SimulationScenarioManager()

    try:
        rclpy.spin(scenario_manager)
    except KeyboardInterrupt:
        pass
    finally:
        scenario_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Gazebo is a comprehensive 3D simulation environment that provides realistic physics simulation, sensor modeling, and environment creation capabilities essential for robotics development. It enables safe, repeatable testing of robot behaviors before deployment to real hardware.

The key components of Gazebo simulation include:
- **Physics Engine**: Provides realistic simulation of forces, collisions, and dynamics
- **Sensor Simulation**: Models cameras, LiDAR, IMU, and other sensors with realistic noise and limitations
- **Environment Modeling**: Creates complex 3D worlds with objects, lighting, and physics properties
- **ROS 2 Integration**: Seamless communication between simulated robots and ROS 2 nodes

Gazebo is particularly valuable for Physical AI and humanoid robotics because it allows for testing complex behaviors like walking, manipulation, and navigation in a safe virtual environment. The realistic physics simulation ensures that behaviors developed in Gazebo have a high likelihood of working on real robots.

Understanding Gazebo simulation is crucial for developing robust robot behaviors, as it provides the bridge between algorithm development and real-world deployment.

## Exercises

1. **Basic Understanding**: Explain the difference between a Gazebo world file and a URDF robot model. How do they work together in a simulation?

2. **Application Exercise**: Design a Gazebo world file for a humanoid robot to practice walking. Include a flat ground plane, some obstacles to navigate around, and a target location. Describe the physics properties you would set for realistic walking simulation.

3. **Implementation Exercise**: Create a Python node that interfaces with Gazebo to spawn a new object in the simulation at a random location when a service is called. The object should be a simple geometric shape (cube, sphere, or cylinder).

4. **Challenge Exercise**: Implement a simulation scenario where a humanoid robot must navigate through a changing environment (moving obstacles). Create a controller that adjusts the robot's behavior based on sensor feedback from the simulated environment.