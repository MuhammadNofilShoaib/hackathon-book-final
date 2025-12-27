# Navigation and Path Planning with Nav2 for Humanoid Robots (Advanced Level)

## Concept

Navigation2 (Nav2) is ROS 2's state-of-the-art navigation framework that provides advanced path planning, obstacle avoidance, and navigation capabilities for mobile robots. Think of it as a comprehensive navigation system that combines perception, planning, and control to enable robots to move safely and efficiently through complex environments.

In humanoid robotics, navigation becomes particularly challenging due to the robot's complex kinematic structure, balance requirements, and human-like form factor. Unlike wheeled robots that can simply avoid obstacles, humanoid robots must consider their bipedal locomotion, center of mass, and the need to maintain balance while navigating. Nav2 provides the flexibility to adapt navigation algorithms to these unique humanoid requirements.

Nav2 matters in Physical AI because it bridges the gap between basic path planning and sophisticated, real-world navigation. It incorporates advanced features like dynamic obstacle avoidance, recovery behaviors, and multi-layered costmaps that are essential for safe robot operation in human environments. For humanoid robots, Nav2 can be customized to account for their specific locomotion patterns and balance constraints.

If you're familiar with GPS navigation systems in cars, Nav2 provides similar route planning and obstacle avoidance capabilities but specifically designed for robots. Instead of following roads, robots using Nav2 can navigate through complex indoor environments, around furniture, and through doorways while considering their unique kinematic constraints.



> **Best Practice**: For production systems, consider [advanced technique] to optimize performance.

> **Performance Note**: This approach has O(n) complexity and may require optimization for large-scale applications.

## ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NAV2 ARCHITECTURE FOR HUMANOID ROBOTS                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    NAV2 PLUGINS    ┌─────────────────────────┐   │
│  │   NAV2          │ ──────────────────▶│    NAV2 PLUGINS         │   │
│  │   CORE          │                     │    (Customized for      │   │
│  │   (Server)      │ ◀───────────────── │     Humanoid Robots)    │   │
│  │                 │   CONFIGURATION    │                         │   │
│  │  ┌──────────┐   │                    │  ┌──────────────────┐   │   │
│  │  │Global    │   │                    │  │ Global Planner   │   │   │
│  │  │Planner   │───┼─────────────────────────│ (A*, Dijkstra,    │   │   │
│  │  │          │   │                    │  │  Humanoid-aware) │   │   │
│  │  └──────────┘   │                    │  ├──────────────────┤   │   │
│  │                 │                    │  │ Local Planner    │   │   │
│  │  ┌──────────┐   │                    │  │ (Teb, DWA,       │   │   │
│  │  │Local     │   │                    │  │  Humanoid-aware) │   │   │
│  │  │Planner   │───┼─────────────────────────│                  │   │   │
│  │  │          │   │                    │  ├──────────────────┤   │   │
│  │  └──────────┘   │                    │  │ Costmap Layers   │   │   │
│  │                 │                    │  │ (Static,         │   │   │
│  │  ┌──────────┐   │                    │  │  Obstacle,       │   │   │
│  │  │Recovery  │   │                    │  │  Inflation,      │   │   │
│  │  │Behaviors │───┼─────────────────────────│  Humanoid-aware) │   │   │
│  │  │          │   │                    │  └──────────────────┘   │   │
│  └─────────────────┘                    └─────────────────────────┘   │
│         │                                           │                   │
│         ▼                                           ▼                   │
│  ┌─────────────────┐                      ┌─────────────────────────┐   │
│  │  PERCEPTION     │                      │    HUMANOID LOCOMOTION  │   │
│  │  (Sensors)      │─────────────────────▶│    CONTROLLER           │   │
│  │                 │    SENSOR DATA       │                         │   │
│  │  • LiDAR        │                      │  ┌──────────────────┐   │   │
│  │  • Cameras      │                      │  │ Balance Control  │   │   │
│  │  • IMU          │                      │  │ (ZMP, Capture     │   │   │
│  │  • Odometry     │                      │  │  Point, etc.)    │   │   │
│  └─────────────────┘                      │  ├──────────────────┤   │   │
│                                           │  │ Step Planning    │   │   │
│                                           │  │ (Footstep       │   │   │
│                                           │  │  Generation)     │   │   │
│                                           │  ├──────────────────┤   │   │
│                                           │  │ Gait Control     │   │   │
│                                           │  │ (Walking,        │   │   │
│                                           │  │  Balancing)      │   │   │
│                                           │  └──────────────────┘   │   │
│                                           └─────────────────────────┘   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                NAV2 NAVIGATION WORKFLOW                                  │
│                                                                         │
│  ┌─────────────┐    1. GOAL SETTING    ┌─────────────────────────────┐ │
│  │   USER      │ ────────────────────▶  │    NAV2 SERVER              │ │
│  │   REQUEST   │                        │                             │ │
│  └─────────────┘                        │  ┌─────────────────────────┐│ │
│         │                                │  │  Global Costmap       ││ │
│         │                                │  │  (Static obstacles,   ││ │
│         │                                │  │   map-based)          ││ │
│         │                                │  └─────────────────────────┘│ │
│         │                                │         │                    │ │
│         │                                │         ▼                    │ │
│         │    2. PATH PLANNING           │  ┌─────────────────────────┐│ │
│         └───────────────────────────────────│  Global Planner         ││ │
│                                          │  │  (A* with humanoid     ││ │
│                                          │  │   constraints)         ││ │
│                                          │  └─────────────────────────┘│ │
│                                                 │                      │ │
│                                                 ▼                      │ │
│  ┌─────────────────┐    3. LOCAL PLANNING   ┌─────────────────────────┐│ │
│  │  CURRENT ROBOT  │ ────────────────────▶  │    Local Planner        ││ │
│  │  STATE &        │                        │  (Teb/DWA with         ││ │
│  │  SENSOR DATA   │                        │   humanoid dynamics)    ││ │
│  └─────────────────┘                        └─────────────────────────┘│ │
│         │                                               │                │ │
│         │                                               ▼                │ │
│         │                                       ┌─────────────────────────┤ │
│         │                                       │    HUMANOID LOCOMOTION  │ │
│         │                                       │    CONTROLLER           │ │
│         │                                       │                         │ │
│         │                                       │  ┌──────────────────┐   │ │
│         │    4. EXECUTION                     │  │  Step Generator  │   │ │
│         └───────────────────────────────────────┼──│  (Footstep       │   │ │
│                                                 │  │   Planning)      │   │ │
│                                                 │  ├──────────────────┤   │ │
│                                                 │  │  Balance Control │   │ │
│                                                 │  │  (ZMP Control,   │   │ │
│                                                 │  │   COM tracking)  │   │ │
│                                                 │  └──────────────────┘   │ │
│                                                 └─────────────────────────┘ │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                HUMANOID-SPECIFIC NAVIGATION CONSTRAINTS                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BALANCE CONSTRAINTS                                            │   │
│  │  • Center of Mass (COM) limits                                  │   │
│  │  • Zero Moment Point (ZMP) constraints                          │   │
│  │  • Capture Point stability                                      │   │
│  │  • Step timing and placement                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  KINEMATIC CONSTRAINTS                                          │   │
│  │  • Joint angle limits                                           │   │
│  │  • Reachable workspace                                          │   │
│  │  • Gait patterns (walking, stepping)                            │   │
│  │  • Upper body movement during locomotion                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ENVIRONMENTAL CONSTRAINTS                                      │   │
│  │  • Doorway navigation (lateral movement)                        │   │
│  │  • Stair climbing (if capable)                                  │   │
│  │  • Narrow passages (shoulder width)                             │   │
│  │  • Slope limitations                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This diagram illustrates the Nav2 architecture for humanoid robots, showing how the navigation system integrates with humanoid-specific locomotion controllers and considers unique balance and kinematic constraints.

## Real-world Analogy

Think of Nav2 for humanoid robots like a personal assistant for a person walking through a busy shopping mall. Just as a person must navigate around other shoppers, avoid obstacles, maintain their balance, and choose the most efficient path to their destination, a humanoid robot using Nav2 must do the same but with additional complexity.

A person walking through a mall needs to:
- Plan a route from their current location to their destination
- Adjust their path in real-time to avoid other people and obstacles
- Maintain their balance while walking and changing direction
- Consider their physical limitations (step size, turning radius, etc.)

Similarly, Nav2 for humanoid robots:
- Plans global routes considering the robot's unique kinematic structure
- Adjusts local paths in real-time based on sensor data
- Maintains balance through integration with locomotion controllers
- Accounts for humanoid-specific constraints like step size and balance

Just as a personal assistant would consider a person's walking speed, physical capabilities, and preferred route when providing navigation guidance, Nav2 customizes navigation for the specific capabilities and constraints of humanoid robots, making it safe and effective for human environments.

## Pseudo-code (Nav2 / ROS 2 style)
# Advanced Implementation:
# - Real-time performance considerations
# - Memory management optimizations
# - Parallel processing opportunities
# - Safety and fault-tolerance measures
# - Hardware-specific optimizations



```python
# Nav2 configuration and server setup for humanoid robots
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan, Imu, JointState
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException
import numpy as np
import math
from typing import List, Tuple, Optional
import threading
import time

class HumanoidNav2Server(Node):
    """Custom Nav2 server for humanoid robot navigation"""

    def __init__(self):
        super().__init__('humanoid_nav2_server')

        # Create action server for navigation
        self.nav_to_pose_action_server = self.create_action_server(
            NavigateToPose,
            'navigate_to_pose',
            self.execute_nav_to_pose_callback,
            result_timeout=300  # 5 minutes timeout
        )

        # Subscribers for sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Publishers for navigation commands
        self.cmd_vel_pub = self.create_publisher(
            String, '/humanoid/navigation_commands', 10)
        self.path_pub = self.create_publisher(
            Path, '/humanoid/local_plan', 10)
        self.global_path_pub = self.create_publisher(
            Path, '/humanoid/global_plan', 10)

        # TF buffer for transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Navigation state
        self.current_pose = None
        self.current_imu = None
        self.current_joints = None
        self.navigation_active = False
        self.cancel_requested = False

        # Humanoid-specific parameters
        self.step_size = 0.3  # meters
        self.turn_radius = 0.5  # meters
        self.max_walk_speed = 0.5  # m/s
        self.balance_threshold = 0.1  # IMU angle threshold for balance
        self.step_height = 0.1  # meters for obstacle clearance

        # Costmap for humanoid navigation
        self.humanoid_costmap = HumanoidCostmap()

        self.get_logger().info('Humanoid Nav2 Server initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        if self.navigation_active:
            # Update costmap with laser data
            self.humanoid_costmap.update_with_laser(msg)

    def imu_callback(self, msg):
        """Process IMU data for balance"""
        self.current_imu = msg

        # Check balance if navigating
        if self.navigation_active:
            roll, pitch, _ = self.quaternion_to_euler([
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w
            ])

            # Check if robot is within balance limits
            if abs(roll) > self.balance_threshold or abs(pitch) > self.balance_threshold:
                self.get_logger().warn('Robot balance compromised during navigation')

    def joint_callback(self, msg):
        """Process joint state data"""
        self.current_joints = msg

    def execute_nav_to_pose_callback(self, goal_handle):
        """Execute navigation to pose goal"""
        self.get_logger().info('Received navigation goal')
        goal = goal_handle.request.pose

        # Validate goal for humanoid constraints
        if not self.is_valid_humanoid_goal(goal):
            self.get_logger().error('Invalid humanoid navigation goal')
            goal_handle.succeed()
            result = NavigateToPose.Result()
            result.result = 0  # Failure
            return result

        self.navigation_active = True
        self.cancel_requested = False

        try:
            # Plan global path
            global_path = self.plan_global_path(goal)
            if not global_path:
                self.get_logger().error('Failed to plan global path')
                goal_handle.succeed()
                result = NavigateToPose.Result()
                result.result = 0
                return result

            # Publish global path
            self.global_path_pub.publish(global_path)

            # Execute navigation
            success = self.execute_navigation(goal, global_path)

            if success:
                goal_handle.succeed()
                result = NavigateToPose.Result()
                result.result = 1  # Success
            else:
                goal_handle.succeed()
                result = NavigateToPose.Result()
                result.result = 0  # Failure

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            goal_handle.succeed()
            result = NavigateToPose.Result()
            result.result = 0

        finally:
            self.navigation_active = False

        return result

    def is_valid_humanoid_goal(self, goal):
        """Check if goal is valid for humanoid robot"""
        # Get current pose
        current_pose = self.get_current_pose()
        if current_pose is None:
            return False

        # Check distance (humanoid robots typically have limited range per session)
        distance = self.calculate_distance(current_pose, goal.pose)
        if distance > 100.0:  # 100 meters max for safety
            self.get_logger().warn(f'Goal too far: {distance:.2f}m')
            return False

        # Check if goal is in navigable space considering humanoid dimensions
        goal_point = (goal.pose.position.x, goal.pose.position.y)
        if not self.humanoid_costmap.is_navigable(goal_point):
            self.get_logger().warn('Goal location not navigable for humanoid')
            return False

        return True

    def plan_global_path(self, goal):
        """Plan global path considering humanoid constraints"""
        # Get current pose
        current_pose = self.get_current_pose()
        if current_pose is None:
            return None

        # In a real implementation, this would use a path planning algorithm
        # like A* or Dijkstra with humanoid-specific cost functions
        # For this example, we'll simulate the path planning

        # Get path from costmap considering humanoid dimensions
        path = self.humanoid_costmap.plan_path(
            (current_pose.pose.position.x, current_pose.pose.position.y),
            (goal.pose.position.x, goal.pose.position.y)
        )

        if path:
            # Convert to ROS Path message
            ros_path = Path()
            ros_path.header.stamp = self.get_clock().now().to_msg()
            ros_path.header.frame_id = "map"

            for point in path:
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = ros_path.header.stamp
                pose_stamped.header.frame_id = "map"
                pose_stamped.pose.position.x = point[0]
                pose_stamped.pose.position.y = point[1]
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation.w = 1.0

                ros_path.poses.append(pose_stamped)

            return ros_path

        return None

    def execute_navigation(self, goal, global_path):
        """Execute navigation with humanoid-specific locomotion"""
        self.get_logger().info('Starting humanoid navigation execution')

        # Follow the global path with local obstacle avoidance
        for i, path_pose in enumerate(global_path.poses):
            if self.cancel_requested:
                self.get_logger().info('Navigation cancelled')
                return False

            # Get current robot pose
            current_pose = self.get_current_pose()
            if current_pose is None:
                continue

            # Calculate distance to current path pose
            distance = self.calculate_distance(current_pose, path_pose)

            # If close enough to current waypoint, move to next
            if distance < 0.5:  # 0.5m threshold
                continue

            # Plan local path with obstacle avoidance
            local_path = self.plan_local_path(current_pose, path_pose)
            if local_path:
                # Execute humanoid locomotion along local path
                success = self.execute_humanoid_locomotion(local_path)
                if not success:
                    self.get_logger().error('Failed to execute humanoid locomotion')
                    return False
            else:
                self.get_logger().warn('Local path planning failed, stopping')
                return False

            # Small delay to allow for movement
            time.sleep(0.1)

        # Final check if we reached the goal
        current_pose = self.get_current_pose()
        if current_pose:
            final_distance = self.calculate_distance(current_pose, goal.pose)
            return final_distance < 1.0  # 1m tolerance for humanoid

        return False

    def plan_local_path(self, current_pose, target_pose):
        """Plan local path with obstacle avoidance"""
        # In a real implementation, this would use local planners like DWA or Teb
        # For humanoid robots, considering balance and step constraints
        # For this example, we'll return a simple path

        current_pos = (current_pose.pose.position.x, current_pose.pose.position.y)
        target_pos = (target_pose.pose.position.x, target_pose.pose.position.y)

        # Check for obstacles along the direct path
        if self.humanoid_costmap.has_obstacles_along_path(current_pos, target_pos):
            # Plan around obstacles using costmap
            local_path = self.humanoid_costmap.plan_local_path(current_pos, target_pos)
        else:
            # Direct path is clear
            local_path = [current_pos, target_pos]

        return local_path

    def execute_humanoid_locomotion(self, path):
        """Execute humanoid-specific locomotion along path"""
        # This would interface with the humanoid's walking controller
        # For this example, we'll simulate the locomotion

        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]

            # Calculate direction and distance
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            distance = math.sqrt(dx*dx + dy*dy)

            if distance == 0:
                continue

            # Calculate orientation for movement
            yaw = math.atan2(dy, dx)

            # Generate footstep plan for this segment
            footsteps = self.generate_footsteps(start_point, end_point, yaw)

            # Execute footsteps (in real implementation, send to walking controller)
            for footstep in footsteps:
                if self.cancel_requested:
                    return False

                # In real implementation, this would send commands to the robot's
                # walking controller to execute the footstep
                self.execute_footstep(footstep)

                # Wait for step completion
                time.sleep(0.5)

        return True

    def generate_footsteps(self, start, end, orientation):
        """Generate footstep plan for humanoid locomotion"""
        footsteps = []

        # Calculate the path vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        total_distance = math.sqrt(dx*dx + dy*dy)

        # Generate footsteps at regular intervals based on step size
        num_steps = max(1, int(total_distance / self.step_size))
        step_dx = dx / num_steps
        step_dy = dy / num_steps

        for i in range(num_steps + 1):
            x = start[0] + i * step_dx
            y = start[1] + i * step_dy

            # Alternate between left and right foot
            foot_type = "left" if i % 2 == 0 else "right"

            footstep = {
                'position': (x, y, 0.0),
                'orientation': orientation,
                'foot_type': foot_type,
                'step_height': self.step_height
            }

            footsteps.append(footstep)

        return footsteps

    def execute_footstep(self, footstep):
        """Execute a single footstep"""
        # In a real implementation, this would send commands to the
        # humanoid's walking controller
        cmd_msg = String()
        cmd_msg.data = f"step_to:{footstep['position'][0]},{footstep['position'][1]},{footstep['orientation']},{footstep['foot_type']}"
        self.cmd_vel_pub.publish(cmd_msg)

    def get_current_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())

            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = transform.transform.translation.x
            pose_stamped.pose.position.y = transform.pose.position.y
            pose_stamped.pose.position.z = transform.transform.translation.z
            pose_stamped.pose.orientation = transform.transform.rotation

            return pose_stamped
        except TransformException as ex:
            self.get_logger().error(f'Could not get transform: {ex}')
            return None

    def calculate_distance(self, pose1, pose2):
        """Calculate 2D distance between two poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx*dx + dy*dy)

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
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

class HumanoidCostmap:
    """Custom costmap for humanoid robots with balance and kinematic constraints"""

    def __init__(self, resolution=0.1, width=100, height=100):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2

        # Initialize costmap (0 = free, 255 = occupied)
        self.costmap = np.zeros((height, width), dtype=np.uint8)

        # Humanoid-specific parameters
        self.humanoid_width = 0.6  # shoulder width
        self.humanoid_height = 0.5  # step-over height
        self.inflation_radius = 0.8  # safety margin

    def update_with_laser(self, laser_scan):
        """Update costmap with laser scan data"""
        # Convert laser ranges to occupancy grid
        for i, range_val in enumerate(laser_scan.ranges):
            if not np.isnan(range_val) and range_val < laser_scan.range_max:
                angle = laser_scan.angle_min + i * laser_scan.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                # Mark cell as occupied
                self.set_cost(x, y, 255)

                # Inflate around obstacle for humanoid safety
                self.inflate_around_point(x, y, self.inflation_radius)

    def set_cost(self, world_x, world_y, cost):
        """Set cost at world coordinates"""
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)

        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            self.costmap[grid_y, grid_x] = cost

    def inflate_around_point(self, world_x, world_y, radius):
        """Inflate costmap around a point with humanoid safety margin"""
        grid_radius = int(radius / self.resolution)
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                dist = math.sqrt(dx*dx + dy*dy)
                if dist <= grid_radius:
                    new_x = grid_x + dx
                    new_y = grid_y + dy

                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        # Only increase cost, don't decrease
                        if self.costmap[new_y, new_x] < 200:
                            self.costmap[new_y, new_x] = 200

    def is_navigable(self, point):
        """Check if a point is navigable for humanoid"""
        grid_x = int((point[0] - self.origin_x) / self.resolution)
        grid_y = int((point[1] - self.origin_y) / self.resolution)

        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.costmap[grid_y, grid_x] < 100  # Threshold for navigable

        return False

    def plan_path(self, start, goal):
        """Plan path using A* algorithm with humanoid constraints"""
        # Simple A* implementation for demonstration
        # In real implementation, would use more sophisticated path planners
        # that consider humanoid kinematic constraints

        # Convert to grid coordinates
        start_grid = (
            int((start[0] - self.origin_x) / self.resolution),
            int((start[1] - self.origin_y) / self.resolution)
        )
        goal_grid = (
            int((goal[0] - self.origin_x) / self.resolution),
            int((goal[1] - self.origin_y) / self.resolution)
        )

        # Check if start and goal are valid
        if (not (0 <= start_grid[0] < self.width and 0 <= start_grid[1] < self.height) or
            not (0 <= goal_grid[0] < self.width and 0 <= goal_grid[1] < self.height)):
            return None

        if self.costmap[goal_grid[1], goal_grid[0]] >= 250:
            return None  # Goal is occupied

        # For simplicity, return direct path (in real implementation, use A*)
        path = [start, goal]
        return path

    def has_obstacles_along_path(self, start, end):
        """Check if there are obstacles along a path"""
        # Simple line-of-sight check
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)

        if distance == 0:
            return False

        steps = int(distance / self.resolution)
        for i in range(steps):
            t = i / steps
            x = start[0] + t * dx
            y = start[1] + t * dy

            if not self.is_navigable((x, y)):
                return True

        return False

    def plan_local_path(self, start, goal):
        """Plan local path around obstacles"""
        # For local planning, consider immediate obstacles
        # In real implementation, would use local planners like DWA or Teb
        return [start, goal]

def main(args=None):
    rclpy.init(args=args)

    nav2_server = HumanoidNav2Server()

    try:
        rclpy.spin(nav2_server)
    except KeyboardInterrupt:
        pass
    finally:
        nav2_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# Nav2 behavior trees and recovery behaviors for humanoid robots
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String
import math
import numpy as np
from typing import List, Dict, Any
import time

class HumanoidNav2Behaviors(Node):
    """Custom Nav2 behaviors and recovery for humanoid robots"""

    def __init__(self):
        super().__init__('humanoid_nav2_behaviors')

        # Create action client to interact with Nav2 server
        self.nav_to_pose_client = self.create_client(
            NavigateToPose, 'navigate_to_pose')

        # Subscribers for sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Publishers for commands and status
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/nav2_status', 10)

        # Navigation state
        self.current_scan = None
        self.current_imu = None
        self.navigation_state = 'IDLE'  # IDLE, NAVIGATING, RECOVERING, STOPPED
        self.balance_ok = True
        self.obstacle_detected = False

        # Humanoid-specific parameters
        self.balance_threshold = 0.15  # Radians
        self.obstacle_distance_threshold = 0.8  # Meters
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

        # Recovery behaviors
        self.recovery_behaviors = [
            self.backup_recovery,
            self.spot_turn_recovery,
            self.wander_recovery
        ]

        self.get_logger().info('Humanoid Nav2 Behaviors initialized')

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        self.current_scan = msg

        # Check for nearby obstacles
        if msg.ranges:
            min_range = min([r for r in msg.ranges if not math.isnan(r)], default=float('inf'))
            self.obstacle_detected = min_range < self.obstacle_distance_threshold

    def imu_callback(self, msg):
        """Process IMU for balance monitoring"""
        self.current_imu = msg

        # Extract roll and pitch from orientation
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        roll, pitch, _ = self.quaternion_to_euler(quat)

        # Check balance thresholds
        self.balance_ok = (abs(roll) < self.balance_threshold and
                          abs(pitch) < self.balance_threshold)

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]

    def check_navigation_safety(self):
        """Check if navigation is safe to continue"""
        # Check balance
        if not self.balance_ok:
            self.get_logger().warn('Balance compromised, stopping navigation')
            return False

        # Check for obstacles
        if self.obstacle_detected:
            self.get_logger().info('Obstacle detected, assessing situation')
            return self.assess_obstacle_safety()

        return True

    def assess_obstacle_safety(self):
        """Assess if obstacle can be safely navigated around"""
        if not self.current_scan:
            return True  # Can't assess, assume safe

        # Analyze scan to find safe passage
        ranges = self.current_scan.ranges
        angle_increment = self.current_scan.angle_increment
        angle_min = self.current_scan.angle_min

        # Look for gaps in obstacles
        safe_angles = []
        for i, r in enumerate(ranges):
            if not math.isnan(r) and r > self.obstacle_distance_threshold + 0.3:
                angle = angle_min + i * angle_increment
                safe_angles.append(angle)

        # If we found safe angles, navigation can continue
        return len(safe_angles) > 0

    def execute_recovery_behavior(self):
        """Execute appropriate recovery behavior for humanoid"""
        if self.recovery_attempts >= self.max_recovery_attempts:
            self.get_logger().error('Max recovery attempts reached, stopping navigation')
            self.stop_navigation()
            return False

        self.get_logger().info(f'Executing recovery behavior {self.recovery_attempts + 1}')

        # Try each recovery behavior in sequence
        for behavior in self.recovery_behaviors:
            if behavior():
                self.recovery_attempts += 1
                self.get_logger().info(f'Recovery behavior {behavior.__name__} successful')
                return True

        self.recovery_attempts += 1
        self.get_logger().warn('All recovery behaviors failed, trying next')
        return self.recovery_attempts < self.max_recovery_attempts

    def backup_recovery(self):
        """Humanoid-specific backup recovery behavior"""
        self.get_logger().info('Executing backup recovery')

        # For humanoid, backup might involve stepping backwards
        # In simulation, we'll send a reverse velocity command
        cmd = Twist()
        cmd.linear.x = -0.2  # Back up slowly
        cmd.angular.z = 0.0

        # Execute for 2 seconds
        start_time = time.time()
        while time.time() - start_time < 2.0:
            if not self.balance_ok:
                self.stop_navigation()
                return False
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

        # Stop
        cmd.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd)

        return True

    def spot_turn_recovery(self):
        """Humanoid-specific spot turn recovery"""
        self.get_logger().info('Executing spot turn recovery')

        # Turn in place to reassess the situation
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # Turn at 0.5 rad/s

        # Execute for 2 seconds (about 90 degree turn)
        start_time = time.time()
        while time.time() - start_time < 2.0:
            if not self.balance_ok:
                self.stop_navigation()
                return False
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

        # Stop
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        return True

    def wander_recovery(self):
        """Humanoid-specific wandering recovery"""
        self.get_logger().info('Executing wander recovery')

        # Move in a small random pattern to get unstuck
        import random

        # Random direction and distance
        angle = random.uniform(-math.pi, math.pi)
        distance = random.uniform(0.5, 1.5)

        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward
        cmd.angular.z = 0.0  # No turn initially

        # Calculate time to travel the distance
        travel_time = distance / 0.2

        start_time = time.time()
        while time.time() - start_time < travel_time:
            if not self.balance_ok:
                self.stop_navigation()
                return False
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

        # Stop
        cmd.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd)

        return True

    def stop_navigation(self):
        """Stop all navigation commands"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        self.navigation_state = 'STOPPED'
        status_msg = String()
        status_msg.data = 'NAVIGATION_STOPPED'
        self.status_pub.publish(status_msg)

    def navigate_with_safety(self, goal_pose):
        """Navigate with safety checks and recovery behaviors"""
        self.navigation_state = 'NAVIGATING'
        self.recovery_attempts = 0

        # In a real implementation, we would send the goal to Nav2
        # and monitor the execution with our safety checks
        # For this example, we'll simulate the navigation loop

        start_time = time.time()
        while self.navigation_state == 'NAVIGATING':
            # Check if navigation is safe to continue
            if not self.check_navigation_safety():
                # Try recovery behaviors
                if not self.execute_recovery_behavior():
                    self.get_logger().error('Recovery failed, stopping navigation')
                    self.stop_navigation()
                    break

            # Publish status
            status_msg = String()
            status_msg.data = f'NAVIGATING_WITH_SAFETY_{self.recovery_attempts}'
            self.status_pub.publish(status_msg)

            # Small delay
            time.sleep(0.1)

            # Simulate reaching goal after some time
            if time.time() - start_time > 30.0:  # 30 seconds timeout
                self.get_logger().info('Navigation completed (simulated)')
                self.navigation_state = 'IDLE'
                break

    def send_navigation_goal(self, x, y, theta=0.0):
        """Send navigation goal to Nav2 server"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        cos_half_theta = math.cos(theta / 2.0)
        sin_half_theta = math.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = cos_half_theta
        goal_msg.pose.pose.orientation.z = sin_half_theta

        # In a real implementation, we would send this goal to the Nav2 server
        # For this example, we'll start our simulated navigation
        self.navigate_with_safety(goal_msg.pose)

class HumanoidPathPlanner(Node):
    """Advanced path planning for humanoid robots with balance constraints"""

    def __init__(self):
        super().__init__('humanoid_path_planner')

        # Publishers and subscribers
        self.path_pub = self.create_publisher(String, '/humanoid/planned_path', 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        # Path planning parameters
        self.grid_resolution = 0.2  # meters per cell
        self.robot_radius = 0.35    # Humanoid shoulder width / 2
        self.max_slope = 0.3        # Maximum passable slope
        self.step_height = 0.1      # Maximum step-over height

        # Grid map for path planning
        self.grid_map = None
        self.map_width = 100
        self.map_height = 100
        self.origin_x = 0.0
        self.origin_y = 0.0

        self.get_logger().info('Humanoid Path Planner initialized')

    def laser_callback(self, msg):
        """Update grid map with laser data"""
        if self.grid_map is None:
            self.grid_map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        # Convert laser scan to grid map
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        for i, range_val in enumerate(msg.ranges):
            if not math.isnan(range_val) and range_val < msg.range_max:
                angle = angle_min + i * angle_increment
                world_x = range_val * math.cos(angle)
                world_y = range_val * math.sin(angle)

                # Convert to grid coordinates
                grid_x = int((world_x - self.origin_x) / self.grid_map.shape[1] * self.grid_resolution)
                grid_y = int((world_y - self.origin_y) / self.grid_map.shape[0] * self.grid_resolution)

                if 0 <= grid_x < self.grid_map.shape[1] and 0 <= grid_y < self.grid_map.shape[0]:
                    # Mark as occupied
                    self.grid_map[grid_y, grid_x] = 255

    def plan_humanoid_path(self, start, goal):
        """Plan path considering humanoid-specific constraints"""
        # In a real implementation, this would use advanced path planning
        # algorithms that consider humanoid kinematics and balance
        # For this example, we'll implement a basic A* with humanoid constraints

        # Convert start and goal to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        if not self.is_valid_grid_cell(start_grid) or not self.is_valid_grid_cell(goal_grid):
            return None

        # Check if goal is free
        if self.grid_map[goal_grid[1], goal_grid[0]] > 200:
            return None

        # Run A* path planning
        path = self.a_star_search(start_grid, goal_grid)

        if path:
            # Convert back to world coordinates
            world_path = [self.grid_to_world(grid_pos) for grid_pos in path]
            return world_path

        return None

    def a_star_search(self, start, goal):
        """A* pathfinding algorithm with humanoid constraints"""
        from queue import PriorityQueue

        def heuristic(pos1, pos2):
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while not open_set.empty():
            current = open_set.get()[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                if not self.is_valid_grid_cell(neighbor):
                    continue

                # Check if neighbor is passable for humanoid
                if self.grid_map[neighbor[1], neighbor[0]] > 150:  # Occupied or dangerous
                    continue

                tentative_g_score = g_score[current] + heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))

        return None  # No path found

    def get_neighbors(self, pos):
        """Get 8-connected neighbors for path planning"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((pos[0] + dx, pos[1] + dy))
        return neighbors

    def is_valid_grid_cell(self, pos):
        """Check if grid position is valid"""
        x, y = pos
        return (0 <= x < self.grid_map.shape[1] and
                0 <= y < self.grid_map.shape[0])

    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_pos[0] - self.origin_x) / self.grid_resolution)
        grid_y = int((world_pos[1] - self.origin_y) / self.grid_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_pos):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_pos[0] * self.grid_resolution + self.origin_x
        world_y = grid_pos[1] * self.grid_resolution + self.origin_y
        return (world_x, world_y)

def main(args=None):
    rclpy.init(args=args)

    # Create navigation behavior nodes
    behaviors_node = HumanoidNav2Behaviors()
    path_planner = HumanoidPathPlanner()

    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(behaviors_node)
    executor.add_node(path_planner)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        behaviors_node.destroy_node()
        path_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Navigation2 (Nav2) provides a comprehensive navigation framework that can be customized for humanoid robots with their unique balance, kinematic, and locomotion constraints. Unlike traditional navigation for wheeled robots, humanoid navigation must consider complex factors like bipedal locomotion, balance maintenance, and human-like form factor limitations.

The key aspects of Nav2 for humanoid robots include:
- **Humanoid-Specific Path Planning**: Algorithms that consider step size, turning radius, and balance constraints
- **Balance-Aware Navigation**: Integration with IMU and locomotion controllers to maintain stability
- **Adaptive Recovery Behaviors**: Specialized recovery actions suitable for bipedal robots
- **Custom Costmaps**: Maps that account for humanoid dimensions and capabilities

For Physical AI and humanoid robotics, Nav2 provides the foundation for safe and effective navigation in human environments. The framework's flexibility allows for customization of planners, controllers, and safety systems to match the specific requirements of different humanoid platforms.

The integration of Nav2 with humanoid-specific locomotion controllers enables robots to navigate complex environments while maintaining balance and following safe, efficient paths that account for their unique kinematic constraints.

## Exercises

1. **Basic Understanding**: Explain the key differences between navigation for wheeled robots and humanoid robots. What specific constraints must be considered for humanoid navigation?

2. **Application Exercise**: Design a Nav2 configuration for a humanoid robot that needs to navigate through a cluttered home environment. Include custom parameters for step size, balance thresholds, and recovery behaviors suitable for indoor navigation.

3. **Implementation Exercise**: Create a custom Nav2 plugin that modifies the local planner to consider humanoid balance constraints. The plugin should adjust the robot's path based on real-time IMU data to maintain stability.

4. **Challenge Exercise**: Implement a complete navigation system that integrates Nav2 with a humanoid's walking controller. Include path planning, obstacle avoidance, balance monitoring, and recovery behaviors that work together to enable safe navigation.
> **Advanced Exercises**: Challenge students with production-level implementations and performance optimization.

