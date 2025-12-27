# ROS 2 Basics for Physical AI (Advanced Level)

## Concept

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software that provides a collection of tools, libraries, and conventions to simplify the development of complex and robust robot behavior across a wide variety of robot platforms. Unlike traditional software frameworks, ROS 2 is designed specifically for robotics applications, emphasizing distributed computation, real-time performance, and hardware abstraction.

In the context of Physical AI and humanoid robotics, ROS 2 serves as the middleware that enables different components of a robot system to communicate and coordinate effectively. It provides standardized message types, communication patterns, and tools that allow developers to focus on implementing robot behavior rather than low-level communication details.

ROS 2 is built on DDS (Data Distribution Service) which provides reliable, real-time communication between distributed components. This is crucial for Physical AI systems where sensors, controllers, and actuators may be running on different computers or processing units but need to coordinate in real-time.

Key concepts in ROS 2 include:
- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous goal-oriented communication
- **Parameters**: Configuration values that can be changed at runtime



> **Best Practice**: For production systems, consider [advanced technique] to optimize performance.

> **Performance Note**: This approach has O(n) complexity and may require optimization for large-scale applications.

## Diagram

```
                    ROS 2 ARCHITECTURE OVERVIEW
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
    ROS 2 NODES          COMMUNICATION        TOOLS & UTILITIES
        │                     │                     │
    ┌───▼───┐            ┌─────▼─────┐         ┌───▼───┐
    │Sensor │            │  Topics   │         │ rqt   │
    │Nodes  │ ◄──────────┤  Services │────────►│ rviz  │
    │       │            │  Actions  │         │ ros2  │
    │Action │            │ Parameters│         │ tools │
    │Nodes  │            │  DDS      │         │       │
    │       │            │  Layer    │         │       │
    └───────┘            └───────────┘         └───────┘
         │                       │                   │
         ▼                       ▼                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │                 ROS 2 RUNTIME                           │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
    │  │  Physical   │   │  Middleware │   │  Developer  │  │
    │  │  Hardware   │   │   Layer     │   │  Tools      │  │
    │  └─────────────┘   └─────────────┘   └─────────────┘  │
    └─────────────────────────────────────────────────────────┘
```

## Real-world Analogy

Think of ROS 2 like a sophisticated communication network in a large hospital. Just as a hospital has different departments (emergency, surgery, radiology) that need to share information and coordinate patient care, a robot has different components (sensors, controllers, actuators) that need to share data and coordinate actions.

In a hospital, information flows through standardized forms, electronic records, and communication protocols. Nurses, doctors, and technicians all use the same systems to share patient information, request services, and coordinate care. Similarly, in ROS 2, different software components (nodes) use standardized message types and communication patterns (topics, services, actions) to share sensor data, request computations, and coordinate robot behavior.

The DDS middleware in ROS 2 is like the hospital's internal communication system - it ensures that the right information gets to the right department at the right time, even when some communication paths are temporarily unavailable. Just as a hospital needs reliable communication to provide quality patient care, a robot needs reliable communication between its components to operate safely and effectively.

## Pseudo-code (ROS 2 / Python)
# Advanced Implementation:
# - Real-time performance considerations
# - Memory management optimizations
# - Parallel processing opportunities
# - Safety and fault-tolerance measures
# - Hardware-specific optimizations



```python
# ROS 2 Node for Basic Physical AI System
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time

class RobotBasicsNode(Node):
    def __init__(self):
        super().__init__('robot_basics')

        # Create a QoS profile for sensor data (reliable, keep last 10 samples)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create a QoS profile for command data (best effort, keep last 1 sample)
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscribers for sensor data
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            sensor_qos
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            sensor_qos
        )

        # Create publishers for commands and status
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            command_qos
        )

        self.status_pub = self.create_publisher(
            String,
            '/robot_status',
            10
        )

        # Create a service server for basic commands
        self.basic_command_service = self.create_service(
            String,
            '/basic_command',
            self.command_callback
        )

        # Create a parameter to control behavior
        self.declare_parameter('robot_mode', 'idle')
        self.declare_parameter('max_velocity', 0.5)

        # Internal state
        self.current_joint_positions = {}
        self.current_imu_data = None
        self.robot_mode = 'idle'

        # Create a timer for periodic tasks
        self.timer = self.create_timer(0.1, self.periodic_task)  # 10 Hz

    def joint_callback(self, msg):
        """Process joint state messages"""
        # Update internal joint position tracking
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

        # Log when we receive joint data
        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def imu_callback(self, msg):
        """Process IMU messages"""
        # Store IMU data
        self.current_imu_data = {
            'orientation': (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w),
            'angular_velocity': (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z),
            'linear_acceleration': (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        }

        # Check for balance issues
        self.check_balance_safety()

    def check_balance_safety(self):
        """Check if robot is in a safe orientation based on IMU data"""
        if self.current_imu_data:
            # Extract orientation quaternion
            x, y, z, w = self.current_imu_data['orientation']

            # Calculate pitch and roll (simplified)
            pitch = 2.0 * (w * y - z * x)
            roll = 2.0 * (w * x + y * z)

            # Define safety thresholds
            max_roll = 0.5  # About 28 degrees
            max_pitch = 0.5  # About 28 degrees

            if abs(roll) > max_roll or abs(pitch) > max_pitch:
                self.get_logger().warn(f'Robot may be in unsafe orientation: roll={roll}, pitch={pitch}')
                # In a real system, we might trigger safety procedures here

    def command_callback(self, request, response):
        """Handle basic command service requests"""
        command = request.data.lower()

        if command == 'start':
            self.robot_mode = 'active'
            response.data = 'Robot started'
        elif command == 'stop':
            self.robot_mode = 'idle'
            response.data = 'Robot stopped'
        elif command == 'status':
            response.data = f'Robot mode: {self.robot_mode}'
        else:
            response.data = f'Unknown command: {command}'

        self.get_logger().info(f'Received command: {command}, Response: {response.data}')
        return response

    def periodic_task(self):
        """Execute periodic tasks at 10 Hz"""
        # Update robot status
        status_msg = String()
        status_msg.data = f'Robot operating in {self.robot_mode} mode'
        self.status_pub.publish(status_msg)

        # Update robot mode from parameters
        self.robot_mode = self.get_parameter('robot_mode').value

        # Example: Send a velocity command if in active mode
        if self.robot_mode == 'active':
            self.send_velocity_command()

    def send_velocity_command(self):
        """Send a velocity command to move the robot"""
        cmd = Twist()

        # Get max velocity from parameters
        max_vel = self.get_parameter('max_velocity').value

        # Simple movement pattern based on mode
        if self.robot_mode == 'active':
            cmd.linear.x = max_vel * 0.5  # Move forward at half speed
            cmd.angular.z = 0.0  # No rotation
        else:
            cmd.linear.x = 0.0  # Stop
            cmd.angular.z = 0.0  # No rotation

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)

    # Create the node
    robot_node = RobotBasicsNode()

    try:
        # Spin the node to process callbacks
        rclpy.spin(robot_node)
    except KeyboardInterrupt:
        robot_node.get_logger().info('Interrupted by user')
    finally:
        # Clean up
        robot_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

ROS 2 provides the essential middleware infrastructure for Physical AI and humanoid robotics applications. It enables different components of a robot system to communicate and coordinate effectively through standardized message types and communication patterns.

Key concepts in ROS 2 include nodes (computational processes), topics (asynchronous message passing), services (synchronous request/response), actions (goal-oriented communication), and parameters (runtime configuration). The DDS-based communication layer ensures reliable, real-time communication between distributed components.

For Physical AI applications, ROS 2 provides the foundation for building complex, distributed robot systems where sensors, controllers, and actuators can operate independently while maintaining coordinated behavior. The framework's emphasis on hardware abstraction allows the same algorithms to run on different robot platforms, making it ideal for Physical AI research and development.

## Exercises

1. **Setup Exercise**: Install ROS 2 (Humble Hawksbill or later) on your development machine and verify the installation by running the basic publisher/subscriber tutorial.

2. **Conceptual Exercise**: Design a ROS 2 system architecture for a simple mobile robot with a camera, IMU, and differential drive. Identify what nodes you would create and what topics/services they would use.

3. **Programming Exercise**: Create a ROS 2 node that subscribes to a camera topic and publishes the average brightness of the image as a Float64 message.

4. **Integration Exercise**: Modify the provided example to include a service that allows external nodes to query the current joint positions of the robot.

5. **Advanced Exercise**: Implement a simple action server that accepts navigation goals and provides feedback on progress, similar to ROS 2 Navigation 2 (Nav2).
> **Advanced Exercises**: Challenge students with production-level implementations and performance optimization.

