# ROS 2 Nodes, Topics, Services & Actions (Advanced Level)

## Concept

In ROS 2, the fundamental building blocks of a robotic system are Nodes, Topics, Services, and Actions. These components work together to create a distributed system where different parts of your robot can communicate and coordinate with each other.

These components serve different communication patterns in robotic systems:
- **Nodes** are independent processes that perform specific computational tasks (like sensor processing, motion control, or decision making)
- **Topics** enable asynchronous, continuous data flow using publish/subscribe patterns (e.g., sensor data, robot status)
- **Services** provide synchronous request/response communication for immediate operations (e.g., asking the robot to move to a specific location)
- **Actions** handle long-running tasks with feedback and cancellation capabilities (e.g., navigation to a distant location)

These components matter in robotics because robots typically need to coordinate multiple simultaneous operations. For example, while a robot is executing a long navigation task (action), it needs to continuously receive sensor data (topics) and may need to respond to immediate requests (services). Having different communication patterns allows for flexible and robust robot software architecture.

The distributed nature of ROS 2 allows nodes to run on different machines while maintaining seamless communication through DDS (Data Distribution Service), making it ideal for Physical AI systems where computation might be distributed across different hardware components.



> **Best Practice**: For production systems, consider [advanced technique] to optimize performance.

> **Performance Note**: This approach has O(n) complexity and may require optimization for large-scale applications.

## Diagram

```
                    ROS 2 COMMUNICATION PATTERNS
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
    TOPIC COMMUNICATION   SERVICE COMMUNICATION   ACTION COMMUNICATION
        │                     │                     │
    ┌───▼───┐            ┌─────▼─────┐         ┌───▼───┐
    │Sensor │            │  Request  │         │  Goal   │
    │Node   │───────────►│  /move    │◄────────┤Action  │
    │       │            │  Robot    │         │Client  │
    │Publish│            │   (Sync)  │         │        │
    └───────┘            └───────────┘         └───────┬─┘
         │                       │                      │
         ▼                       ▼                      ▼
    ┌─────────┐            ┌─────────────┐        ┌───────────┐
    │  Topic  │            │   Service   │        │   Action  │
    │ /sensor │            │  /move_robot│        │  Server   │
    │  Data   │            │             │        │  /navigate│
    └─────────┘            └─────────────┘        │   _to     │
         │                       │                 │   _goal   │
         ▼                       ▼                 └───────────┘
    ┌─────────┐            ┌─────────────┐               │
    │Process- │            │   Response  │               │
    │ing Node │◄───────────┤   Result    │◄──────────────┤
    │         │            │             │               │
    │Subscribe│            │   (Sync)    │               │
    └─────────┘            └─────────────┘               │
                                                         │
                                                  ┌──────▼──────┐
                                                  │   Feedback  │
                                                  │   Updates   │
                                                  │   (Async)   │
                                                  └─────────────┘
```

## Real-world Analogy

Think of these ROS 2 communication patterns like different ways people communicate in a large, complex organization:

- **Nodes** are like different departments in a company (Engineering, Sales, HR), each with specific responsibilities and expertise
- **Topics** are like a company-wide newsletter or announcement board - information is broadcast continuously and anyone interested can receive it (e.g., daily reports, status updates, sensor readings)
- **Services** are like making a direct phone call or sending a message requiring an immediate response (e.g., "Can you approve this request?" or "Move the robot to location X?")
- **Actions** are like assigning a complex project that takes time to complete - you give the goal, receive progress updates along the way, and get a final result (e.g., "Navigate to the conference room - update me on your progress")

Just as an organization needs different communication channels for different types of information, a robot needs these different communication patterns to coordinate its complex behaviors effectively. The beauty of ROS 2 is that these communication patterns work together seamlessly, allowing for sophisticated robot behaviors.

## Pseudo-code (ROS 2 / Python)
# Advanced Implementation:
# - Real-time performance considerations
# - Memory management optimizations
# - Parallel processing opportunities
# - Safety and fault-tolerance measures
# - Hardware-specific optimizations



```python
# ROS 2 Node for Physical AI Communication Patterns
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from example_interfaces.srv import SetBool
from example_interfaces.action import Fibonacci
import rclpy.action
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse

class CommunicationPatternsNode(Node):
    def __init__(self):
        super().__init__('communication_patterns')

        # Create QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # TOPIC PUBLISHER - Publish robot status
        self.status_publisher = self.create_publisher(
            String, '/robot_status', 10)

        # TOPIC SUBSCRIBER - Subscribe to sensor data
        self.joint_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, sensor_qos)

        self.imu_subscriber = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, sensor_qos)

        # SERVICE SERVER - Handle immediate requests
        self.emergency_service = self.create_service(
            SetBool, '/emergency_stop', self.emergency_callback)

        # ACTION SERVER - Handle long-running tasks
        self.navigation_action_server = ActionServer(
            self,
            Fibonacci,  # Using Fibonacci as example action
            '/navigate_to_goal',
            self.execute_navigation_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Internal state
        self.is_emergency = False
        self.current_joint_positions = {}
        self.navigation_active = False

        # Timer for periodic status updates
        self.status_timer = self.create_timer(1.0, self.publish_status)

    def joint_callback(self, msg):
        """Handle incoming joint state data (Topic subscription)"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def imu_callback(self, msg):
        """Handle incoming IMU data (Topic subscription)"""
        # Process IMU data for balance, orientation, etc.
        orientation = msg.orientation
        self.get_logger().debug(f'IMU orientation: ({orientation.x}, {orientation.y}, {orientation.z}, {orientation.w})')

    def emergency_callback(self, request, response):
        """Handle emergency stop service request"""
        if request.data:  # If request is to activate emergency stop
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            self.is_emergency = True
            response.success = True
            response.message = 'Emergency stop activated'
        else:  # If request is to clear emergency stop
            self.get_logger().info('Emergency stop cleared')
            self.is_emergency = False
            response.success = True
            response.message = 'Emergency stop cleared'

        return response

    def goal_callback(self, goal_request):
        """Accept or reject navigation goal"""
        self.get_logger().info('Received navigation goal request')

        # Check if we can accept the goal
        if self.is_emergency:
            self.get_logger().warn('Cannot accept navigation goal during emergency')
            return GoalResponse.REJECT
        elif self.navigation_active:
            self.get_logger().warn('Navigation already active, rejecting new goal')
            return GoalResponse.REJECT
        else:
            self.get_logger().info('Accepting navigation goal')
            return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject goal cancel request"""
        self.get_logger().info('Received navigation cancel request')
        return CancelResponse.ACCEPT

    def execute_navigation_callback(self, goal_handle):
        """Execute the navigation task (Action server)"""
        self.get_logger().info('Starting navigation task...')
        self.navigation_active = True

        # Initialize feedback and result messages
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()

        # Simulate navigation progress
        sequence = [0, 1]

        try:
            for i in range(1, goal_handle.request.order):
                # Check if the goal was canceled
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.get_logger().info('Navigation goal canceled')
                    result_msg.sequence = sequence
                    self.navigation_active = False
                    return result_msg

                # Update feedback
                sequence.append(sequence[i] + sequence[i-1])
                feedback_msg.sequence = sequence
                goal_handle.publish_feedback(feedback_msg)

                # Log progress
                self.get_logger().info(f'Navigation progress: {len(sequence)}/{goal_handle.request.order}')

                # Sleep to simulate actual navigation time
                from time import sleep
                sleep(0.5)

                # Check for emergency during navigation
                if self.is_emergency:
                    goal_handle.abort()
                    self.get_logger().warn('Navigation aborted due to emergency')
                    result_msg.sequence = sequence
                    self.navigation_active = False
                    return result_msg

            # Complete the goal successfully
            goal_handle.succeed()
            result_msg.sequence = sequence

            self.get_logger().info(f'Navigation completed with sequence: {sequence}')
        except Exception as e:
            goal_handle.abort()
            self.get_logger().error(f'Navigation failed with error: {e}')
            result_msg.sequence = sequence

        self.navigation_active = False
        return result_msg

    def publish_status(self):
        """Publish robot status periodically (Topic publishing)"""
        status_msg = String()
        status_msg.data = f'Operational - Joints: {len(self.current_joint_positions)}, Emergency: {self.is_emergency}'
        self.status_publisher.publish(status_msg)

class CommunicationClientNode(Node):
    def __init__(self):
        super().__init__('communication_client')

        # SERVICE CLIENT - For immediate requests
        self.emergency_client = self.create_client(SetBool, '/emergency_stop')

        # ACTION CLIENT - For long-running tasks
        self.navigation_action_client = ActionClient(self, Fibonacci, '/navigate_to_goal')

        # Wait for services/actions to be available
        while not self.emergency_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Emergency service not available, waiting...')

    def send_emergency_request(self, activate):
        """Send emergency stop request (Service client)"""
        request = SetBool.Request()
        request.data = activate

        future = self.emergency_client.call_async(request)
        return future

    def send_navigation_goal(self, order):
        """Send navigation goal (Action client)"""
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        # Wait for action server
        self.navigation_action_client.wait_for_server()

        # Send goal with feedback callback
        send_goal_future = self.navigation_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        return send_goal_future

    def feedback_callback(self, feedback_msg):
        """Handle feedback from navigation action"""
        self.get_logger().info(f'Navigation feedback: {feedback_msg.feedback.sequence}')

def main(args=None):
    rclpy.init(args=args)

    # Create and run the communication patterns node
    comm_node = CommunicationPatternsNode()

    try:
        rclpy.spin(comm_node)
    except KeyboardInterrupt:
        comm_node.get_logger().info('Shutting down communication patterns node')
    finally:
        comm_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

ROS 2 provides four fundamental communication patterns that enable complex robot behaviors:

- **Nodes** serve as the basic execution units containing your robot's functionality
- **Topics** enable asynchronous, continuous data flow using publish/subscribe patterns
- **Services** provide synchronous request/response communication for immediate operations
- **Actions** handle long-running tasks with feedback and cancellation capabilities

These patterns work together to create a flexible, distributed system where different components can communicate efficiently based on their specific needs. Understanding when to use each pattern is crucial for designing effective robotic applications in Physical AI.

For Physical AI systems, these communication patterns enable the coordination of sensors, controllers, and actuators across distributed hardware, allowing for sophisticated robot behaviors while maintaining real-time performance and reliability.

## Exercises

1. **Conceptual Exercise**: Explain the difference between a ROS 2 Service and a ROS 2 Action. Provide specific examples of when you would use each one in a humanoid robot application.

2. **Design Exercise**: Design a complete ROS 2 system for a mobile manipulator robot with the following components: camera node, joint controller, navigation system, and user interface. Identify which communication patterns each component would use.

3. **Programming Exercise**: Create a ROS 2 node that subscribes to IMU data and publishes a filtered orientation estimate. Include proper error handling and QoS configuration.

4. **Integration Exercise**: Extend the provided example to include a service that allows querying the current joint positions of the robot.

5. **Advanced Exercise**: Implement a safety system using ROS 2 communication patterns that monitors robot state and can trigger emergency procedures when dangerous conditions are detected.
> **Advanced Exercises**: Challenge students with production-level implementations and performance optimization.

