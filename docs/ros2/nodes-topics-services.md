# ROS 2 Nodes, Topics, Services, and Actions

## Concept

In ROS 2, the fundamental building blocks of a robotic system are Nodes, Topics, Services, and Actions. These components work together to create a distributed system where different parts of your robot can communicate and coordinate with each other.

Think of these as different communication patterns in a software system, similar to how different types of functions and methods in Python serve different purposes:

- **Nodes** are like individual programs or processes - they contain the actual code that performs specific tasks (like sensor processing, motion control, or decision making)
- **Topics** are like pub/sub systems in distributed applications - they allow data to flow continuously between nodes in a fire-and-forget manner (e.g., sensor data, robot status)
- **Services** are like function calls or API endpoints - they provide synchronous request/response communication (e.g., asking the robot to move to a specific location)
- **Actions** are like async functions with progress tracking - they handle long-running tasks that might take time and need feedback (e.g., navigation to a distant location)

These components matter in robotics because robots typically need to coordinate multiple simultaneous operations. For example, while a robot is moving (a long-running task), it needs to continuously receive sensor data (topics) and may need to respond to emergency stop requests (services). Having different communication patterns allows for flexible and robust robot software architecture.

## ASCII Diagram

```
┌─────────────────┐    Publish    ┌──────────────────┐    Subscribe   ┌─────────────────┐
│   Sensor Node   │ ────────────▶ │     Topic        │ ◀──────────── │  Processing Node│
│                 │               │  /sensor_data    │               │                 │
│  ┌───────────┐  │               └──────────────────┘               │  ┌───────────┐  │
│  │Camera/LiDAR│  │                                                  │  │Algorithm  │  │
│  │Sensors    │──┼──────────────────────────────────────────────────┼──│Processor  │  │
│  └───────────┘  │                    Messages                      │  └───────────┘  │
└─────────────────┘                                                 └─────────────────┘

┌─────────────────┐    Service    ┌──────────────────┐    Response   ┌─────────────────┐
│   Client Node   │ ────────────▶ │   Service        │ ◀──────────── │   Server Node   │
│                 │    Request    │  /move_robot     │    Result     │                 │
│  ┌──────────┐   │               │                  │               │  ┌────────────┐ │
│  │Planning  │   │               │                  │               │  │Motion      │ │
│  │Module    │───┼───────────────┼──────────────────┼───────────────┼──│Controller  │ │
│  └──────────┘   │               │                  │               │  │            │ │
└─────────────────┘               └──────────────────┘               └─────────────────┘

┌─────────────────┐    Action     ┌──────────────────┐    Feedback    ┌─────────────────┐
│   Action Client │ ────────────▶ │    Action        │ ◀──────────── │   Action Server │
│                 │   Goal/Cancel │  /navigate_to    │   Result/     │                 │
│  ┌──────────┐   │               │    _goal        │   Status       │  ┌────────────┐ │
│  │Navigation│   │               │                  │                │  │Navigation  │ │
│  │Interface │───┼───────────────┼──────────────────┼────────────────┼──│Controller  │ │
│  └──────────┘   │               │                  │                │  │            │ │
└─────────────────┘               └──────────────────┘                └─────────────────┘
                    (Long-running task with status updates)
```

This diagram shows the three main communication patterns in ROS 2:
- Top: Topic communication (publish/subscribe) for continuous data flow
- Middle: Service communication (request/response) for synchronous operations
- Bottom: Action communication (goal-based with feedback) for long-running tasks

## Real-world Analogy

Think of these ROS 2 communication patterns like different ways people communicate in an organization:

- **Nodes** are like different departments in a company (Engineering, Sales, HR), each with specific responsibilities
- **Topics** are like a company-wide newsletter or announcement board - information is broadcast and anyone interested can read it (e.g., daily reports, status updates)
- **Services** are like making a phone call or sending a direct message - you ask a specific question and get an immediate response (e.g., "Can you approve this request?")
- **Actions** are like assigning a project that takes time to complete - you give the goal, receive progress updates along the way, and get a final result (e.g., "Plan our quarterly strategy - update me on your progress")

Just as an organization needs different communication channels for different types of information, a robot needs these different communication patterns to coordinate its complex behaviors effectively.

## Pseudo-code (ROS 2 / Python style)

```python
# Node Example - Complete Node Structure
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts  # Standard service type
from example_interfaces.action import Fibonacci  # Standard action type
import rclpy.action
from rclpy.action import ActionServer, GoalResponse, CancelResponse

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Topic Publisher - Publish robot status
        self.status_publisher = self.create_publisher(String, '/robot_status', 10)

        # Topic Subscriber - Listen to sensor data
        self.sensor_subscriber = self.create_subscription(
            String,
            '/sensor_data',
            self.sensor_callback,
            10
        )

        # Service Server - Handle movement requests
        self.move_service = self.create_service(
            AddTwoInts,  # Using AddTwoInts as an example service type
            '/move_robot',
            self.move_robot_callback
        )

        # Action Server - Handle navigation tasks
        self.navigation_action_server = ActionServer(
            self,
            Fibonacci,  # Using Fibonacci as an example action type
            '/navigate_to_goal',
            self.execute_navigation_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Timer for periodic status updates
        self.timer = self.create_timer(1.0, self.publish_status)

    def sensor_callback(self, msg):
        """Handle incoming sensor data"""
        self.get_logger().info(f'Received sensor data: {msg.data}')
        # Process sensor data and make decisions

    def move_robot_callback(self, request, response):
        """Handle service request to move robot"""
        self.get_logger().info(f'Received move request: {request.a}, {request.b}')

        # Simulate movement (in real case, this would control actual robot)
        result = request.a + request.b  # Example calculation
        response.sum = result
        self.get_logger().info(f'Movement result: {response.sum}')

        return response

    def goal_callback(self, goal_request):
        """Accept or reject navigation goal"""
        self.get_logger().info('Received navigation goal request')
        # Check if we can accept the goal (e.g., not busy with another task)
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject goal cancel request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_navigation_callback(self, goal_handle):
        """Execute the navigation task"""
        self.get_logger().info('Executing navigation goal...')

        # Simulate a long-running task with feedback
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()

        # Simulate navigation progress
        sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            # Check if the goal was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Navigation goal canceled')
                result_msg.sequence = sequence
                return result_msg

            # Update feedback
            sequence.append(sequence[i] + sequence[i-1])
            feedback_msg.sequence = sequence
            goal_handle.publish_feedback(feedback_msg)

            # Log progress
            self.get_logger().info(f'Navigation progress: {i}/{goal_handle.request.order}')

            # Sleep to simulate actual navigation time
            from time import sleep
            sleep(0.5)

        # Complete the goal
        goal_handle.succeed()
        result_msg.sequence = sequence

        self.get_logger().info(f'Navigation completed with sequence: {sequence}')
        return result_msg

    def publish_status(self):
        """Publish robot status periodically"""
        msg = String()
        msg.data = f'Robot is operational at {self.get_clock().now().seconds_nanoseconds()}'
        self.status_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotControllerNode()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# Client-side code for Services and Actions
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from example_interfaces.action import Fibonacci
import rclpy.action
from rclpy.action import ActionClient

class RobotClientNode(Node):
    def __init__(self):
        super().__init__('robot_client')

        # Create service client
        self.cli = self.create_client(AddTwoInts, '/move_robot')

        # Create action client
        self.action_client = ActionClient(self, Fibonacci, '/navigate_to_goal')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/move_robot service not available, waiting again...')

    def send_movement_request(self, a, b):
        """Send a request to the movement service"""
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        # Call the service asynchronously
        future = self.cli.call_async(request)
        return future

    def send_navigation_goal(self, order):
        """Send a navigation goal to the action server"""
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        # Wait for action server
        self.action_client.wait_for_server()

        # Send goal and return the future
        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        return send_goal_future

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server"""
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback.sequence}')

def main(args=None):
    rclpy.init(args=args)
    robot_client = RobotClientNode()

    # Example: Send a service request
    future = robot_client.send_movement_request(10, 20)

    # Example: Send an action goal
    goal_future = robot_client.send_navigation_goal(10)

    # Spin until the service call is complete
    rclpy.spin_until_future_complete(robot_client, future)

    if future.result() is not None:
        response = future.result()
        robot_client.get_logger().info(f'Service response: {response.sum}')
    else:
        robot_client.get_logger().info('Service call failed')

    robot_client.destroy_node()
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

These patterns work together to create a flexible, distributed system where different components can communicate efficiently based on their specific needs. Understanding when to use each pattern is crucial for designing effective robotic applications.

In upcoming topics, we'll explore specific ROS 2 packages, navigation systems, and how to build complete robotic applications using these communication primitives.

## Exercises

1. **Basic Understanding**: Explain the difference between a ROS 2 Service and a ROS 2 Action. Provide an example of when you would use each one in a robot application.

2. **Application Exercise**: Design a robot system with the following requirements: a camera node that publishes images, an image processing node that subscribes to images and publishes object detections, and a decision-making node that uses a service to request robot movement based on detections. Draw the node architecture and identify the communication patterns used.

3. **Implementation Exercise**: Create a node that acts as both a publisher and subscriber. It should subscribe to a topic called `/temperature_data`, process the data (e.g., calculate average), and publish the result to `/temperature_average`. Include error handling for malformed messages.

4. **Challenge Exercise**: Implement a system with an action server that simulates a robot arm movement. The action should accept a goal with joint positions, provide feedback on the current position during movement, and handle cancellation requests properly.