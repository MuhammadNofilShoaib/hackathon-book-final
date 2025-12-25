# URDF and Robot Description for Humanoid Robots

## Concept

Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models. Think of it as a blueprint or specification document that defines everything about a robot's physical structure - its links (body parts), joints (connections), sensors, and visual properties.

In humanoid robotics, URDF becomes especially important because these robots have complex kinematic structures with multiple degrees of freedom that mimic human-like movement. A humanoid robot typically has a torso, head, two arms with hands, and two legs with feet, all connected through various joint types (revolute, prismatic, fixed) that allow for complex movements.

URDF matters in humanoid robotics because it provides a standardized way to represent the robot's physical structure, which is essential for:
- Simulation environments (Gazebo, RViz)
- Kinematics and dynamics calculations
- Visualization and debugging
- Motion planning and control algorithms
- Collision detection and avoidance

If you're familiar with 3D modeling or CAD software, URDF serves a similar purpose but specifically for robotics applications. Instead of just visual representation, URDF includes physical properties like mass, inertia, and joint limits that are critical for realistic simulation and control.

## ASCII Diagram

```
                    ┌─────────────┐
                    │    Head     │
                    │  [link_1]   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Neck Joint │
                    │ [joint_1]   │
                    └──────┬──────┘
                           │
            ┌──────────────▼──────────────┐
            │           Torso             │
            │         [link_2]            │
            └──────────────┬──────────────┘
                           │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐      ┌─────▼─────┐      ┌───▼────┐
   │  Left   │      │  Waist    │      │ Right  │
   │  Arm    │      │  Joint    │      │  Arm   │
   │[link_3] │      │[joint_2]  │      │[link_4]│
   └────┬────┘      └───────────┘      └───┬────┘
        │                                  │
   ┌────▼────┐                        ┌────▼────┐
   │  Left   │                        │ Right   │
   │  Hand   │                        │  Hand   │
   │[link_5] │                        │[link_6] │
   └─────────┘                        └─────────┘

        ┌─────────────────────────────────────────┐
        │              Humanoid Structure         │
        │            URDF Representation          │
        └─────────────────────────────────────────┘
```

This diagram shows the hierarchical structure of a humanoid robot in URDF. The robot consists of multiple links (body parts) connected by joints, forming a kinematic chain that allows for complex movements similar to human anatomy.

## Real-world Analogy

Think of URDF like an architectural blueprint for a building, but specifically designed for robots. Just as an architectural blueprint specifies the dimensions, materials, and connections between different parts of a building, a URDF file specifies:

- **Links** are like rooms or structural components - they have dimensions, weight, and material properties
- **Joints** are like doors, hinges, or connections between rooms - they define how parts can move relative to each other
- **Materials** are like paint colors or textures that define visual appearance
- **Sensors** are like security cameras or environmental monitors placed at specific locations

Just as architects and engineers use blueprints to plan, build, and visualize buildings, roboticists use URDF files to plan, simulate, and control robots. The difference is that robot blueprints must account for physics, movement, and interaction with the environment.

## Pseudo-code (ROS 2 / Python style)

```xml
<!-- Example URDF for a simple humanoid robot -->
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Materials for visualization -->
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link (torso) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <!-- Left arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.0 0.15 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0.0 0.0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <!-- Right arm (similar to left, with mirrored positions) -->
  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.0 -0.15 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0.0 0.0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <!-- Left leg -->
  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.5" radius="0.06"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.5" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.0 0.08 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.5" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.5" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0.0 0.0 -0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="0.0" effort="100" velocity="1.0"/>
  </joint>

  <!-- Right leg -->
  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.5" radius="0.06"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.5" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_leg"/>
    <origin xyz="0.0 -0.08 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.5" radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.5" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0.0 0.0 -0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="0.0" effort="100" velocity="1.0"/>
  </joint>
</robot>
```

```python
# Python code to work with URDF in ROS 2 - Robot State Publisher example
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math
import numpy as np

class HumanoidStatePublisher(Node):
    def __init__(self):
        super().__init__('humanoid_state_publisher')

        # Create a publisher for joint states
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create a transform broadcaster for TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer to publish joint states at regular intervals
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Initialize joint positions (in radians)
        self.joint_positions = {
            'neck_joint': 0.0,
            'left_shoulder_joint': 0.0,
            'left_elbow_joint': 0.0,
            'right_shoulder_joint': 0.0,
            'right_elbow_joint': 0.0,
            'left_hip_joint': 0.0,
            'left_knee_joint': 0.0,
            'right_hip_joint': 0.0,
            'right_knee_joint': 0.0
        }

        self.time = 0.0

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Update joint positions with some oscillating motion for demonstration
        self.time += 0.1
        self.joint_positions['neck_joint'] = 0.3 * math.sin(self.time)
        self.joint_positions['left_shoulder_joint'] = 0.5 * math.sin(self.time * 0.5)
        self.joint_positions['right_shoulder_joint'] = 0.5 * math.sin(self.time * 0.5 + math.pi)
        self.joint_positions['left_elbow_joint'] = 0.3 * math.sin(self.time * 0.7)
        self.joint_positions['right_elbow_joint'] = 0.3 * math.sin(self.time * 0.7 + math.pi)

        # Publish the joint state
        self.joint_publisher.publish(msg)

        # Broadcast transforms
        self.broadcast_transforms()

    def broadcast_transforms(self):
        # In a real implementation, you would calculate the actual transforms
        # based on forward kinematics. For this example, we'll just broadcast
        # placeholder transforms.

        # This is where you would use the URDF to calculate transforms
        # using forward kinematics libraries like KDL or PyKDL
        pass

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# URDF parsing and validation example
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

@dataclass
class Link:
    name: str
    mass: float
    visual_geometry: Dict
    collision_geometry: Dict
    inertia: Dict

@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: List[float]
    origin_rpy: List[float]
    axis: List[float]
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None
    limit_effort: Optional[float] = None
    limit_velocity: Optional[float] = None

class URDFParser:
    def __init__(self, urdf_file_path: str):
        self.urdf_file_path = urdf_file_path
        self.links: Dict[str, Link] = {}
        self.joints: Dict[str, Joint] = {}
        self.materials: Dict[str, Dict] = {}

    def parse(self):
        """Parse the URDF file and extract links and joints"""
        tree = ET.parse(self.urdf_file_path)
        root = tree.getroot()

        # Parse materials
        for material_elem in root.findall('material'):
            name = material_elem.get('name')
            color_elem = material_elem.find('color')
            if color_elem is not None:
                rgba = [float(x) for x in color_elem.get('rgba', '1.0 1.0 1.0 1.0').split()]
                self.materials[name] = {'color': rgba}

        # Parse links
        for link_elem in root.findall('link'):
            name = link_elem.get('name')

            # Parse visual and collision geometry
            visual_elem = link_elem.find('visual')
            visual_geometry = {}
            if visual_elem is not None:
                geometry_elem = visual_elem.find('geometry')
                if geometry_elem is not None:
                    if geometry_elem.find('box') is not None:
                        size = [float(x) for x in geometry_elem.find('box').get('size', '1.0 1.0 1.0').split()]
                        visual_geometry = {'type': 'box', 'size': size}
                    elif geometry_elem.find('sphere') is not None:
                        radius = float(geometry_elem.find('sphere').get('radius', '1.0'))
                        visual_geometry = {'type': 'sphere', 'radius': radius}
                    elif geometry_elem.find('cylinder') is not None:
                        radius = float(geometry_elem.find('cylinder').get('radius', '1.0'))
                        length = float(geometry_elem.find('cylinder').get('length', '1.0'))
                        visual_geometry = {'type': 'cylinder', 'radius': radius, 'length': length}

            collision_elem = link_elem.find('collision')
            collision_geometry = {}
            if collision_elem is not None:
                geometry_elem = collision_elem.find('geometry')
                if geometry_elem is not None:
                    if geometry_elem.find('box') is not None:
                        size = [float(x) for x in geometry_elem.find('box').get('size', '1.0 1.0 1.0').split()]
                        collision_geometry = {'type': 'box', 'size': size}
                    elif geometry_elem.find('sphere') is not None:
                        radius = float(geometry_elem.find('sphere').get('radius', '1.0'))
                        collision_geometry = {'type': 'sphere', 'radius': radius}
                    elif geometry_elem.find('cylinder') is not None:
                        radius = float(geometry_elem.find('cylinder').get('radius', '1.0'))
                        length = float(geometry_elem.find('cylinder').get('length', '1.0'))
                        collision_geometry = {'type': 'cylinder', 'radius': radius, 'length': length}

            # Parse inertial properties
            inertial_elem = link_elem.find('inertial')
            mass = 0.0
            inertia = {}
            if inertial_elem is not None:
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value', '0.0'))

                inertia_elem = inertial_elem.find('inertia')
                if inertia_elem is not None:
                    inertia = {
                        'ixx': float(inertia_elem.get('ixx', '0.0')),
                        'ixy': float(inertia_elem.get('ixy', '0.0')),
                        'ixz': float(inertia_elem.get('ixz', '0.0')),
                        'iyy': float(inertia_elem.get('iyy', '0.0')),
                        'iyz': float(inertia_elem.get('iyz', '0.0')),
                        'izz': float(inertia_elem.get('izz', '0.0'))
                    }

            self.links[name] = Link(
                name=name,
                mass=mass,
                visual_geometry=visual_geometry,
                collision_geometry=collision_geometry,
                inertia=inertia
            )

        # Parse joints
        for joint_elem in root.findall('joint'):
            name = joint_elem.get('name')
            joint_type = joint_elem.get('type')

            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')

            if parent_elem is None or child_elem is None:
                continue

            parent = parent_elem.get('link')
            child = child_elem.get('link')

            # Parse origin (position and orientation)
            origin_elem = joint_elem.find('origin')
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]

            if origin_elem is not None:
                if origin_elem.get('xyz'):
                    xyz = [float(x) for x in origin_elem.get('xyz').split()]
                if origin_elem.get('rpy'):
                    rpy = [float(x) for x in origin_elem.get('rpy').split()]

            # Parse axis
            axis_elem = joint_elem.find('axis')
            axis = [0.0, 0.0, 1.0]  # Default to Z-axis
            if axis_elem is not None:
                axis = [float(x) for x in axis_elem.get('xyz', '0 0 1').split()]

            # Parse limits
            limit_elem = joint_elem.find('limit')
            limit_lower = None
            limit_upper = None
            limit_effort = None
            limit_velocity = None

            if limit_elem is not None:
                if limit_elem.get('lower'):
                    limit_lower = float(limit_elem.get('lower'))
                if limit_elem.get('upper'):
                    limit_upper = float(limit_elem.get('upper'))
                if limit_elem.get('effort'):
                    limit_effort = float(limit_elem.get('effort'))
                if limit_elem.get('velocity'):
                    limit_velocity = float(limit_elem.get('velocity'))

            self.joints[name] = Joint(
                name=name,
                type=joint_type,
                parent=parent,
                child=child,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                limit_effort=limit_effort,
                limit_velocity=limit_velocity
            )

    def validate_kinematic_chain(self, base_link: str = 'base_link') -> bool:
        """Validate that the URDF forms a proper kinematic chain"""
        # Check that all links are connected through joints
        all_links = set(self.links.keys())
        connected_links = {base_link}
        visited_joints = set()

        # Keep adding links that are connected through joints until no more can be added
        changed = True
        while changed:
            changed = False
            for joint_name, joint in self.joints.items():
                if joint_name in visited_joints:
                    continue

                # If parent is connected and child isn't, connect the child
                if joint.parent in connected_links and joint.child not in connected_links:
                    connected_links.add(joint.child)
                    visited_joints.add(joint_name)
                    changed = True
                # If child is connected and parent isn't, connect the parent
                elif joint.child in connected_links and joint.parent not in connected_links:
                    connected_links.add(joint.parent)
                    visited_joints.add(joint_name)
                    changed = True

        # Check if all links are connected
        unconnected_links = all_links - connected_links
        if unconnected_links:
            self.get_logger().error(f'Unconnected links: {unconnected_links}')
            return False

        return True

    def get_mass_properties(self) -> Dict:
        """Calculate total mass and center of mass of the robot"""
        total_mass = 0.0
        weighted_pos = [0.0, 0.0, 0.0]

        for link_name, link in self.links.items():
            total_mass += link.mass
            # For now, we're not calculating actual positions based on joint transforms
            # In a real implementation, you would calculate the actual positions

        return {
            'total_mass': total_mass,
            'center_of_mass': [0.0, 0.0, 0.0]  # Placeholder
        }

# Example usage of the URDF parser
def main():
    # Assuming the URDF file is saved as 'simple_humanoid.urdf'
    parser = URDFParser('simple_humanoid.urdf')
    parser.parse()

    print(f"Loaded {len(parser.links)} links and {len(parser.joints)} joints")

    # Validate the kinematic chain
    is_valid = parser.validate_kinematic_chain()
    print(f"Kinematic chain validation: {'PASSED' if is_valid else 'FAILED'}")

    # Get mass properties
    mass_props = parser.get_mass_properties()
    print(f"Total robot mass: {mass_props['total_mass']:.2f} kg")

if __name__ == '__main__':
    main()
```

## Summary

URDF (Unified Robot Description Format) is a critical component in ROS for describing robot models, especially for complex humanoid robots. It defines the robot's physical structure through links (body parts) and joints (connections), including their geometric, inertial, and visual properties.

For humanoid robots, URDF is particularly important because of the complex kinematic chains required to mimic human-like movement. A proper URDF file enables:
- Accurate simulation in environments like Gazebo
- Proper visualization in tools like RViz
- Correct kinematic and dynamic calculations
- Effective motion planning and control

The key components of a humanoid URDF include:
- Links that represent body parts (torso, head, limbs)
- Joints that connect the links with appropriate movement constraints
- Visual and collision geometry for simulation
- Inertial properties for physics calculations
- Materials for visualization

Understanding and properly structuring URDF files is essential for developing humanoid robots in ROS, as it forms the foundation for all simulation, visualization, and control operations.

## Exercises

1. **Basic Understanding**: Explain the difference between visual and collision geometry in URDF. Why might you want these to be different for a humanoid robot's link?

2. **Application Exercise**: Design a URDF for a simplified bipedal robot (just torso, head, and two legs). Include appropriate joint types and limits that would allow for walking motion. Draw a sketch of your robot's kinematic structure.

3. **Implementation Exercise**: Modify the URDF parsing code to calculate the center of mass of the robot based on the actual positions of the links. You'll need to implement forward kinematics to determine link positions based on joint angles.

4. **Challenge Exercise**: Create a URDF for a humanoid robot with additional sensors (cameras, IMU, force-torque sensors) attached at appropriate locations. Include the sensor specifications in the URDF using appropriate extensions like Gazebo plugins.