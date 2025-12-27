import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '3b7'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '048'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'e3b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', '245'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', 'd80'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '0e9'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '126'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '83f'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '4af'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '599'),
            routes: [
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', 'a8c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', '6ce'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/capstone/autonomous-humanoid',
                component: ComponentCreator('/docs/capstone/autonomous-humanoid', '4df'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/installation',
                component: ComponentCreator('/docs/installation', '001'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', 'aed'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/local-development',
                component: ComponentCreator('/docs/local-development', '4d9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1-the-robotic-nervous-system',
                component: ComponentCreator('/docs/module1-the-robotic-nervous-system', '477'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2-the-digital-twin',
                component: ComponentCreator('/docs/module2-the-digital-twin', 'c28'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3-the-ai-robot-brain',
                component: ComponentCreator('/docs/module3-the-ai-robot-brain', '185'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4-vision-language-action',
                component: ComponentCreator('/docs/module4-vision-language-action', '370'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module5-capstone-project-the-autonomous-humanoid',
                component: ComponentCreator('/docs/module5-capstone-project-the-autonomous-humanoid', 'b58'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nvidia-isaac/isaac-ros',
                component: ComponentCreator('/docs/nvidia-isaac/isaac-ros', '7ae'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nvidia-isaac/isaac-sim',
                component: ComponentCreator('/docs/nvidia-isaac/isaac-sim', 'ebc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nvidia-isaac/nav2',
                component: ComponentCreator('/docs/nvidia-isaac/nav2', 'e7b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nvidia-isaac/urdf-humanoids',
                component: ComponentCreator('/docs/nvidia-isaac/urdf-humanoids', 'ab7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai/embodied-intelligence',
                component: ComponentCreator('/docs/physical-ai/embodied-intelligence', 'bde'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai/sensors',
                component: ComponentCreator('/docs/physical-ai/sensors', 'b0e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/quickstart',
                component: ComponentCreator('/docs/quickstart', 'e30'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2/nodes-topics-services',
                component: ComponentCreator('/docs/ros2/nodes-topics-services', 'ccc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2/ros2-basics',
                component: ComponentCreator('/docs/ros2/ros2-basics', 'b7b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2/urdf-humanoids',
                component: ComponentCreator('/docs/ros2/urdf-humanoids', '5c8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/simulation/gazebo',
                component: ComponentCreator('/docs/simulation/gazebo', 'a68'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/simulation/unity',
                component: ComponentCreator('/docs/simulation/unity', 'daf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/upgrade',
                component: ComponentCreator('/docs/upgrade', 'a6e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vla/llm-planning',
                component: ComponentCreator('/docs/vla/llm-planning', 'fd5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vla/vision-language-action',
                component: ComponentCreator('/docs/vla/vision-language-action', '159'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vla/whisper-voice',
                component: ComponentCreator('/docs/vla/whisper-voice', '1b9'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/docs/personalized-advanced',
    component: ComponentCreator('/docs/personalized-advanced', '2f8'),
    routes: [
      {
        path: '/docs/personalized-advanced',
        component: ComponentCreator('/docs/personalized-advanced', '347'),
        routes: [
          {
            path: '/docs/personalized-advanced',
            component: ComponentCreator('/docs/personalized-advanced', '8de'),
            routes: [
              {
                path: '/docs/personalized-advanced/capstone/autonomous-humanoid-advanced',
                component: ComponentCreator('/docs/personalized-advanced/capstone/autonomous-humanoid-advanced', 'fbe'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/capstone/autonomous-humanoid-beginner',
                component: ComponentCreator('/docs/personalized-advanced/capstone/autonomous-humanoid-beginner', '185'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/capstone/autonomous-humanoid-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/capstone/autonomous-humanoid-intermediate', 'fd0'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/intro-advanced',
                component: ComponentCreator('/docs/personalized-advanced/intro-advanced', '0f8'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/module1-the-robotic-nervous-system-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module1-the-robotic-nervous-system-advanced', '421'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/module2-the-digital-twin-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module2-the-digital-twin-advanced', 'c8d'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/module3-the-ai-robot-brain-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module3-the-ai-robot-brain-advanced', 'a3c'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/module4-vision-language-action-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module4-vision-language-action-advanced', '5af'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/module5-capstone-project-the-autonomous-humanoid-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module5-capstone-project-the-autonomous-humanoid-advanced', 'adb'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/isaac-ros-advanced',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/isaac-ros-advanced', 'f73'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/isaac-ros-beginner',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/isaac-ros-beginner', '5bd'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/isaac-ros-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/isaac-ros-intermediate', '25b'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/isaac-sim-advanced',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/isaac-sim-advanced', '519'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/isaac-sim-beginner',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/isaac-sim-beginner', 'ec2'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/isaac-sim-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/isaac-sim-intermediate', '849'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/nav2-advanced',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/nav2-advanced', '889'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/nav2-beginner',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/nav2-beginner', '708'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/nav2-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/nav2-intermediate', 'a3f'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/urdf-humanoids-advanced',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/urdf-humanoids-advanced', '881'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/urdf-humanoids-beginner',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/urdf-humanoids-beginner', 'e27'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/urdf-humanoids-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/urdf-humanoids-intermediate', '8ac'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/physical-ai/embodied-intelligence-advanced',
                component: ComponentCreator('/docs/personalized-advanced/physical-ai/embodied-intelligence-advanced', '254'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/physical-ai/embodied-intelligence-beginner',
                component: ComponentCreator('/docs/personalized-advanced/physical-ai/embodied-intelligence-beginner', '07f'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/physical-ai/embodied-intelligence-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/physical-ai/embodied-intelligence-intermediate', 'b3b'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/physical-ai/sensors-advanced',
                component: ComponentCreator('/docs/personalized-advanced/physical-ai/sensors-advanced', '9cf'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/physical-ai/sensors-beginner',
                component: ComponentCreator('/docs/personalized-advanced/physical-ai/sensors-beginner', 'bc0'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/physical-ai/sensors-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/physical-ai/sensors-intermediate', '25b'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/nodes-topics-services-advanced',
                component: ComponentCreator('/docs/personalized-advanced/ros2/nodes-topics-services-advanced', '7ab'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/nodes-topics-services-beginner',
                component: ComponentCreator('/docs/personalized-advanced/ros2/nodes-topics-services-beginner', '585'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/nodes-topics-services-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/ros2/nodes-topics-services-intermediate', 'b15'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/ros2-basics-advanced',
                component: ComponentCreator('/docs/personalized-advanced/ros2/ros2-basics-advanced', '08c'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/ros2-basics-beginner',
                component: ComponentCreator('/docs/personalized-advanced/ros2/ros2-basics-beginner', '0a9'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/ros2-basics-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/ros2/ros2-basics-intermediate', 'f59'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/urdf-humanoids-advanced',
                component: ComponentCreator('/docs/personalized-advanced/ros2/urdf-humanoids-advanced', '091'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/urdf-humanoids-beginner',
                component: ComponentCreator('/docs/personalized-advanced/ros2/urdf-humanoids-beginner', 'f90'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/urdf-humanoids-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/ros2/urdf-humanoids-intermediate', 'df5'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/simulation/gazebo-advanced',
                component: ComponentCreator('/docs/personalized-advanced/simulation/gazebo-advanced', 'fe3'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/simulation/gazebo-beginner',
                component: ComponentCreator('/docs/personalized-advanced/simulation/gazebo-beginner', '172'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/simulation/gazebo-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/simulation/gazebo-intermediate', 'dac'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/simulation/unity-advanced',
                component: ComponentCreator('/docs/personalized-advanced/simulation/unity-advanced', '32a'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/simulation/unity-beginner',
                component: ComponentCreator('/docs/personalized-advanced/simulation/unity-beginner', '315'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/simulation/unity-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/simulation/unity-intermediate', '123'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/llm-planning-advanced',
                component: ComponentCreator('/docs/personalized-advanced/vla/llm-planning-advanced', 'e3f'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/llm-planning-beginner',
                component: ComponentCreator('/docs/personalized-advanced/vla/llm-planning-beginner', '714'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/llm-planning-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/vla/llm-planning-intermediate', '1f6'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/vision-language-action-advanced',
                component: ComponentCreator('/docs/personalized-advanced/vla/vision-language-action-advanced', '422'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/vision-language-action-beginner',
                component: ComponentCreator('/docs/personalized-advanced/vla/vision-language-action-beginner', 'ae0'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/vision-language-action-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/vla/vision-language-action-intermediate', '449'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/whisper-voice-advanced',
                component: ComponentCreator('/docs/personalized-advanced/vla/whisper-voice-advanced', '6b9'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/whisper-voice-beginner',
                component: ComponentCreator('/docs/personalized-advanced/vla/whisper-voice-beginner', 'ca4'),
                exact: true,
                sidebar: "advancedSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/whisper-voice-intermediate',
                component: ComponentCreator('/docs/personalized-advanced/vla/whisper-voice-intermediate', '830'),
                exact: true,
                sidebar: "advancedSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/docs/personalized-beginner',
    component: ComponentCreator('/docs/personalized-beginner', 'f31'),
    routes: [
      {
        path: '/docs/personalized-beginner',
        component: ComponentCreator('/docs/personalized-beginner', 'b22'),
        routes: [
          {
            path: '/docs/personalized-beginner',
            component: ComponentCreator('/docs/personalized-beginner', '82d'),
            routes: [
              {
                path: '/docs/personalized-beginner/capstone/autonomous-humanoid-advanced',
                component: ComponentCreator('/docs/personalized-beginner/capstone/autonomous-humanoid-advanced', '088'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/capstone/autonomous-humanoid-beginner',
                component: ComponentCreator('/docs/personalized-beginner/capstone/autonomous-humanoid-beginner', 'd8d'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/capstone/autonomous-humanoid-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/capstone/autonomous-humanoid-intermediate', 'e8c'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/intro-beginner',
                component: ComponentCreator('/docs/personalized-beginner/intro-beginner', 'e8d'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/module1-the-robotic-nervous-system-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module1-the-robotic-nervous-system-beginner', '8b5'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/module2-the-digital-twin-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module2-the-digital-twin-beginner', '2aa'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/module3-the-ai-robot-brain-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module3-the-ai-robot-brain-beginner', '749'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/module4-vision-language-action-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module4-vision-language-action-beginner', '87f'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/module5-capstone-project-the-autonomous-humanoid-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module5-capstone-project-the-autonomous-humanoid-beginner', 'dfb'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/isaac-ros-advanced',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/isaac-ros-advanced', '5ea'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/isaac-ros-beginner',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/isaac-ros-beginner', '166'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/isaac-ros-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/isaac-ros-intermediate', '29f'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/isaac-sim-advanced',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/isaac-sim-advanced', '02f'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/isaac-sim-beginner',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/isaac-sim-beginner', '948'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/isaac-sim-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/isaac-sim-intermediate', 'c65'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/nav2-advanced',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/nav2-advanced', '5eb'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/nav2-beginner',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/nav2-beginner', '4c7'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/nav2-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/nav2-intermediate', 'c69'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/urdf-humanoids-advanced',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/urdf-humanoids-advanced', 'cd0'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/urdf-humanoids-beginner',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/urdf-humanoids-beginner', '2c9'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/urdf-humanoids-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/urdf-humanoids-intermediate', '2ad'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/physical-ai/embodied-intelligence-advanced',
                component: ComponentCreator('/docs/personalized-beginner/physical-ai/embodied-intelligence-advanced', '392'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/physical-ai/embodied-intelligence-beginner',
                component: ComponentCreator('/docs/personalized-beginner/physical-ai/embodied-intelligence-beginner', '6ae'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/physical-ai/embodied-intelligence-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/physical-ai/embodied-intelligence-intermediate', '8d9'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/physical-ai/sensors-advanced',
                component: ComponentCreator('/docs/personalized-beginner/physical-ai/sensors-advanced', '26f'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/physical-ai/sensors-beginner',
                component: ComponentCreator('/docs/personalized-beginner/physical-ai/sensors-beginner', '996'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/physical-ai/sensors-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/physical-ai/sensors-intermediate', 'ce3'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/nodes-topics-services-advanced',
                component: ComponentCreator('/docs/personalized-beginner/ros2/nodes-topics-services-advanced', '960'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/nodes-topics-services-beginner',
                component: ComponentCreator('/docs/personalized-beginner/ros2/nodes-topics-services-beginner', '7ce'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/nodes-topics-services-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/ros2/nodes-topics-services-intermediate', 'f22'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/ros2-basics-advanced',
                component: ComponentCreator('/docs/personalized-beginner/ros2/ros2-basics-advanced', '402'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/ros2-basics-beginner',
                component: ComponentCreator('/docs/personalized-beginner/ros2/ros2-basics-beginner', '257'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/ros2-basics-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/ros2/ros2-basics-intermediate', 'fbb'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/urdf-humanoids-advanced',
                component: ComponentCreator('/docs/personalized-beginner/ros2/urdf-humanoids-advanced', '256'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/urdf-humanoids-beginner',
                component: ComponentCreator('/docs/personalized-beginner/ros2/urdf-humanoids-beginner', '56c'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/urdf-humanoids-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/ros2/urdf-humanoids-intermediate', 'eb3'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/simulation/gazebo-advanced',
                component: ComponentCreator('/docs/personalized-beginner/simulation/gazebo-advanced', '4d8'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/simulation/gazebo-beginner',
                component: ComponentCreator('/docs/personalized-beginner/simulation/gazebo-beginner', 'a9c'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/simulation/gazebo-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/simulation/gazebo-intermediate', 'ccf'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/simulation/unity-advanced',
                component: ComponentCreator('/docs/personalized-beginner/simulation/unity-advanced', '3b4'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/simulation/unity-beginner',
                component: ComponentCreator('/docs/personalized-beginner/simulation/unity-beginner', '56f'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/simulation/unity-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/simulation/unity-intermediate', '726'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/llm-planning-advanced',
                component: ComponentCreator('/docs/personalized-beginner/vla/llm-planning-advanced', '5fa'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/llm-planning-beginner',
                component: ComponentCreator('/docs/personalized-beginner/vla/llm-planning-beginner', 'a07'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/llm-planning-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/vla/llm-planning-intermediate', '838'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/vision-language-action-advanced',
                component: ComponentCreator('/docs/personalized-beginner/vla/vision-language-action-advanced', '923'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/vision-language-action-beginner',
                component: ComponentCreator('/docs/personalized-beginner/vla/vision-language-action-beginner', 'a44'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/vision-language-action-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/vla/vision-language-action-intermediate', '207'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/whisper-voice-advanced',
                component: ComponentCreator('/docs/personalized-beginner/vla/whisper-voice-advanced', '19d'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/whisper-voice-beginner',
                component: ComponentCreator('/docs/personalized-beginner/vla/whisper-voice-beginner', '66b'),
                exact: true,
                sidebar: "beginnerSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/whisper-voice-intermediate',
                component: ComponentCreator('/docs/personalized-beginner/vla/whisper-voice-intermediate', '1ab'),
                exact: true,
                sidebar: "beginnerSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/docs/personalized-intermediate',
    component: ComponentCreator('/docs/personalized-intermediate', '1ac'),
    routes: [
      {
        path: '/docs/personalized-intermediate',
        component: ComponentCreator('/docs/personalized-intermediate', 'b4e'),
        routes: [
          {
            path: '/docs/personalized-intermediate',
            component: ComponentCreator('/docs/personalized-intermediate', '7db'),
            routes: [
              {
                path: '/docs/personalized-intermediate/capstone/autonomous-humanoid-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/capstone/autonomous-humanoid-advanced', '720'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/capstone/autonomous-humanoid-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/capstone/autonomous-humanoid-beginner', '6ba'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/capstone/autonomous-humanoid-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/capstone/autonomous-humanoid-intermediate', '209'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/intro-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/intro-intermediate', '6a9'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module1-the-robotic-nervous-system-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module1-the-robotic-nervous-system-intermediate', '999'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module2-the-digital-twin-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module2-the-digital-twin-intermediate', '05f'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module3-the-ai-robot-brain-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module3-the-ai-robot-brain-intermediate', '73e'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module4-vision-language-action-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module4-vision-language-action-intermediate', '289'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module5-capstone-project-the-autonomous-humanoid-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module5-capstone-project-the-autonomous-humanoid-intermediate', '08f'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/isaac-ros-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/isaac-ros-advanced', 'a71'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/isaac-ros-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/isaac-ros-beginner', 'b40'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/isaac-ros-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/isaac-ros-intermediate', 'c7b'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/isaac-sim-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/isaac-sim-advanced', 'ef4'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/isaac-sim-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/isaac-sim-beginner', 'e15'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/isaac-sim-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/isaac-sim-intermediate', '1b4'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/nav2-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/nav2-advanced', '006'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/nav2-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/nav2-beginner', '9ea'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/nav2-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/nav2-intermediate', 'fb4'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/urdf-humanoids-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/urdf-humanoids-advanced', '25f'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/urdf-humanoids-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/urdf-humanoids-beginner', '563'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/urdf-humanoids-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/urdf-humanoids-intermediate', 'd05'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/physical-ai/embodied-intelligence-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/physical-ai/embodied-intelligence-advanced', 'ab2'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/physical-ai/embodied-intelligence-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/physical-ai/embodied-intelligence-beginner', '780'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/physical-ai/embodied-intelligence-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/physical-ai/embodied-intelligence-intermediate', 'a2c'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/physical-ai/sensors-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/physical-ai/sensors-advanced', '7db'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/physical-ai/sensors-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/physical-ai/sensors-beginner', '5f4'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/physical-ai/sensors-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/physical-ai/sensors-intermediate', '480'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/nodes-topics-services-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/nodes-topics-services-advanced', '5e8'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/nodes-topics-services-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/nodes-topics-services-beginner', '10f'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/nodes-topics-services-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/nodes-topics-services-intermediate', '190'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/ros2-basics-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/ros2-basics-advanced', 'd49'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/ros2-basics-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/ros2-basics-beginner', '84c'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/ros2-basics-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/ros2-basics-intermediate', '2a9'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/urdf-humanoids-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/urdf-humanoids-advanced', 'a37'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/urdf-humanoids-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/urdf-humanoids-beginner', '51a'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/urdf-humanoids-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/urdf-humanoids-intermediate', 'cf2'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/simulation/gazebo-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/simulation/gazebo-advanced', 'ce4'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/simulation/gazebo-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/simulation/gazebo-beginner', '727'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/simulation/gazebo-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/simulation/gazebo-intermediate', '56c'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/simulation/unity-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/simulation/unity-advanced', 'd22'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/simulation/unity-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/simulation/unity-beginner', '5a5'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/simulation/unity-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/simulation/unity-intermediate', '5da'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/llm-planning-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/vla/llm-planning-advanced', 'b12'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/llm-planning-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/vla/llm-planning-beginner', 'ed0'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/llm-planning-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/vla/llm-planning-intermediate', 'fe9'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/vision-language-action-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/vla/vision-language-action-advanced', '178'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/vision-language-action-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/vla/vision-language-action-beginner', '407'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/vision-language-action-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/vla/vision-language-action-intermediate', 'bbc'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/whisper-voice-advanced',
                component: ComponentCreator('/docs/personalized-intermediate/vla/whisper-voice-advanced', 'a6c'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/whisper-voice-beginner',
                component: ComponentCreator('/docs/personalized-intermediate/vla/whisper-voice-beginner', 'efd'),
                exact: true,
                sidebar: "intermediateSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/whisper-voice-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/vla/whisper-voice-intermediate', '2b8'),
                exact: true,
                sidebar: "intermediateSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/docs/urdu',
    component: ComponentCreator('/docs/urdu', '0ef'),
    routes: [
      {
        path: '/docs/urdu',
        component: ComponentCreator('/docs/urdu', 'c43'),
        routes: [
          {
            path: '/docs/urdu',
            component: ComponentCreator('/docs/urdu', '66e'),
            routes: [
              {
                path: '/docs/urdu/capstone/autonomous-humanoid',
                component: ComponentCreator('/docs/urdu/capstone/autonomous-humanoid', 'e17'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/intro',
                component: ComponentCreator('/docs/urdu/intro', '28d'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/module1-the-robotic-nervous-system',
                component: ComponentCreator('/docs/urdu/module1-the-robotic-nervous-system', '106'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/module2-the-digital-twin',
                component: ComponentCreator('/docs/urdu/module2-the-digital-twin', '7e3'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/module3-the-ai-robot-brain',
                component: ComponentCreator('/docs/urdu/module3-the-ai-robot-brain', '88a'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/module4-vision-language-action',
                component: ComponentCreator('/docs/urdu/module4-vision-language-action', '76f'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/module5-capstone-project-the-autonomous-humanoid',
                component: ComponentCreator('/docs/urdu/module5-capstone-project-the-autonomous-humanoid', '097'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/nvidia-isaac/isaac-ros',
                component: ComponentCreator('/docs/urdu/nvidia-isaac/isaac-ros', '38f'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/nvidia-isaac/isaac-sim',
                component: ComponentCreator('/docs/urdu/nvidia-isaac/isaac-sim', 'fbe'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/nvidia-isaac/nav2',
                component: ComponentCreator('/docs/urdu/nvidia-isaac/nav2', 'e34'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/nvidia-isaac/urdf-humanoids',
                component: ComponentCreator('/docs/urdu/nvidia-isaac/urdf-humanoids', '9c9'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/physical-ai/embodied-intelligence',
                component: ComponentCreator('/docs/urdu/physical-ai/embodied-intelligence', '194'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/physical-ai/sensors',
                component: ComponentCreator('/docs/urdu/physical-ai/sensors', '30a'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/ros2/nodes-topics-services',
                component: ComponentCreator('/docs/urdu/ros2/nodes-topics-services', '8da'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/ros2/ros2-basics',
                component: ComponentCreator('/docs/urdu/ros2/ros2-basics', 'a0e'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/ros2/urdf-humanoids',
                component: ComponentCreator('/docs/urdu/ros2/urdf-humanoids', 'cdf'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/simulation/gazebo',
                component: ComponentCreator('/docs/urdu/simulation/gazebo', 'e2b'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/simulation/unity',
                component: ComponentCreator('/docs/urdu/simulation/unity', '465'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/vla/llm-planning',
                component: ComponentCreator('/docs/urdu/vla/llm-planning', '516'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/vla/vision-language-action',
                component: ComponentCreator('/docs/urdu/vla/vision-language-action', '61f'),
                exact: true,
                sidebar: "urduSidebar"
              },
              {
                path: '/docs/urdu/vla/whisper-voice',
                component: ComponentCreator('/docs/urdu/vla/whisper-voice', '5fb'),
                exact: true,
                sidebar: "urduSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '954'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
