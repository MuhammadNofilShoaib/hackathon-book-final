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
    component: ComponentCreator('/docs/personalized-advanced', 'fa6'),
    routes: [
      {
        path: '/docs/personalized-advanced',
        component: ComponentCreator('/docs/personalized-advanced', '2e0'),
        routes: [
          {
            path: '/docs/personalized-advanced',
            component: ComponentCreator('/docs/personalized-advanced', 'f51'),
            routes: [
              {
                path: '/docs/personalized-advanced/capstone/autonomous-humanoid-advanced',
                component: ComponentCreator('/docs/personalized-advanced/capstone/autonomous-humanoid-advanced', 'e9a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/intro-advanced',
                component: ComponentCreator('/docs/personalized-advanced/intro-advanced', '97d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/module1-the-robotic-nervous-system-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module1-the-robotic-nervous-system-advanced', '0c5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/module2-the-digital-twin-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module2-the-digital-twin-advanced', '790'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/module3-the-ai-robot-brain-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module3-the-ai-robot-brain-advanced', '75a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/module4-vision-language-action-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module4-vision-language-action-advanced', '001'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/module5-capstone-project-the-autonomous-humanoid-advanced',
                component: ComponentCreator('/docs/personalized-advanced/module5-capstone-project-the-autonomous-humanoid-advanced', '99e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/isaac-ros-advanced',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/isaac-ros-advanced', '802'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/isaac-sim-advanced',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/isaac-sim-advanced', 'fad'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/nav2-advanced',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/nav2-advanced', '112'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/nvidia-isaac/urdf-humanoids-advanced',
                component: ComponentCreator('/docs/personalized-advanced/nvidia-isaac/urdf-humanoids-advanced', '2c0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/physical-ai/embodied-intelligence-advanced',
                component: ComponentCreator('/docs/personalized-advanced/physical-ai/embodied-intelligence-advanced', '0a1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/physical-ai/sensors-advanced',
                component: ComponentCreator('/docs/personalized-advanced/physical-ai/sensors-advanced', '9a7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/nodes-topics-services-advanced',
                component: ComponentCreator('/docs/personalized-advanced/ros2/nodes-topics-services-advanced', '717'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/ros2-basics-advanced',
                component: ComponentCreator('/docs/personalized-advanced/ros2/ros2-basics-advanced', 'af8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/ros2/urdf-humanoids-advanced',
                component: ComponentCreator('/docs/personalized-advanced/ros2/urdf-humanoids-advanced', '85e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/simulation/gazebo-advanced',
                component: ComponentCreator('/docs/personalized-advanced/simulation/gazebo-advanced', '6e9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/simulation/unity-advanced',
                component: ComponentCreator('/docs/personalized-advanced/simulation/unity-advanced', 'fb4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/llm-planning-advanced',
                component: ComponentCreator('/docs/personalized-advanced/vla/llm-planning-advanced', 'd45'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/vision-language-action-advanced',
                component: ComponentCreator('/docs/personalized-advanced/vla/vision-language-action-advanced', 'd30'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-advanced/vla/whisper-voice-advanced',
                component: ComponentCreator('/docs/personalized-advanced/vla/whisper-voice-advanced', 'dba'),
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
    path: '/docs/personalized-beginner',
    component: ComponentCreator('/docs/personalized-beginner', 'bc4'),
    routes: [
      {
        path: '/docs/personalized-beginner',
        component: ComponentCreator('/docs/personalized-beginner', '95d'),
        routes: [
          {
            path: '/docs/personalized-beginner',
            component: ComponentCreator('/docs/personalized-beginner', '702'),
            routes: [
              {
                path: '/docs/personalized-beginner/capstone/autonomous-humanoid-beginner',
                component: ComponentCreator('/docs/personalized-beginner/capstone/autonomous-humanoid-beginner', '468'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/intro-beginner',
                component: ComponentCreator('/docs/personalized-beginner/intro-beginner', '455'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/module1-the-robotic-nervous-system-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module1-the-robotic-nervous-system-beginner', '146'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/module2-the-digital-twin-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module2-the-digital-twin-beginner', 'dbd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/module3-the-ai-robot-brain-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module3-the-ai-robot-brain-beginner', '835'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/module4-vision-language-action-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module4-vision-language-action-beginner', '275'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/module5-capstone-project-the-autonomous-humanoid-beginner',
                component: ComponentCreator('/docs/personalized-beginner/module5-capstone-project-the-autonomous-humanoid-beginner', '90d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/isaac-ros-beginner',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/isaac-ros-beginner', 'a99'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/isaac-sim-beginner',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/isaac-sim-beginner', '18c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/nav2-beginner',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/nav2-beginner', '7be'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/nvidia-isaac/urdf-humanoids-beginner',
                component: ComponentCreator('/docs/personalized-beginner/nvidia-isaac/urdf-humanoids-beginner', '6dd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/physical-ai/embodied-intelligence-beginner',
                component: ComponentCreator('/docs/personalized-beginner/physical-ai/embodied-intelligence-beginner', 'da1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/physical-ai/sensors-beginner',
                component: ComponentCreator('/docs/personalized-beginner/physical-ai/sensors-beginner', 'c0a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/nodes-topics-services-beginner',
                component: ComponentCreator('/docs/personalized-beginner/ros2/nodes-topics-services-beginner', '16e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/ros2-basics-beginner',
                component: ComponentCreator('/docs/personalized-beginner/ros2/ros2-basics-beginner', 'b85'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/ros2/urdf-humanoids-beginner',
                component: ComponentCreator('/docs/personalized-beginner/ros2/urdf-humanoids-beginner', '2a0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/simulation/gazebo-beginner',
                component: ComponentCreator('/docs/personalized-beginner/simulation/gazebo-beginner', '7fb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/simulation/unity-beginner',
                component: ComponentCreator('/docs/personalized-beginner/simulation/unity-beginner', '08f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/llm-planning-beginner',
                component: ComponentCreator('/docs/personalized-beginner/vla/llm-planning-beginner', '070'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/vision-language-action-beginner',
                component: ComponentCreator('/docs/personalized-beginner/vla/vision-language-action-beginner', '916'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-beginner/vla/whisper-voice-beginner',
                component: ComponentCreator('/docs/personalized-beginner/vla/whisper-voice-beginner', 'aa5'),
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
    path: '/docs/personalized-intermediate',
    component: ComponentCreator('/docs/personalized-intermediate', 'ac8'),
    routes: [
      {
        path: '/docs/personalized-intermediate',
        component: ComponentCreator('/docs/personalized-intermediate', '6a1'),
        routes: [
          {
            path: '/docs/personalized-intermediate',
            component: ComponentCreator('/docs/personalized-intermediate', 'b37'),
            routes: [
              {
                path: '/docs/personalized-intermediate/capstone/autonomous-humanoid-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/capstone/autonomous-humanoid-intermediate', '771'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/intro-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/intro-intermediate', 'b22'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module1-the-robotic-nervous-system-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module1-the-robotic-nervous-system-intermediate', '78a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module2-the-digital-twin-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module2-the-digital-twin-intermediate', '1cd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module3-the-ai-robot-brain-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module3-the-ai-robot-brain-intermediate', '74d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module4-vision-language-action-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module4-vision-language-action-intermediate', '082'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/module5-capstone-project-the-autonomous-humanoid-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/module5-capstone-project-the-autonomous-humanoid-intermediate', '4c0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/isaac-ros-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/isaac-ros-intermediate', '684'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/isaac-sim-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/isaac-sim-intermediate', '8d2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/nav2-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/nav2-intermediate', '579'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/nvidia-isaac/urdf-humanoids-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/nvidia-isaac/urdf-humanoids-intermediate', '77c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/physical-ai/embodied-intelligence-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/physical-ai/embodied-intelligence-intermediate', '3a3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/physical-ai/sensors-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/physical-ai/sensors-intermediate', 'd21'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/nodes-topics-services-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/nodes-topics-services-intermediate', '400'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/ros2-basics-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/ros2-basics-intermediate', '04b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/ros2/urdf-humanoids-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/ros2/urdf-humanoids-intermediate', 'ec1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/simulation/gazebo-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/simulation/gazebo-intermediate', 'c73'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/simulation/unity-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/simulation/unity-intermediate', '74f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/llm-planning-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/vla/llm-planning-intermediate', 'f44'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/vision-language-action-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/vla/vision-language-action-intermediate', '32c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalized-intermediate/vla/whisper-voice-intermediate',
                component: ComponentCreator('/docs/personalized-intermediate/vla/whisper-voice-intermediate', '5c4'),
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
    path: '/docs/urdu',
    component: ComponentCreator('/docs/urdu', '51a'),
    routes: [
      {
        path: '/docs/urdu',
        component: ComponentCreator('/docs/urdu', '9cc'),
        routes: [
          {
            path: '/docs/urdu',
            component: ComponentCreator('/docs/urdu', 'e4d'),
            routes: [
              {
                path: '/docs/urdu/capstone/autonomous-humanoid',
                component: ComponentCreator('/docs/urdu/capstone/autonomous-humanoid', '56c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/intro',
                component: ComponentCreator('/docs/urdu/intro', '9e9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/module1-the-robotic-nervous-system',
                component: ComponentCreator('/docs/urdu/module1-the-robotic-nervous-system', 'bdc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/module2-the-digital-twin',
                component: ComponentCreator('/docs/urdu/module2-the-digital-twin', 'c98'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/module3-the-ai-robot-brain',
                component: ComponentCreator('/docs/urdu/module3-the-ai-robot-brain', 'b55'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/module4-vision-language-action',
                component: ComponentCreator('/docs/urdu/module4-vision-language-action', 'c1f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/module5-capstone-project-the-autonomous-humanoid',
                component: ComponentCreator('/docs/urdu/module5-capstone-project-the-autonomous-humanoid', '948'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/nvidia-isaac/isaac-ros',
                component: ComponentCreator('/docs/urdu/nvidia-isaac/isaac-ros', '2c0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/nvidia-isaac/isaac-sim',
                component: ComponentCreator('/docs/urdu/nvidia-isaac/isaac-sim', '633'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/nvidia-isaac/nav2',
                component: ComponentCreator('/docs/urdu/nvidia-isaac/nav2', '1ac'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/nvidia-isaac/urdf-humanoids',
                component: ComponentCreator('/docs/urdu/nvidia-isaac/urdf-humanoids', '762'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/physical-ai/embodied-intelligence',
                component: ComponentCreator('/docs/urdu/physical-ai/embodied-intelligence', '92d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/physical-ai/sensors',
                component: ComponentCreator('/docs/urdu/physical-ai/sensors', 'bc0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/ros2/nodes-topics-services',
                component: ComponentCreator('/docs/urdu/ros2/nodes-topics-services', '789'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/ros2/ros2-basics',
                component: ComponentCreator('/docs/urdu/ros2/ros2-basics', 'acb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/ros2/urdf-humanoids',
                component: ComponentCreator('/docs/urdu/ros2/urdf-humanoids', '8ff'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/simulation/gazebo',
                component: ComponentCreator('/docs/urdu/simulation/gazebo', '62e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/simulation/unity',
                component: ComponentCreator('/docs/urdu/simulation/unity', '466'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/vla/llm-planning',
                component: ComponentCreator('/docs/urdu/vla/llm-planning', 'fe3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/vla/vision-language-action',
                component: ComponentCreator('/docs/urdu/vla/vision-language-action', 'c50'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/urdu/vla/whisper-voice',
                component: ComponentCreator('/docs/urdu/vla/whisper-voice', 'e3b'),
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
    path: '/',
    component: ComponentCreator('/', '954'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
