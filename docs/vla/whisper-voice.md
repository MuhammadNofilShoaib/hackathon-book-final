# Voice-to-Action Systems Using Speech Recognition

## Concept

Voice-to-Action systems represent a critical component of natural human-robot interaction, enabling robots to understand spoken commands and translate them directly into physical actions. Think of these systems as the "ears" and "interpreter" of a robot, allowing it to listen to human instructions and respond appropriately through physical behavior.

In robotics, voice-to-action systems bridge the gap between human natural language and robot control. Traditional robotic interfaces required users to input commands through specialized interfaces or programming languages. Voice-to-action systems allow for more intuitive interaction, where users can simply speak to robots in natural language, and the robots can understand and execute the requested actions.

Voice-to-action systems matter in Physical AI because they enable seamless human-robot collaboration in environments where humans and robots work together. For humanoid robots designed to operate in human-centric spaces, voice interaction is often the most natural and efficient communication method. These systems can process spoken commands in real-time, accounting for ambient noise, multiple speakers, and varied accents.

If you're familiar with voice assistants like Siri, Alexa, or Google Assistant, voice-to-action systems for robotics extend this concept by adding physical action execution. While traditional voice assistants can control smart home devices, voice-to-action systems for robots can control complex physical behaviors like navigation, manipulation, and interaction with the environment.

## ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VOICE-TO-ACTION SYSTEM ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌──────────────────────────────────────────┐   │
│  │   HUMAN         │    │                                          │   │
│  │   SPEAKER       │    │  ┌────────────────────────────────────┐  │   │
│  │                 │───▶│  │    VOICE-TO-ACTION PIPELINE      │  │   │
│  │  "Pick up the  │    │  │  ┌──────────────────────────────┐  │  │   │
│  │   red cup from │    │  │  │    SPEECH RECOGNITION        │  │  │   │
│  │   the table"   │    │  │  │                              │  │  │   │
│  │                 │    │  │  │  • Audio Preprocessing      │  │  │   │
│  │                 │    │  │  │  • Feature Extraction       │  │  │   │
│  │                 │    │  │  │  • ASR Model (Whisper, etc.)│  │  │   │
│  └─────────────────┘    │  │  │  • Text Output             │  │  │   │
│                         │  │  └──────────────────────────────┘  │  │   │
│                         │  │                                    │  │   │
│  ┌─────────────────┐    │  │  ┌──────────────────────────────┐  │  │   │
│  │   AUDIO         │    │  │  │    NATURAL LANGUAGE        │  │  │   │
│  │   INPUT         │    │  │  │    UNDERSTANDING           │  │  │   │
│  │                 │───▶│  │  │                              │  │  │   │
│  │  [Microphone   │    │  │  │  • Intent Classification    │  │  │   │
│  │   Stream]      │    │  │  │  • Entity Extraction       │  │  │   │
│  │                 │    │  │  │  • Command Parsing        │  │  │   │
│  └─────────────────┘    │  │  │  • Context Understanding   │  │  │   │
│                         │  │  └──────────────────────────────┘  │  │   │
│                         │  │                                    │  │   │
│                         │  │  ┌──────────────────────────────┐  │  │   │
│                         │  │  │    ACTION MAPPING          │  │  │   │
│                         │  │  │                              │  │  │   │
│                         │  │  │  • Command to Action       │  │  │   │
│                         │  │  │  • Parameter Extraction    │  │  │   │
│                         │  │  │  • Robot Capability        │  │  │   │
│                         │  │  │    Verification            │  │  │   │
│                         │  │  └──────────────────────────────┘  │  │   │
│                         │  │                                    │  │   │
│                         │  │  ┌──────────────────────────────┐  │  │   │
│                         │  │  │    ACTION EXECUTION         │  │  │   │
│                         │  │  │                              │  │  │   │
│                         │  │  │  • Path Planning            │  │  │   │
│                         │  │  │  • Motion Planning         │  │  │   │
│                         │  │  │  • Control Generation      │  │  │   │
│                         │  │  │  • Safety Verification     │  │  │   │
│                         │  │  └──────────────────────────────┘  │  │   │
│                         │  └──────────────────────────────────────────┘  │
│                         │         │                                   │   │
│                         │         ▼                                   │   │
│  ┌─────────────────┐    │  ┌──────────────────────────────────────────┐│   │
│  │   PHYSICAL      │    │  │         ROBOT EXECUTION                ││   │
│  │   ROBOT         │    │  │                                          ││   │
│  │                 │    │  │  ┌────────────────────────────────────┐  ││   │
│  │  • Manipulation │    │  │  │    EXECUTION PLANNING            │  ││   │
│  │  • Navigation  │    │  │  │                                    │  ││   │
│  │  • Locomotion  │    │  │  │  • Trajectory Generation          │  ││   │
│  │  • Interaction │    │  │  │  • Control Law Application        │  ││   │
│  │                 │    │  │  │  • Safety Verification           │  ││   │
│  │                 │    │  │  └────────────────────────────────────┘  ││   │
│  └─────────────────┘    │  │                                          ││   │
│         │                │  │  ┌────────────────────────────────────┐  ││   │
│         ▼                │  │  │    PHYSICAL EXECUTION             │  ││   │
│  ┌─────────────────┐    │  │  │                                    │  ││   │
│  │   ACTION        │    │  │  │  • Motor Control                  │  ││   │
│  │   EXECUTION     │    │  │  │  • Sensor Feedback               │  ││   │
│  │                 │    │  │  │  • Closed-loop Control           │  ││   │
│  │  • Grasping     │    │  │  │  • Compliance Control           │  ││   │
│  │  • Moving       │    │  │  │  • Balance Maintenance          │  ││   │
│  │  • Manipulating │    │  │  └────────────────────────────────────┘  ││   │
│  └─────────────────┘    │  └──────────────────────────────────────────┘│   │
│                         └──────────────────────────────────────────────────────┘
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                SPEECH RECOGNITION PROCESSING PIPELINE                   │
│                                                                         │
│  INPUT: Raw Audio ──▶ Preprocessing ──▶ Feature Extraction ──▶ ASR    │
│      │                    │                    │                    │   │
│      ▼                    ▼                    ▼                    ▼   │
│  [Raw Audio]      [Noise Reduction]    [MFCC/Filter Banks]    [Text]  │
│      │                    │                    │                    │   │
│      └────────────────────┼────────────────────┼────────────────────┘   │
│                           │                    │                        │
│                           ▼                    ▼                        │
│  NATURAL LANGUAGE    ┌─────────────┐    ┌─────────────────┐            │
│  UNDERSTANDING       │ Intent      │    │ Entity          │            │
│                      │ Classification│    │ Extraction      │            │
│                      └─────────────┘    └─────────────────┘            │
│                           │                    │                        │
│                           ▼                    ▼                        │
│  ACTION MAPPING    ┌─────────────────┐    ┌─────────────────┐          │
│                    │ Command         │    │ Parameters      │          │
│                    │ Mapping         │    │ Extraction      │          │
│                    └─────────────────┘    └─────────────────┘          │
│                           │                    │                        │
│                           └────────────────────┼────────────────────────┘
│                                                ▼
│  ACTION EXECUTION                     ┌─────────────────┐
│                                       │ Robot Action    │
│                                       │ Execution       │
│                                       └─────────────────┘
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                WHISPER INTEGRATION FOR ROBOTICS                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  WHISPER ASR MODEL                                            │   │
│  │  (OpenAI Whisper or similar)                                  │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  Encoder:                                               │  │   │
│  │  │  • Multi-layer Transformer                              │  │   │
│  │  │  • Audio Feature Processing                             │  │   │
│  │  │  • Spectrogram Analysis                                 │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  Decoder:                                               │  │   │
│  │  │  • Text Generation                                      │  │   │
│  │  │  • Language Modeling                                    │  │   │
│  │  │  • Context Understanding                                │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CUSTOM ROBOTICS LAYER                                        │   │
│  │  (Built on top of Whisper)                                    │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  Command Classifier                                     │  │   │
│  │  │  • Maps general text to robot commands                 │  │   │
│  │  │  • Context-aware interpretation                        │  │   │
│  │  │  • Intent detection for robotics                       │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  Action Generator                                       │  │   │
│  │  │  • Converts commands to robot actions                  │  │   │
│  │  │  • Parameter extraction for robot control              │  │   │
│  │  │  • Safety constraint verification                      │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This diagram illustrates the Voice-to-Action system architecture, showing how spoken commands are processed through speech recognition, natural language understanding, action mapping, and finally robot execution.

## Real-world Analogy

Think of voice-to-action systems like a highly skilled interpreter at an international conference who not only translates spoken language but also executes the translated instructions physically. Just as an interpreter listens to a speaker in one language, understands the meaning, and conveys it in another language, a voice-to-action system listens to human speech, understands the intent, and executes appropriate physical actions.

A human interpreter needs to:
- Listen carefully to the original speaker
- Understand the meaning and context of the message
- Translate the message accurately to the target language
- Ensure the translation preserves the original intent

Similarly, a voice-to-action system:
- Captures and processes audio input from the environment
- Recognizes speech and converts it to text
- Understands the intent and extracts relevant information
- Maps the understood command to appropriate robot actions

Just as a skilled interpreter can handle complex instructions and adapt to different speaking styles, a well-designed voice-to-action system can process varied commands and adapt to different users' speaking patterns, accents, and environmental conditions.

The difference is that while human interpreters translate between languages, voice-to-action systems translate between human language and robot behavior, requiring an additional layer of action planning and execution.

## Pseudo-code (Voice-to-Action / Robotics style)

```python
# Voice-to-Action System using Speech Recognition
import torch
import torch.nn as nn
import whisper
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import sounddevice as sd
import librosa
from typing import Dict, List, Tuple, Optional
import asyncio
import threading
import queue
import time
from dataclasses import dataclass

@dataclass
class RobotCommand:
    """Data class for robot commands"""
    action_type: str  # 'navigation', 'manipulation', 'interaction', etc.
    parameters: Dict[str, any]
    confidence: float
    raw_text: str

class SpeechRecognizer:
    """Speech recognition component using Whisper or similar models"""

    def __init__(self, model_name="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Load Whisper model
        try:
            self.model = whisper.load_model(model_name).to(device)
        except:
            # Fallback to a simpler approach if Whisper is not available
            self.model = None
            print("Whisper not available, using mock speech recognition")

        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = 1.0  # seconds
        self.min_decibels = -40  # Minimum audio level to process

    def record_audio(self, duration=5.0):
        """Record audio from microphone"""
        print(f"Recording audio for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete
        return audio.flatten()

    def preprocess_audio(self, audio):
        """Preprocess audio for speech recognition"""
        # Normalize audio
        audio = audio / np.max(np.abs(audio))

        # Apply noise reduction if needed
        # This is a simple noise gate - in practice, more sophisticated methods would be used
        threshold = 10 ** (self.min_decibels / 20.0)
        if np.max(np.abs(audio)) < threshold:
            return None  # Audio too quiet

        return audio

    def transcribe_audio(self, audio):
        """Transcribe audio to text using Whisper"""
        if self.model is None:
            # Mock transcription for demonstration
            return "mock command to robot"

        # Preprocess audio
        processed_audio = self.preprocess_audio(audio)
        if processed_audio is None:
            return ""

        # Transcribe using Whisper
        result = self.model.transcribe(processed_audio)
        return result["text"]

    def transcribe_from_microphone(self, timeout=10.0):
        """Record and transcribe audio from microphone in real-time"""
        print("Listening for voice command...")

        # Record audio
        audio = self.record_audio(duration=timeout)

        # Transcribe
        text = self.transcribe_audio(audio)

        return text.strip()

class NaturalLanguageUnderstanding:
    """Natural Language Understanding component for command interpretation"""

    def __init__(self):
        # Initialize intent classification model
        # In practice, this could be a custom trained model for robotics commands
        self.intent_classifier = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium"  # This is a placeholder - in practice, use a robotics-specific model
        )

        # Define robot command vocabulary
        self.robot_commands = {
            'navigation': [
                'go to', 'move to', 'navigate to', 'walk to', 'go', 'move', 'navigate',
                'come here', 'follow me', 'stop', 'turn left', 'turn right', 'go forward', 'go back'
            ],
            'manipulation': [
                'pick up', 'grasp', 'grab', 'take', 'lift', 'hold', 'release', 'put down',
                'place', 'move', 'pick', 'get', 'bring', 'fetch'
            ],
            'interaction': [
                'greet', 'hello', 'hi', 'goodbye', 'bye', 'introduce yourself',
                'what can you do', 'help', 'assist'
            ]
        }

        # Object recognition patterns
        self.object_patterns = [
            'the (.+)',  # the red cup
            'a (.+)',    # a book
            'an (.+)',   # an apple
            '(.+)',      # book (when context is clear)
        ]

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of the given text"""
        text_lower = text.lower()

        # Simple keyword-based classification (in practice, use ML models)
        for action_type, keywords in self.robot_commands.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return action_type, 0.9  # High confidence for keyword match

        # If no clear match, return general
        return 'general', 0.5

    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities (objects, locations, etc.) from text"""
        entities = {}
        text_lower = text.lower()

        # Extract object
        for pattern in self.object_patterns:
            import re
            match = re.search(pattern, text_lower)
            if match:
                entities['object'] = match.group(1)
                break

        # Extract location
        location_keywords = ['kitchen', 'living room', 'bedroom', 'office', 'table', 'counter', 'couch']
        for keyword in location_keywords:
            if keyword in text_lower:
                entities['location'] = keyword
                break

        # Extract color
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown']
        for color in colors:
            if color in text_lower:
                entities['color'] = color
                break

        return entities

    def parse_command(self, text: str) -> RobotCommand:
        """Parse natural language text into a robot command"""
        # Classify intent
        action_type, confidence = self.classify_intent(text)

        # Extract entities
        parameters = self.extract_entities(text)

        # Add raw text for context
        parameters['raw_text'] = text

        return RobotCommand(
            action_type=action_type,
            parameters=parameters,
            confidence=confidence,
            raw_text=text
        )

class ActionMapper:
    """Maps understood commands to robot actions"""

    def __init__(self):
        # Robot capabilities and constraints
        self.robot_capabilities = {
            'navigation': True,
            'manipulation': True,
            'interaction': True,
            'max_speed': 1.0,  # m/s
            'max_lift_height': 1.5,  # meters
            'gripper_range': (0.0, 0.1)  # meters
        }

        # Action mapping rules
        self.action_mapping = {
            'navigation': self._map_navigation,
            'manipulation': self._map_manipulation,
            'interaction': self._map_interaction
        }

    def _map_navigation(self, parameters: Dict) -> Dict:
        """Map navigation parameters to robot navigation commands"""
        action_params = {
            'target_location': parameters.get('location', 'unknown'),
            'speed': parameters.get('speed', 0.5),
            'avoid_obstacles': True
        }

        # If specific location not provided, use context
        if action_params['target_location'] == 'unknown':
            if 'object' in parameters:
                action_params['target_location'] = f'near_{parameters["object"]}'

        return action_params

    def _map_manipulation(self, parameters: Dict) -> Dict:
        """Map manipulation parameters to robot manipulation commands"""
        action_params = {
            'object_to_manipulate': parameters.get('object', 'unknown'),
            'color': parameters.get('color'),
            'action': 'grasp',  # default action
            'precision': 'medium'
        }

        # Determine specific action based on command
        raw_text = parameters.get('raw_text', '').lower()
        if 'put' in raw_text or 'place' in raw_text:
            action_params['action'] = 'place'
        elif 'release' in raw_text or 'let go' in raw_text:
            action_params['action'] = 'release'

        return action_params

    def _map_interaction(self, parameters: Dict) -> Dict:
        """Map interaction parameters to robot interaction commands"""
        action_params = {
            'interaction_type': 'greeting',
            'message': parameters.get('raw_text', 'Hello')
        }

        raw_text = parameters.get('raw_text', '').lower()
        if 'hello' in raw_text or 'hi' in raw_text:
            action_params['interaction_type'] = 'greeting'
        elif 'goodbye' in raw_text or 'bye' in raw_text:
            action_params['interaction_type'] = 'farewell'
        elif 'help' in raw_text or 'assist' in raw_text:
            action_params['interaction_type'] = 'assistance_request'

        return action_params

    def map_command(self, robot_command: RobotCommand) -> Optional[Dict]:
        """Map robot command to executable action"""
        if robot_command.action_type not in self.action_mapping:
            print(f"Unknown action type: {robot_command.action_type}")
            return None

        # Verify robot can perform this action
        if not self.robot_capabilities.get(robot_command.action_type, False):
            print(f"Robot cannot perform {robot_command.action_type} actions")
            return None

        # Map the command
        action_params = self.action_mapping[robot_command.action_type](robot_command.parameters)

        # Add confidence and raw command info
        action_params['confidence'] = robot_command.confidence
        action_params['raw_command'] = robot_command.raw_text

        return action_params

class VoiceToActionSystem:
    """Complete Voice-to-Action system integrating all components"""

    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.nlu = NaturalLanguageUnderstanding()
        self.action_mapper = ActionMapper()

        # Robot interface (simulated)
        self.robot_connected = True
        self.robot_state = {
            'position': (0.0, 0.0, 0.0),
            'orientation': (0.0, 0.0, 0.0, 1.0),
            'gripper': 'open',
            'battery_level': 0.85
        }

        # Processing queue for commands
        self.command_queue = queue.Queue()
        self.processing_thread = None
        self.running = False

    def start_listening(self):
        """Start continuous listening for voice commands"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.processing_thread.start()

        print("Voice-to-Action system started. Listening for commands...")

        while self.running:
            try:
                # Listen for command
                text = self.speech_recognizer.transcribe_from_microphone(timeout=10.0)

                if text and text.strip():
                    print(f"Heard: {text}")

                    # Parse command
                    robot_command = self.nlu.parse_command(text)
                    print(f"Parsed command: {robot_command.action_type} with confidence {robot_command.confidence:.2f}")

                    # Map to action
                    action_params = self.action_mapper.map_command(robot_command)

                    if action_params and action_params['confidence'] > 0.5:
                        # Add to processing queue
                        self.command_queue.put(action_params)
                        print(f"Command added to queue: {action_params}")
                    else:
                        print("Command not understood or confidence too low")

                time.sleep(0.5)  # Small delay between listening attempts

            except KeyboardInterrupt:
                print("Stopping voice-to-action system...")
                self.stop()
                break

    def _process_commands(self):
        """Process commands from the queue in a separate thread"""
        while self.running:
            try:
                # Get command from queue (with timeout)
                action_params = self.command_queue.get(timeout=1.0)

                # Execute command
                success = self.execute_action(action_params)

                if success:
                    print(f"Command executed successfully: {action_params}")
                else:
                    print(f"Command execution failed: {action_params}")

                self.command_queue.task_done()

            except queue.Empty:
                continue  # No commands to process, continue loop
            except Exception as e:
                print(f"Error processing command: {e}")

    def execute_action(self, action_params: Dict) -> bool:
        """Execute the mapped action on the robot"""
        if not self.robot_connected:
            print("Error: Robot not connected")
            return False

        try:
            action_type = action_params.get('action', 'unknown')

            if action_type == 'navigation':
                return self._execute_navigation(action_params)
            elif action_type == 'manipulation':
                return self._execute_manipulation(action_params)
            elif action_type == 'interaction':
                return self._execute_interaction(action_params)
            else:
                print(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            print(f"Error executing action: {e}")
            return False

    def _execute_navigation(self, params: Dict) -> bool:
        """Execute navigation action"""
        target_location = params.get('target_location', 'unknown')
        speed = params.get('speed', 0.5)

        print(f"Navigating to {target_location} at speed {speed}")

        # In a real implementation, this would send navigation commands to the robot
        # For simulation, we'll just update the robot's position
        if target_location == 'kitchen':
            self.robot_state['position'] = (3.0, 2.0, 0.0)
        elif target_location == 'living room':
            self.robot_state['position'] = (1.0, -1.0, 0.0)
        elif 'near' in target_location:
            # Move near the object
            self.robot_state['position'] = (0.5, 0.5, 0.0)

        print(f"Robot moved to position: {self.robot_state['position']}")
        return True

    def _execute_manipulation(self, params: Dict) -> bool:
        """Execute manipulation action"""
        object_name = params.get('object_to_manipulate', 'unknown')
        action = params.get('action', 'grasp')
        color = params.get('color')

        print(f"Manipulating {object_name} (color: {color}), action: {action}")

        # In a real implementation, this would send manipulation commands to the robot
        # For simulation, we'll update the gripper state
        if action == 'grasp':
            self.robot_state['gripper'] = 'closed'
            print("Gripper closed")
        elif action == 'release':
            self.robot_state['gripper'] = 'open'
            print("Gripper opened")
        elif action == 'place':
            self.robot_state['gripper'] = 'open'
            print("Object placed")

        return True

    def _execute_interaction(self, params: Dict) -> bool:
        """Execute interaction action"""
        interaction_type = params.get('interaction_type', 'greeting')
        message = params.get('message', 'Hello')

        print(f"Interaction: {interaction_type}, message: {message}")

        # In a real implementation, this might trigger speech synthesis
        # or other interactive behaviors
        if interaction_type == 'greeting':
            print("Robot says: Hello! How can I help you?")
        elif interaction_type == 'farewell':
            print("Robot says: Goodbye! Have a great day!")
        elif interaction_type == 'assistance_request':
            print("Robot says: I'm ready to help. What would you like me to do?")

        return True

    def stop(self):
        """Stop the voice-to-action system"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("Voice-to-Action system stopped")

# Example usage and demonstration
def main():
    print("Initializing Voice-to-Action System...")

    # Create voice-to-action system
    vta_system = VoiceToActionSystem()

    try:
        # Start listening for commands
        vta_system.start_listening()
    except KeyboardInterrupt:
        print("\nShutting down...")
        vta_system.stop()

if __name__ == "__main__":
    main()
```

```python
# Advanced Voice-to-Action with Context and State Management
import torch
import torch.nn as nn
import whisper
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import datetime
from dataclasses import dataclass

@dataclass
class ConversationContext:
    """Context information for maintaining conversation state"""
    current_task: str = ""
    last_action: str = ""
    last_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    objects_of_interest: List[Dict] = None
    user_preferences: Dict = None
    timestamp: datetime.datetime = None

    def __post_init__(self):
        if self.objects_of_interest is None:
            self.objects_of_interest = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now()

class ContextAwareVoiceToAction:
    """Voice-to-Action system with context awareness and state management"""

    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.nlu = NaturalLanguageUnderstanding()
        self.action_mapper = ActionMapper()

        # Conversation context
        self.context = ConversationContext()

        # Robot state
        self.robot_state = {
            'position': (0.0, 0.0, 0.0),
            'orientation': (0.0, 0.0, 0.0, 1.0),
            'gripper': 'open',
            'battery_level': 0.85,
            'last_action_time': datetime.datetime.now()
        }

        # Command history for context
        self.command_history = []
        self.max_history = 10

    def update_context(self, command: RobotCommand, action_params: Dict = None):
        """Update conversation context based on command and action"""
        # Update current task if navigation command
        if command.action_type == 'navigation' and 'location' in command.parameters:
            self.context.current_task = f"navigating_to_{command.parameters['location']}"

        # Update last action
        self.context.last_action = command.action_type

        # Update objects of interest if manipulation command
        if command.action_type == 'manipulation' and 'object' in command.parameters:
            obj_info = {
                'name': command.parameters['object'],
                'color': command.parameters.get('color'),
                'timestamp': datetime.datetime.now()
            }
            self.context.objects_of_interest.append(obj_info)

            # Keep only recent objects
            if len(self.context.objects_of_interest) > 5:
                self.context.objects_of_interest = self.context.objects_of_interest[-5:]

        # Update timestamp
        self.context.timestamp = datetime.datetime.now()

        # Add to command history
        self.command_history.append({
            'command': command,
            'action_params': action_params,
            'timestamp': self.context.timestamp
        })

        # Keep only recent history
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]

    def resolve_pronouns_and_references(self, text: str) -> str:
        """Resolve pronouns and references based on context"""
        text_lower = text.lower()

        # Replace "it" with last mentioned object
        if 'it' in text_lower and self.context.objects_of_interest:
            last_object = self.context.objects_of_interest[-1]['name']
            text = text.replace(' it ', f' {last_object} ')

        # Replace "there" with last location
        if 'there' in text_lower and self.context.last_action == 'navigation':
            # In a real implementation, you'd have the actual location
            text = text.replace(' there ', ' to the previous location ')

        # Replace "same" with context-appropriate reference
        if 'same' in text_lower:
            if self.context.last_action == 'manipulation':
                last_object = self.context.objects_of_interest[-1]['name'] if self.context.objects_of_interest else 'object'
                text = text.replace(' same ', f' {last_object} ')

        return text

    def parse_command_with_context(self, text: str) -> RobotCommand:
        """Parse command with context awareness"""
        # Resolve pronouns and references first
        resolved_text = self.resolve_pronouns_and_references(text)

        # If command is ambiguous, use context to disambiguate
        if 'it' in resolved_text.lower() or 'there' in resolved_text.lower():
            # Add context to make command more specific
            if self.context.current_task:
                resolved_text += f" related to {self.context.current_task}"

        # Parse the resolved command
        command = self.nlu.parse_command(resolved_text)

        # Enhance command with context
        if self.context.current_task and command.action_type == 'navigation':
            command.parameters['context_task'] = self.context.current_task

        return command

    def execute_command_with_context(self, text: str) -> bool:
        """Execute command with full context awareness"""
        try:
            # Parse command with context
            robot_command = self.parse_command_with_context(text)

            # Map to action
            action_params = self.action_mapper.map_command(robot_command)

            if action_params and action_params['confidence'] > 0.5:
                # Execute action
                success = self.execute_action(action_params)

                if success:
                    # Update context
                    self.update_context(robot_command, action_params)
                    print(f"Command executed successfully: {action_params}")
                    return True
                else:
                    print(f"Command execution failed: {action_params}")
                    return False
            else:
                print("Command not understood or confidence too low")
                return False

        except Exception as e:
            print(f"Error executing command with context: {e}")
            return False

    def execute_action(self, action_params: Dict) -> bool:
        """Execute action with context considerations"""
        # In a real implementation, this would be similar to the basic version
        # but could use context to make better decisions

        action_type = action_params.get('action', 'unknown')

        if action_type == 'navigation':
            return self._execute_navigation(action_params)
        elif action_type == 'manipulation':
            return self._execute_manipulation(action_params)
        elif action_type == 'interaction':
            return self._execute_interaction(action_params)
        else:
            print(f"Unknown action type: {action_type}")
            return False

    def _execute_navigation(self, params: Dict) -> bool:
        """Execute navigation with context awareness"""
        target_location = params.get('target_location', 'unknown')
        print(f"Navigating to {target_location}")

        # Update robot state
        if target_location == 'kitchen':
            self.robot_state['position'] = (3.0, 2.0, 0.0)
        elif target_location == 'living room':
            self.robot_state['position'] = (1.0, -1.0, 0.0)

        # Update context
        self.context.last_location = self.robot_state['position']

        return True

    def _execute_manipulation(self, params: Dict) -> bool:
        """Execute manipulation with context awareness"""
        object_name = params.get('object_to_manipulate', 'unknown')
        action = params.get('action', 'grasp')
        print(f"Manipulating {object_name}, action: {action}")

        # Update gripper state
        if action == 'grasp':
            self.robot_state['gripper'] = 'closed'
        elif action in ['release', 'place']:
            self.robot_state['gripper'] = 'open'

        return True

    def _execute_interaction(self, params: Dict) -> bool:
        """Execute interaction with context awareness"""
        interaction_type = params.get('interaction_type', 'greeting')
        message = params.get('message', 'Hello')
        print(f"Interaction: {interaction_type}, message: {message}")

        return True

    def get_system_status(self) -> Dict:
        """Get current system status including context"""
        return {
            'robot_state': self.robot_state,
            'context': {
                'current_task': self.context.current_task,
                'last_action': self.context.last_action,
                'objects_of_interest': self.context.objects_of_interest,
                'command_history_count': len(self.command_history)
            },
            'battery_level': self.robot_state['battery_level']
        }

# Voice-to-Action with Machine Learning Enhancement
class MLVoiceToAction:
    """Voice-to-Action system enhanced with machine learning components"""

    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.context_aware_system = ContextAwareVoiceToAction()

        # ML models for enhancement
        self.intent_model = self._initialize_intent_model()
        self.action_model = self._initialize_action_model()
        self.user_model = self._initialize_user_model()

    def _initialize_intent_model(self):
        """Initialize intent classification model"""
        # In practice, this would be a trained model
        # For this example, we'll use a simple rule-based approach
        # enhanced with ML principles
        return {
            'command_patterns': {
                'navigation': [
                    (r'go to (.+)', 0.9),
                    (r'move to (.+)', 0.9),
                    (r'navigate to (.+)', 0.9),
                    (r'go (.+)', 0.7),
                ],
                'manipulation': [
                    (r'pick up (.+)', 0.9),
                    (r'grasp (.+)', 0.9),
                    (r'take (.+)', 0.8),
                    (r'get (.+)', 0.8),
                ],
                'interaction': [
                    (r'hello', 0.8),
                    (r'hi', 0.8),
                    (r'goodbye', 0.8),
                    (r'bye', 0.8),
                ]
            }
        }

    def _initialize_action_model(self):
        """Initialize action prediction model"""
        # This model would predict the best action based on context
        return {
            'action_weights': {
                'navigation': 0.3,
                'manipulation': 0.4,
                'interaction': 0.3
            }
        }

    def _initialize_user_model(self):
        """Initialize user preference model"""
        # This model would learn user preferences over time
        return {
            'preferences': {},
            'interaction_history': []
        }

    def classify_intent_ml(self, text: str) -> Tuple[str, float]:
        """Classify intent using ML-enhanced approach"""
        import re

        best_match = ('general', 0.0)

        for intent, patterns in self.intent_model['command_patterns'].items():
            for pattern, base_confidence in patterns:
                if re.search(pattern, text.lower()):
                    # In a real ML model, this would be a learned confidence
                    confidence = base_confidence
                    if confidence > best_match[1]:
                        best_match = (intent, confidence)

        return best_match

    def predict_best_action(self, intent: str, context: ConversationContext, text: str) -> str:
        """Predict the best action based on intent, context, and text"""
        # In a real ML model, this would use a trained neural network
        # For this example, we'll use a weighted approach

        # Adjust weights based on context
        weights = self.action_model['action_weights'].copy()

        # If robot is already near an object, manipulation is more likely
        if context.objects_of_interest:
            weights['manipulation'] *= 1.5

        # If user just navigated somewhere, interaction might be more likely
        if context.last_action == 'navigation':
            weights['interaction'] *= 1.2

        # Return the action type with highest weight
        return max(weights, key=weights.get)

    def execute_command_ml(self, text: str) -> bool:
        """Execute command using ML-enhanced processing"""
        try:
            # Classify intent with ML
            intent, confidence = self.classify_intent_ml(text)

            # Get context
            context = self.context_aware_system.context

            # Predict best action
            predicted_action = self.predict_best_action(intent, context, text)

            # Create robot command
            robot_command = RobotCommand(
                action_type=predicted_action,
                parameters={'raw_text': text, 'predicted_intent': intent},
                confidence=confidence,
                raw_text=text
            )

            # Use context-aware system to execute
            action_params = self.context_aware_system.action_mapper.map_command(robot_command)

            if action_params and action_params['confidence'] > 0.3:  # Lower threshold due to ML enhancement
                success = self.context_aware_system.execute_action(action_params)

                if success:
                    # Update context
                    self.context_aware_system.update_context(robot_command, action_params)
                    print(f"ML-enhanced command executed: {action_params}")
                    return True

            print("ML-enhanced command not executed - low confidence or mapping failed")
            return False

        except Exception as e:
            print(f"Error in ML-enhanced command execution: {e}")
            return False

# Example usage of enhanced voice-to-action system
def run_enhanced_vta_example():
    print("Running Enhanced Voice-to-Action Example...")

    # Create enhanced system
    ml_vta = MLVoiceToAction()

    # Example commands
    commands = [
        "Hey robot, go to the kitchen",
        "Now pick up the red cup from the table",
        "Take it to the living room",
        "Put it on the couch"
    ]

    for i, command in enumerate(commands):
        print(f"\nStep {i+1}: Processing command: '{command}'")

        # Execute with ML enhancement
        success = ml_vta.execute_command_ml(command)

        if success:
            print(f"✓ Command '{command}' executed successfully")
        else:
            print(f"✗ Command '{command}' failed")

        # Get system status
        status = ml_vta.context_aware_system.get_system_status()
        print(f"System status: Robot at {status['robot_state']['position']}, Gripper: {status['robot_state']['gripper']}")

        # Small delay for simulation
        import time
        time.sleep(1)

def main():
    print("Starting Voice-to-Action Systems Using Speech Recognition")

    # Run the enhanced example
    run_enhanced_vta_example()

    print("\nVoice-to-Action systems enable natural human-robot interaction")
    print("by converting spoken language into physical robot behaviors.")

if __name__ == "__main__":
    main()
```

## Summary

Voice-to-Action systems represent a crucial advancement in human-robot interaction, enabling robots to understand spoken commands and translate them directly into physical actions. These systems integrate speech recognition, natural language understanding, and action execution in a seamless pipeline that allows for intuitive communication between humans and robots.

The key components of voice-to-action systems include:
- **Speech Recognition**: Converting audio input to text using models like Whisper
- **Natural Language Understanding**: Interpreting the meaning and intent of spoken commands
- **Action Mapping**: Translating understood commands to robot-specific actions
- **Action Execution**: Executing the mapped actions on the physical robot
- **Context Management**: Maintaining conversation state and resolving references

Voice-to-action systems are particularly valuable for Physical AI and humanoid robotics because they enable more natural and efficient human-robot collaboration. These systems can handle complex, context-dependent commands and adapt to different users and environments.

The integration of machine learning enhances these systems by enabling them to learn from interactions, improve their understanding over time, and adapt to user preferences and environmental conditions.

## Exercises

1. **Basic Understanding**: Explain the difference between speech recognition and natural language understanding in voice-to-action systems. Why are both components necessary for effective robot control?

2. **Application Exercise**: Design a voice-to-action system for a household robot that can handle commands like "Robot, please bring me the blue water bottle from the kitchen counter." Include the specific components needed and how they would work together.

3. **Implementation Exercise**: Modify the voice-to-action system to handle ambiguous commands by asking clarifying questions. For example, if a user says "pick up the cup," and there are multiple cups, the robot should ask "Which cup would you like me to pick up?"

4. **Challenge Exercise**: Implement a voice-to-action system that learns user preferences over time. The system should adapt its interpretations based on past interactions and user feedback, improving its accuracy and relevance for each individual user.