# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Unity project for real-time EEG signal processing and SSVEP stimulation system. Key features include:
- Real-time EEG data acquisition from ESP32 devices with multi-channel support
- Unity visualization of EEG waveforms with adjustable parameters
- SSVEP visual stimulator with configurable frequency/color/duration
- Integration with Sentis/ONNX model inference for real-time EEG classification
- GPU/CPU backend selection with automatic fallback
- Thread-safe data processing to avoid UI blocking
- Support for YVR SDK and Oculus XR

## Code Architecture

### Main Components

1. **EEG Classification** (`Assets/Algorithm/EEG_Classify_test.cs`)
   - Real-time inference component supporting ONNX models
   - Backend selection (GPUCompute/CPU) with automatic fallback
   - Performance timing and diagnostics
   - Softmax normalization of outputs

2. **UDP Communication** (`Assets/Comunication/UDP_1.cs`)
   - Asynchronous UDP data reception from ESP32 devices
   - CRC validation and serial number verification
   - Thread-safe data handling with ConcurrentQueue

3. **TCP Client** (`Assets/Comunication/TCP_Client.cs`)
   - TCP client that reads EEG classification results and sends to robot
   - Background thread handling for network I/O
   - JSON data transmission with UTF-8 encoding
   - Automatic reconnection and connection management

4. **EEG Data Storage** (`Assets/DataStore/EEGDatasave.cs`)
   - EEG data recording and saving functionality
   - Channel selection via UI toggles
   - Configurable data point count for saving
   - CSV format data export with timestamps

5. **EEG Visualization** (`Assets/Data_Display/EEGVisualizer.cs`)
   - Real-time EEG waveform display using LineRenderer
   - RMS and amplitude calculation for each channel

6. **SSVEP Stimulus** (`Assets/Stimulus/SSVEPStimulus.cs`)
   - Visual stimulator with configurable frequency and duration

7. **Main Logic** (`Assets/Logic/MainLogic.cs`)
   - Core application logic and initialization

### Key Directories
```
ESP32-Unity/
├── Assets/
│   ├── Algorithm/           # EEG inference and algorithms
│   ├── Comunication/        # UDP/TCP communication
│   ├── Data_Display/        # EEG visualization
│   ├── DataStore/           # EEG data storage
│   ├── Stimulus/            # SSVEP stimulation
│   ├── Logic/               # Main application logic
│   └── Scenes/              # Unity scenes
├── ProjectSettings/         # Unity project settings
└── Packages/                # Package management
```

## Development Environment

- Unity Version: 2022.3.57f1c1
- Key Dependencies:
  - Unity.Sentis (for ONNX model inference)
  - YVR SDK packages (com.yvr.core, com.yvr.platform, com.yvr.utilities)
  - XR packages (Oculus, AR Foundation, etc.)

## Common Development Tasks

### Building the Project
1. Open the project in Unity 2022.3+
2. Ensure all package dependencies are resolved
3. Build from File > Build Settings

### Adding New Algorithms
1. Create new scripts in `Assets/Algorithm/`
2. Integrate with the UDP_1.cs component through data events
3. Follow the pattern of EEG_Classify_test.cs for consistency

### Adding New Visualizations
1. Reference EEGVisualizer.cs for implementation patterns
2. Subscribe to UDP data events for real-time updates

### Adding New Stimulus Modes
1. Extend from SSVEPStimulus.cs or create new stimulus components
2. Configure parameters in the Unity Inspector

### Extending Data Communication
1. TCP_Client.cs handles robot communication - extend for new protocols
2. UDP_1.cs manages ESP32 data reception - modify for new data formats
3. Use thread-safe patterns with ConcurrentQueue for data handling

### Data Storage and Export
1. EEGDatasave.cs manages EEG data recording
2. Data is saved in CSV format with timestamps
3. Supports both local storage and AR device storage paths

## Important Notes

- Data synchronization between UDP receiver thread and Unity main thread uses queues and flags
- Inference backend automatically falls back from GPU to CPU if unavailable
- Worker initialization and inference are delayed/fractional-frame to avoid blocking
- Data freshness checking ensures real-time processing with timeouts
- Softmax normalization is applied to inference outputs (equivalent to PyTorch torch.softmax)
- TCP client uses background threads for network I/O to prevent UI blocking
- EEG data storage supports configurable channel selection and point counts
- All network communication includes proper error handling and reconnection logic