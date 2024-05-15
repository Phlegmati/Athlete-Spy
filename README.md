# Athlete-Spy

![Output example](assets/processed_output.gif)

### Human pose estimation tool using TensorFlow Lite and OpenCV

A simple commandline tool for extracting and saving keypoints of a human pose from videos.

(C) Copyright Nico Mahler 2024


#### Prerequisites

- Python v 3.4+

        $ python --version

- CMake v 3.15+

        $ cmake --version

- Conan package manager

        $ pip install conan
  
    (Hint: If not installed globally, add the scripts directory to path enviroment variable)

        %USERPROFILE%\AppData\Local\Packages\%PYTHON_VERSION%\local-packages\%PYTHON_VERSION%\scripts
  
#### Installation

- Clone repository

        $ git clone https://github.com/Phlegmati/Athlete-Spy

- Create conan build profile

        $ conan profile detect

- Installing dependencies with Conan

        $ conan install . -s compiler.cppstd=17 --build=missing

- Building

        $ cmake --preset conan-default
        $ cmake --build --preset conan-release

#### Usage

- Run build with examples

        $ build\Release\athlete-spy.exe

    (Hint: Looks for example files in the "asset" folder in the current runtime enviroment)
    
        athlete-spy
        ├── assets
        │   ├── lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite
        │   └── NyjahHuston@OlympicsParis.mp4
        └── build
            └── Release
                └── athlete-spy.exe

- Manual Page

        $ build\Release\athlete-spy.exe --help

- Pass video file (supported: *.mp4 *.avi)

        $ build\Release\athlete-spy.exe --video=path/to/video.file

- Pass model file (supported: *.tflite)

        $ build\Release\athlete-spy.exe --model=path/to/model.file
        
#### Output

- The processed pose video will be saved as an *.avi video and *.txt file

        athlete-spy
        └── output
            ├── video_processed.txt
            └── video_processed.mp4
            
#### References

- See documentation of the used Model
https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4

- See documentation of the used TFLite API (v. 2.12.0) https://www.tensorflow.org/lite/api_docs/cc

- See documentation of the used OpenCV API (v. 4.5.5)
https://docs.opencv.org/4.5.5/
