#pragma region LICENSE
/**
 * Copyright 2024 Nico Mahler
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma endregion

#pragma once
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <filesystem>
#include <fstream>

#include "utils/Transformer.h"
#include "utils/Model.h"
#include "utils/Visualizer.h"

using namespace cv;

const float MILLISECONDS_PER_SECOND = 1000.f;

static bool parse_arguments(const int argc, char* argv[], String& model_path, String& video_path) 
{
    std::map<String, String> arguments;

    for (int i = 1; i < argc; ++i) {
        
        const String arg(argv[i]);

        if (arg.find("--") == 0) {
            
            const size_t equal_sign_pos = arg.find("=");
            const String key = arg.substr(0, equal_sign_pos);
            const String value = equal_sign_pos != std::string::npos ? arg.substr(equal_sign_pos + 1) : "";

            arguments[key] = value;
        }
    }

    const String supported_model_formats{
        ".tflite"
    };

    const String supported_video_formats{
        ".mp4 .avi"
    };

    if (arguments.count("--help") || arguments.count("--man")){

        std::cout
            << "\n"
            << "# # # # # # # # # # # # # # # # # # # # #   MANUAL   # # # # # # # # # # # # # # # # # # # # #"
            << "\n\n"
            << "COMMANDS:"
            << "\n"
            << "--help --man \t\t\t Manual Page"
            << "\n"
            << "--video=/path/to/video.file \t param for video input file \t supported formats: " << supported_video_formats
            << "\n"
            << "--model=/path/to/model.file \t param for tflite model file \t supported formats: " << supported_model_formats
            << "\n\n"
            << "OUTPUT:"
            << "\n"
            << "/path/to/output/video_processed.avi \t Processed Video"
            << "\n"
            << "/path/to/output/video_processed.txt \t Textfile with Keypoints"
            << "\n\n"
            << "For further details, look at README.md"
            << "\n\n"
            << "# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #"
        << std::endl;

        return false;
    }

    if (arguments.count("--model")) {
        
        model_path = arguments["--model"];

        // Check if file exists
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file " << model_path << " not existing. Aborting..." << std::endl;
            return false;
        }

        // check if model file is valid
        const std::filesystem::path file_name = std::filesystem::path(model_path).filename();
        bool model_valid = supported_model_formats.find(file_name.extension().string()) != std::string::npos;
        if (!model_valid) {

            std::cerr << file_name.string() << " is not a valid Tensorflow Lite model file! Aborting..." << std::endl;
            return false;
        }
    }

    if (arguments.count("--video")) {

        video_path = arguments["--video"];

        // Check if file exists
        if (!std::filesystem::exists(video_path)) {

            std::cerr << "Video file " << video_path << " not existing. Aborting..." << std::endl;
            return false;
        }

        // check if video file is valid
        const std::filesystem::path file_name = std::filesystem::path(video_path).filename();
        bool video_valid = supported_video_formats.find( file_name.extension().string() ) != std::string::npos;

        if (!video_valid) {

            std::cerr
                << file_name.string() << " is not a valid Video Format! Supported Formats: "
                << supported_video_formats << " Aborting..."
            << std::endl;

            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {

    // Model from https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4
    String MODEL_PATH = "assets/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite";
    String INPUT_PATH = "assets/NyjahHuston@OlympicsParis.mp4";

    const String OUTPUT_FILENAME = std::filesystem::path(INPUT_PATH).filename().replace_extension("").string() + "_processed";
    const String OUTPUT_DIRECTORY = "output/";

    const String VIDEO_OUTPUT_PATH = OUTPUT_DIRECTORY + OUTPUT_FILENAME + ".avi";
    const String TEXT_OUTPUT_PATH = OUTPUT_DIRECTORY + OUTPUT_FILENAME + ".txt";

    // Check and valididate model and video file
    if (!parse_arguments(argc, argv, MODEL_PATH, INPUT_PATH)) return 1;

    // Create output directory if necessary
    if (!std::filesystem::exists(OUTPUT_DIRECTORY))  std::filesystem::create_directory(OUTPUT_DIRECTORY);

    // TFLite Model 
    TFLiteModel model(MODEL_PATH);

    // Video input
    VideoCapture input(INPUT_PATH);

    // Video output file
    VideoWriter video_file(
        VIDEO_OUTPUT_PATH,
        VideoWriter::fourcc('M','J','P','G'),
        input.get(CAP_PROP_FPS),
        Size(input.get(CAP_PROP_FRAME_WIDTH), input.get(CAP_PROP_FRAME_HEIGHT)),
        true
    );

    // Text output file
    std::ofstream txt_file(TEXT_OUTPUT_PATH);

    // Helper for input mapping & output parsing
    TFLiteTransformer transformer(
        input.get(CAP_PROP_FRAME_WIDTH), 
        input.get(CAP_PROP_FRAME_HEIGHT)
    );

    Mat current_frame;
    std::chrono::steady_clock::time_point start, end;

    // Process each frame
    for (uint frame_pos = 0; frame_pos < input.get(CAP_PROP_FRAME_COUNT); frame_pos++) {

        // Start tracking time
        start = std::chrono::steady_clock::now();

        // Next frame
        input >> current_frame;

        // break the loop if video is corrupt or ended
        if (current_frame.empty())  break;

        // Squared image
        Mat resized_image = transformer.crop_and_resize(current_frame, model.get_input_size());

        // Tensor of the 17 keypoints with y,x coordinate and confidence, each in Range [0,1]
        float* result = model.infer(resized_image);

        // Pose with infered keypoints
        PoseData pose = transformer.parse_pose(result);

        // Visualize keypoints as pose skeleton on frame
        draw_skeleton(current_frame, pose);

        // Draw bounds
        draw_bounds(current_frame, pose);

        // Write Keypoints to text file
        if (txt_file.is_open())     txt_file << "frame " << frame_pos << ": " << pose.print() << '\n';

        // Write frame to output video
        if (video_file.isOpened())  video_file.write(current_frame);

        // Display in window
        imshow("Output", current_frame);

        // Stop tracking Time
        end = std::chrono::steady_clock::now();

        // To render @ TARGET_FPS we need to wait the frame duration minus the processing time
        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        int frame_duration = int(MILLISECONDS_PER_SECOND / input.get(CAP_PROP_FPS));
        int waiting_time = processing_time < frame_duration ? frame_duration - processing_time : 1;

        // Key press terminates window, else wait until new frame can render
        if (waitKey(waiting_time) >= 0) {
            break;
        }
    }

    // Clean up
    input.release();
    video_file.release();
    txt_file.close();

    // Close Preview
    destroyAllWindows();

    return 0;
}