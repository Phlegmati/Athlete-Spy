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
#include "KeypointData.h"

struct PoseData {

public:

    void set_keypoint_data(const int index, const cv::Point& position, const float conf);
    KeypointData get_keypoint(const int index) const;
    
    void set_bounds(const int left, const int top, const int right, const int bottom);
    cv::Rect get_bounds() const { return bounds; }

    /**
    @brief Prints all keypoint positions and their confidence.
    @return Returns a cv::String.
    */
    const cv::String print();

    const int NUM_KEYPOINTS = 17;

private:

    cv::Rect bounds;
    std::vector<KeypointData> keypoints{
        KeypointData("nose"),
        KeypointData("left eye"),
        KeypointData("right eye"),
        KeypointData("left ear"),
        KeypointData("right ear"),
        KeypointData("left shoulder"),
        KeypointData("right shoulder"),
        KeypointData("left elbow"),
        KeypointData("right elbow"),
        KeypointData("left wrist"),
        KeypointData("right wrist"),
        KeypointData("left hip"),
        KeypointData("right hip"),
        KeypointData("left knee"),
        KeypointData("right knee"),
        KeypointData("left ankle"),
        KeypointData("right ankle"),
    };
};