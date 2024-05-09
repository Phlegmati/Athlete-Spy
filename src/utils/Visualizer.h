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
#include "PoseData.h"
#include <opencv2/imgproc.hpp>

static const cv::Scalar GREEN = cv::Scalar(0, 255, 0);
static const cv::Scalar BLUE = cv::Scalar(255, 0, 0);
static const int LINE_THICKNESS = 1;
static const int CIRCLE_RADIUS = 4;
static const float CONFIDENCE_THRESHOLD = 0.2f;

static const std::vector<std::pair<int, int>> LINES = {
    // head
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    // upper body
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12},
    // lower body
    {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

static void draw_bounds(cv::Mat& current_frame, PoseData& pose) 
{
    cv::rectangle(current_frame, pose.get_bounds(), GREEN, LINE_THICKNESS);
}

static void draw_skeleton(cv::Mat& current_frame, PoseData& pose)
{
    for (int i = 0; i < pose.NUM_KEYPOINTS; ++i) {

        cv::Point position = pose.get_keypoint(i).pixel_coord();
        float conf = pose.get_keypoint(i).confidence();

        if (conf > CONFIDENCE_THRESHOLD) {
            cv::circle(current_frame, position, CIRCLE_RADIUS, BLUE, LINE_THICKNESS);
        }
    }

    for (const auto& [index_from, index_to] : LINES) {

        cv::Point from = pose.get_keypoint(index_from).pixel_coord();
        float conf_from = pose.get_keypoint(index_from).confidence();

        cv::Point to = pose.get_keypoint(index_to).pixel_coord();
        float conf_to = pose.get_keypoint(index_to).confidence();

        if (conf_from > CONFIDENCE_THRESHOLD && conf_to > CONFIDENCE_THRESHOLD) {
            line(current_frame, from, to, BLUE, LINE_THICKNESS);
        }
    }
}