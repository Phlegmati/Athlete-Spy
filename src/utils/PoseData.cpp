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

void PoseData::set_keypoint_data(const int index, const cv::Point& position, const float conf)
{
    keypoints.at(index).set(position, conf);
}

KeypointData PoseData::get_keypoint(const int index) const
{
    return keypoints.at(index); 
}

void PoseData::set_bounds(const int left, const int top, const int right, const int bottom)
{
    uint width = abs(right - left);
    uint height = abs(bottom - top);

    bounds = cv::Rect(left, top, width, height);
}

const cv::String PoseData::print()
{
    cv::String content;

    for (auto& entry : keypoints) {

        content += entry.print() + " \t";
    }

    return content;
}