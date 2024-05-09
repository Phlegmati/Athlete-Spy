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
#include "Visualizer.h"

#include <opencv2/core/types.hpp>

void KeypointData::set(const cv::Point& position, const float conf)
{
    m_pixel_coordinate = position;
    m_confidence = conf;
}

const cv::String KeypointData::print()
{
    std::stringstream output;

    float conf = int(m_confidence * 100) / 100.f;

    output << "(" << c_name << ": [" << m_pixel_coordinate.x << ", " << m_pixel_coordinate.y << "] conf: " << conf << ")";

    return output.str();
}