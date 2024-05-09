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
#include <opencv2/core/types.hpp>

struct KeypointData {

public:

    KeypointData(const cv::String& t_name) : c_name(t_name), m_confidence(0), m_pixel_coordinate(0, 0) {};
    void set(const cv::Point& position, const float conf);
    const cv::String print();

    cv::Point pixel_coord() const { return m_pixel_coordinate; }
    float confidence() const { return m_confidence; }

private:

    const cv::String c_name;

    cv::Point m_pixel_coordinate;
    float m_confidence;
};