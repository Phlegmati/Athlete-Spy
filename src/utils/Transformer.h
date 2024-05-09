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

class TFLiteTransformer {

public:

    TFLiteTransformer() : m_left_offset(0), m_top_offset(0), m_square_dim(0) {}
    TFLiteTransformer(const int width, const int height);

    /** 
    @brief Crops given image into a resized square.
    @return Returns formatted image.
    */
    cv::Mat crop_and_resize(const cv::Mat& image, const cv::Size& desired_size);

    /** 
    @brief Reads the keypoints of a [1,1,17,3] shaped tensor and remaps the positions to the original frame size.
    @return Returns a Pose, with collected keypoints and bounding box, in pixel coordinates. 
    */
    PoseData parse_pose(const float* output);

private:

    cv::Rect get_rect() const;
    cv::Point get_pixel_coord(const float x, const float y) const;
    int remap_x_pos(const float x) const;
    int remap_y_pos(const float y) const;

    int m_left_offset;
    int m_top_offset;
    int m_square_dim;
};