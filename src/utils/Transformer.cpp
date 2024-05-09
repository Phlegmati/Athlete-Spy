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
#include "Transformer.h"

#include <opencv2/imgproc.hpp>

TFLiteTransformer::TFLiteTransformer(const int width, const int height)
{
    m_square_dim = MIN(width, height);
    m_left_offset = (width - m_square_dim) / 2;
    m_top_offset = (height - m_square_dim) / 2;
}

cv::Mat TFLiteTransformer::crop_and_resize(const cv::Mat& image, const cv::Size& desired_size)
{
    cv::Mat resized_image;

    cv::resize(
        image(get_rect()),
        resized_image,
        desired_size
    );

    return resized_image;
}

PoseData TFLiteTransformer::parse_pose(const float* output)
{
    PoseData pose;

    float left = INT_MAX;
    float top = INT_MAX;
    float bottom = INT_MIN;
    float right = INT_MIN;

    for (int i = 0; i < pose.NUM_KEYPOINTS; ++i) {

        // Results are in range of [0,1] floating number
        float y = output[i * 3];
        float x = output[i * 3 + 1];
        float conf = output[i * 3 + 2];

        // Remap to pixel coordinates of frame
        cv::Point position = get_pixel_coord(x, y);

        // Lowest Values will be top left corner
        left = MIN(left, position.x);
        top = MIN(top, position.y);

        // Highest value bottom right corner
        right = MAX(right, position.x);
        bottom = MAX(bottom, position.y);

        // Copy data
        pose.set_keypoint_data(i, position, conf);
    }

    pose.set_bounds(left, top, right, bottom);

    return pose;
}

// returns a squared rect, centered in the frame
cv::Rect TFLiteTransformer::get_rect() const 
{ 
    return cv::Rect(m_left_offset, m_top_offset, m_square_dim, m_square_dim); 
}

cv::Point TFLiteTransformer::get_pixel_coord(const float x, const float y) const
{
    return cv::Point(
        remap_x_pos(x),
        remap_y_pos(y)
    );
}

// Retargets from range [0,1] -> [0, width]
int TFLiteTransformer::remap_x_pos(const float x) const 
{
    return (x * m_square_dim) + m_left_offset; 
}

// Retargets from range [0,1] -> [0, height]
int TFLiteTransformer::remap_y_pos(const float y) const 
{ 
    return (y * m_square_dim) + m_top_offset; 
}