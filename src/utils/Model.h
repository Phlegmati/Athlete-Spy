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
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>

#include <opencv2/core/mat.hpp>

class TFLiteModel {

public:

    TFLiteModel(const cv::String& file_path);

    /** @brief Returns desired input frame size for given model.*/
    cv::Size get_input_size() const;

    /** 
    @brief Copies image buffer into model and processes it.
    @return Returns a tensor of shape [1,1,17,3].
    */
    float* infer(cv::Mat& image) const;

private:

    int input_width, input_height;

    std::unique_ptr<tflite::impl::FlatBufferModel> flat_buffer_model;
    std::unique_ptr<tflite::Interpreter> interpreter;
};
