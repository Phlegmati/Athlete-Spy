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
#include "Model.h"

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>

TFLiteModel::TFLiteModel(const cv::String& file_path)
{
    flat_buffer_model = tflite::FlatBufferModel::BuildFromFile(file_path.c_str());

    if (!flat_buffer_model) {
        throw std::runtime_error("Failed to load TFLite model");
    }

    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    tflite::InterpreterBuilder(*flat_buffer_model, op_resolver)(&interpreter);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }

    tflite::PrintInterpreterState(interpreter.get());

    auto input = interpreter->inputs()[0];
    input_height = interpreter->tensor(input)->dims->data[1];
    input_width = interpreter->tensor(input)->dims->data[2];
}

cv::Size TFLiteModel::get_input_size() const
{
    return cv::Size(input_width, input_height);
}

float* TFLiteModel::infer(cv::Mat& image) const
{
    memcpy(interpreter->typed_input_tensor<unsigned char>(0), image.data, image.total() * image.elemSize());

    if (interpreter->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Inference failed");
    }

    return interpreter->typed_output_tensor<float>(0);
}
