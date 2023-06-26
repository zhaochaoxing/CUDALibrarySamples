/*  
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */  

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.hpp>

#include "../../einsum.h"
#include "../../einsum_mg.h"

template<>
struct CuTensorTypeTraits<at::Half> {
  static const cudaDataType_t cudaType = CUDA_R_16F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<at::BFloat16> {
  static const cudaDataType_t cudaType = CUDA_R_16BF;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<float>> {
  static const cudaDataType_t cudaType = CUDA_C_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_TF32;
  typedef c10::complex<float> ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<double>> {
  static const cudaDataType_t cudaType = CUDA_C_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_64F;
  typedef c10::complex<double> ScalarType;
};

torch::Tensor einsum(
    std::string subscripts,
    torch::Tensor input_0,
    torch::Tensor input_1,
    bool conjA = false,
    bool conjB = false
) {
  at::Tensor output_tensor;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB);
    if (!myEinsum.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }

    output_tensor = torch::empty(myEinsum.getOutputShape(), input_0.options());

    size_t worksize = myEinsum.getWorksize();
    at::Tensor workspace = at::empty({static_cast<int>(worksize)}, at::CUDA(at::kByte));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = myEinsum.execute(GetCuTensorHandle(),
                                input_0.data_ptr<scalar_t>(),
                                input_1.data_ptr<scalar_t>(),
                                output_tensor.data_ptr<scalar_t>(),
                                workspace.data_ptr<uint8_t>(),
                                stream);

    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return output_tensor;
}

std::vector<int64_t> getEinsumOutputShape(
    std::string subscripts,
    torch::Tensor input_0,
    torch::Tensor input_1,
    bool conjA = false,
    bool conjB = false
) {
  std::vector<int64_t> output_shape;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB);
    if (!myEinsum.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }
    output_shape = myEinsum.getOutputShape();
  });
  return output_shape;
}

bool einsumV2(
    std::string subscripts,
    torch::Tensor input_0,
    torch::Tensor input_1,
    torch::Tensor output_tensor,
    bool conjA = false,
    bool conjB = false
) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB);
    if (!myEinsum.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }
    size_t worksize = myEinsum.getWorksize();
    at::Tensor workspace = at::empty({static_cast<int>(worksize)}, at::CUDA(at::kByte));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = myEinsum.execute(GetCuTensorHandle(),
                                input_0.data_ptr<scalar_t>(),
                                input_1.data_ptr<scalar_t>(),
                                output_tensor.data_ptr<scalar_t>(),
                                workspace.data_ptr<uint8_t>(),
                                stream);

    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return true;
}

bool init(const int32_t numDevices) {
  bool ret = CutensorMgConfig::Init(numDevices);
  if (! ret) throw std::runtime_error("cutensor: Init failed.");
  return true;
}

bool fromTensor(TensorMg& dst, torch::Tensor src) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "fromTensor", [&] {
    bool ret = TensorMg::fromTensor<scalar_t>(dst, src.sizes().vec(), src.data_ptr<scalar_t>());
    if (! ret) throw std::runtime_error("cutensor: failed.");
  });
  return true;
}

bool toTensor(torch::Tensor dst, TensorMg& src) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dst.scalar_type(), "toTensor", [&] {
    bool ret = TensorMg::toTensor<scalar_t>(dst.sizes().vec(), dst.data_ptr<scalar_t>(), src);
    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return true;
}

std::vector<int64_t> getOutputShapeMg(std::string& subscripts, TensorMg& input_0, TensorMg& input_1) {
  EinsumMg myEinsumMg(subscripts, input_0, input_1);
  if (!myEinsumMg.isInitialized()) {
    throw std::runtime_error("cutensorMg: Initialization failed.");
  }
  return myEinsumMg.getOutputShape();
}

bool einsumMg(std::string& subscripts, TensorMg& input_0, TensorMg& input_1, TensorMg& output, torch::Tensor& origin) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, origin.scalar_type(), "einsumMg", [&] {
    EinsumMg myEinsumMg(subscripts, input_0, input_1);
    if (!myEinsumMg.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }
    bool ret = myEinsumMg.execute<scalar_t>(input_0, input_1, output);
    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("einsum", &einsum, "Einsum");
  m.def("getEinsumOutputShape", &getEinsumOutputShape, "getEinsumOutputShape");
  m.def("einsumV2", &einsumV2, "EinsumV2");

  pybind11::class_<TensorMg>(m, "TensorMg")
        .def(pybind11::init<const std::vector<int64_t> &>())
        .def("getNumModes", &TensorMg::getNumModes)
        .def("setNumModes", &TensorMg::setNumModes)
        .def("getBlockDevices", &TensorMg::getBlockDevices)
        .def("getDeviceCount", &TensorMg::getDeviceCount)
        .def("getExtent", &TensorMg::getExtent)
        .def("getBlockSize", &TensorMg::getBlockSize)
        .def("getData", &TensorMg::getData)
        .def("getRemainingDevices", &TensorMg::getRemainingDevices)
        ;
  
    m.def("init", &init, "init devices");
    m.def("toTensor", &toTensor, "toTensor");
    m.def("fromTensor", &fromTensor, "fromTensor");
    m.def("getOutputShapeMg", &getOutputShapeMg, "getOutputShapeMg");
    m.def("einsumMg", &einsumMg, "einsumMg");
}
