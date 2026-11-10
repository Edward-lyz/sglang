#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {
void put_u64(uint8_t* target, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        *target++ = static_cast<uint8_t>(value >> shift);
    }
}

uint64_t checked_add(uint64_t total, uint64_t bytes) {
  const auto limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (bytes > limit || total > limit - bytes) {
    throw py::value_error("request OOB packet exceeds int64 tensor size");
  }
  return total + bytes;
}
}  // namespace

// The exporter pins each buffer against resize for the entire native copy.
// The request receiver owns incoming token arrays until broadcast returns.
torch::Tensor pack(py::bytes metadata, py::list buffers) {
  char* metadata_data = nullptr;
  Py_ssize_t metadata_size = 0;
  if (PyBytes_AsStringAndSize(metadata.ptr(), &metadata_data, &metadata_size) < 0) {
    throw py::error_already_set();
  }
  std::vector<py::buffer_info> exports;
  exports.reserve(buffers.size());
  uint64_t total = checked_add(24, checked_add(0, buffers.size() * 8));
  total = checked_add(total, static_cast<uint64_t>(metadata_size));
  for (auto item : buffers) {
    auto info = py::cast<py::buffer>(item).request();
    if (info.ndim != 1 || info.itemsize != 1 || info.size < 0 ||
        (info.size && info.strides[0] != 1)) {
      throw py::value_error("request OOB buffers must be contiguous byte views");
    }
    total = checked_add(total, static_cast<uint64_t>(info.size));
    exports.push_back(std::move(info));
  }
  torch::Tensor result;
  {
    // No Python calls or reference-count operations inside this region.
    py::gil_scoped_release release;
    result = torch::empty(
        {static_cast<int64_t>(total)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto* output = result.data_ptr<uint8_t>();
    std::memcpy(output, "SGBOOB01", 8);
    put_u64(output + 8, static_cast<uint64_t>(metadata_size));
    put_u64(output + 16, exports.size());
    uint64_t cursor = 24 + exports.size() * 8;
    for (size_t i = 0; i < exports.size(); ++i) {
      put_u64(output + 24 + i * 8, exports[i].size);
    }
    if (metadata_size) {
      std::memcpy(output + cursor, metadata_data, metadata_size);
    }
    cursor += metadata_size;
    for (const auto& info : exports) {
      if (info.size) {
        std::memcpy(output + cursor, info.ptr, info.size);
      }
      cursor += info.size;
    }
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack", &pack);
}
