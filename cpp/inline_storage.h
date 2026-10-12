// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include <array>
#include <cstddef>
#include <vector>

// Avoid allocator traffic for the usual small argument lists, while retaining
// arbitrary argument counts through the heap fallback.
template <typename T, size_t N>
class InlineStorage {
 public:
  explicit InlineStorage(size_t size) {
    if (size > N) overflow_.resize(size);
  }
  T* data() { return overflow_.empty() ? inline_.data() : overflow_.data(); }
  T& operator[](size_t i) { return data()[i]; }
 private:
  std::array<T, N> inline_;
  std::vector<T> overflow_;
};
