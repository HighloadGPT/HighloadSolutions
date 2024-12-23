#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>

#include <iostream>
#include <cstdint>
#include <cstdio>

static inline unsigned popcount32(uint32_t x) {
    // Depending on your compiler, you can use __builtin_popcount
    // or __builtin_popcountl, or roll your own. For GCC/Clang:
    return __builtin_popcount(x);
}

int main() {
    // Get size of data from stdin
    struct stat st;
    if (fstat(0, &st) < 0) {
        std::perror("fstat failed");
        return 1;
    }
    if (st.st_size == 0) {
        std::cout << 0 << std::endl;
        return 0;
    }

    // Map stdin into memory
    void* mappedData = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, 0, 0);
    if (mappedData == MAP_FAILED) {
        std::perror("mmap failed");
        return 1;
    }

    // We expect 8-bit values; st.st_size is the number of bytes
    const uint8_t* data = static_cast<const uint8_t*>(mappedData);
    size_t size = static_cast<size_t>(st.st_size);

    // AVX2-based counting
    const __m256i target = _mm256_set1_epi8(127);
    uint64_t count = 0;

    // Process 32 bytes at a time
    const size_t chunkSize = 32;
    size_t i = 0;
    size_t limit = size - (size % chunkSize);

    for (; i < limit; i += chunkSize) {
        // Load 256 bits
        __m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        // Compare each byte to 127
        __m256i cmp = _mm256_cmpeq_epi8(vec, target);
        // Create a bit mask from comparisons
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));
        // Count how many bytes were equal to 127 in this chunk
        count += popcount32(mask);
    }

    // Process leftover bytes (if size is not multiple of 32)
    for (; i < size; i++) {
        if (data[i] == 127) {
            ++count;
        }
    }

    // Clean up
    munmap(mappedData, st.st_size);

    // Print result
    std::cout << count << std::endl;

    return 0;
}
