#include <immintrin.h>     // For AVX2 intrinsics
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <cassert>

// Helper function: convert ASCII digits [start, end) to a 64-bit number.
// We assume that the substring contains only characters '0'..'9'.
static inline uint64_t parseNumber(const char* start, const char* end) {
    uint64_t val = 0;
    for (const char* p = start; p < end; ++p) {
        val = val * 10 + static_cast<unsigned>(*p - '0');
    }
    return val;
}

int main() {
    // Get size of stdin
    struct stat st;
    if (fstat(STDIN_FILENO, &st) < 0) {
        std::perror("fstat");
        return 1;
    }
    off_t dataSize = st.st_size;
    if (dataSize == 0) {
        // Nothing to read
        std::cout << 0 << "\n";
        return 0;
    }

    // MMAP stdin
    char* data = static_cast<char*>(
        mmap(nullptr, dataSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, STDIN_FILENO, 0));
    if (data == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }

    // We will accumulate the sum in a 64-bit integer (ignore overflow if it happens).
    // The problem statement says "In case of an integer overflow, just ignore it."
    // So we do not do any special handling beyond using a 64-bit type.
    uint64_t totalSum = 0;

    // Pointers for parsing
    const char* ptr = data;
    const char* end = data + dataSize;

    // AVX2 compare target for '\n'
    const __m256i newlineVec = _mm256_set1_epi8('\n');

    // We'll store the start of the current "line" (current number string).
    const char* lineStart = ptr;

    // Process in large chunks until near the end
    while (ptr + 32 <= end) {
        // Load 32 bytes
        __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));

        // Compare block against '\n'
        __m256i cmp = _mm256_cmpeq_epi8(block, newlineVec);

        // Create a bitmask where each byte is 1 if comparison matched else 0
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));

        if (mask == 0) {
            // No newline in these 32 bytes, continue
            ptr += 32;
        } else {
            // At least one newline in this block
            // We’ll find each newline's position.  Each set bit in 'mask'
            // corresponds to a newline.  We handle them in ascending order.
            while (mask != 0) {
                // Get position of the rightmost (lowest index) set bit
                // _tzcnt_u32 returns the index of the trailing set bit
                unsigned pos = __builtin_ctz(mask);
                // parse the integer from lineStart..(ptr+pos)
                totalSum += parseNumber(lineStart, ptr + pos);
                // skip the newline
                lineStart = ptr + pos + 1;

                // Clear that bit
                mask &= (mask - 1);
            }
            // Advance pointer past the 32 bytes we just scanned
            ptr += 32;
        }
    }

    // Now handle the remainder (less than 32 bytes) with a simple scalar loop
    while (ptr < end) {
        if (*ptr == '\n') {
            // parse the line we've collected so far
            totalSum += parseNumber(lineStart, ptr);
            lineStart = ptr + 1;
        }
        ++ptr;
    }

    // If the last line did not end with a newline, parse it
    if (lineStart < end) {
        totalSum += parseNumber(lineStart, end);
    }

    // Cleanup
    munmap(data, dataSize);

    // Output the result
    std::cout << totalSum << "\n";
    return 0;
}
