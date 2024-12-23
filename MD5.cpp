#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <immintrin.h>

// Slightly optimized rotate-left using SSE2 intrinsics for 32-bit lanes.
// NOTE: For short single-buffer MD5, the benefit may be modest. But at scale, it can help.
static inline uint32_t rol32_sse2(uint32_t x, int n) {
    // For SSE2, we can do the shift/or manually:
    // x << n | x >> (32-n). We can do it with intrinsics, though overhead
    // of setting up __m128i might overshadow the gain. But let's show how:

    __m128i vx = _mm_set1_epi32(x);
    __m128i left = _mm_slli_epi32(vx, n);
    __m128i right = _mm_srli_epi32(vx, 32 - n);
    __m128i res = _mm_or_si128(left, right);
    // Extract the first 32-bit lane:
    return static_cast<uint32_t>(_mm_cvtsi128_si32(res));
}

// Standard MD5 constants
static const uint32_t MD5_INIT_STATE[4] = {
    0x67452301U, // A
    0xEFCDAB89U, // B
    0x98BADCFEU, // C
    0x10325476U  // D
};

static const uint32_t MD5_K[64] = {
    0xd76aa478U, 0xe8c7b756U, 0x242070dbU, 0xc1bdceeeU,
    0xf57c0fafU, 0x4787c62aU, 0xa8304613U, 0xfd469501U,
    0x698098d8U, 0x8b44f7afU, 0xffff5bb1U, 0x895cd7beU,
    0x6b901122U, 0xfd987193U, 0xa679438eU, 0x49b40821U,
    0xf61e2562U, 0xc040b340U, 0x265e5a51U, 0xe9b6c7aaU,
    0xd62f105dU, 0x02441453U, 0xd8a1e681U, 0xe7d3fbc8U,
    0x21e1cde6U, 0xc33707d6U, 0xf4d50d87U, 0x455a14edU,
    0xa9e3e905U, 0xfcefa3f8U, 0x676f02d9U, 0x8d2a4c8aU,
    0xfffa3942U, 0x8771f681U, 0x6d9d6122U, 0xfde5380cU,
    0xa4beea44U, 0x4bdecfa9U, 0xf6bb4b60U, 0xbebfbc70U,
    0x289b7ec6U, 0xeaa127faU, 0xd4ef3085U, 0x04881d05U,
    0xd9d4d039U, 0xe6db99e5U, 0x1fa27cf8U, 0xc4ac5665U,
    0xf4292244U, 0x432aff97U, 0xab9423a7U, 0xfc93a039U,
    0x655b59c3U, 0x8f0ccc92U, 0xffeff47dU, 0x85845dd1U,
    0x6fa87e4fU, 0xfe2ce6e0U, 0xa3014314U, 0x4e0811a1U,
    0xf7537e82U, 0xbd3af235U, 0x2ad7d2bbU, 0xeb86d391U
};

// Per-round shift amounts
static const uint32_t MD5_SHIFT[64] = {
    7,12,17,22,  7,12,17,22,  7,12,17,22,  7,12,17,22,
    5, 9,14,20,  5, 9,14,20,  5, 9,14,20,  5, 9,14,20,
    4,11,16,23,  4,11,16,23,  4,11,16,23,  4,11,16,23,
    6,10,15,21,  6,10,15,21,  6,10,15,21,  6,10,15,21
};

// Basic MD5 inline functions
inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); }

// Process one 512-bit (64-byte) block
static void md5_transform(uint32_t state[4], const uint8_t block[64]) {
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];

    // Convert input block to 16 32-bit words
    uint32_t X[16];
    for (int i = 0; i < 16; i++) {
        X[i] = (uint32_t) block[i*4 + 0]       |
               (uint32_t) block[i*4 + 1] <<  8 |
               (uint32_t) block[i*4 + 2] << 16 |
               (uint32_t) block[i*4 + 3] << 24;
    }

    // 64 rounds
    for (int i = 0; i < 64; i++) {
        uint32_t f_val, g;
        if (i < 16) {
            f_val = F(b, c, d);
            g = i;
        } else if (i < 32) {
            f_val = G(b, c, d);
            g = (5*i + 1) & 0x0F;
        } else if (i < 48) {
            f_val = H(b, c, d);
            g = (3*i + 5) & 0x0F;
        } else {
            f_val = I(b, c, d);
            g = (7*i) & 0x0F;
        }

        uint32_t temp = d;
        d = c;
        c = b;

        // Instead of normal rotate, try SSE2-based rotation:
        uint32_t tmp_add = a + f_val + X[g] + MD5_K[i];
        tmp_add = tmp_add & 0xffffffffUL; // keep 32 bits
        uint32_t rotated = rol32_sse2(tmp_add, MD5_SHIFT[i]);

        b = b + rotated;
        a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

// A helper to compute MD5 of a buffer
// (handles partial block, padding, etc.)
void md5_compute(const uint8_t *data, size_t length, uint8_t out_digest[16]) {
    uint32_t state[4];
    memcpy(state, MD5_INIT_STATE, sizeof(state));

    // Process full 64-byte blocks
    size_t full_chunks = length / 64;
    for (size_t i = 0; i < full_chunks; i++) {
        md5_transform(state, data + (i * 64));
    }

    // Remaining bytes
    size_t remaining = length % 64;

    // Buffer for final block(s)
    uint8_t final_block[128];
    memset(final_block, 0, sizeof(final_block));
    memcpy(final_block, data + (full_chunks * 64), remaining);

    // Append the 0x80 bit
    final_block[remaining] = 0x80;

    // If not enough room for length (8 bytes) in this block, we’ll need 2 blocks
    // The MD5 message length in bits:
    uint64_t total_bits = (uint64_t)length * 8ULL;

    if (remaining >= 56) {
        // We need to process this block, then another
        md5_transform(state, final_block);
        // Clear for the next block
        memset(final_block, 0, 64);
    }

    // Put length (in bits) at the end of the last block, little-endian
    memcpy(final_block + 56, &total_bits, 8);

    // Final transform
    md5_transform(state, final_block);

    // Output the final state in little-endian
    for (int i = 0; i < 4; i++) {
        out_digest[i*4 + 0] = (uint8_t)( state[i]        & 0xFF);
        out_digest[i*4 + 1] = (uint8_t)((state[i] >>  8) & 0xFF);
        out_digest[i*4 + 2] = (uint8_t)((state[i] >> 16) & 0xFF);
        out_digest[i*4 + 3] = (uint8_t)((state[i] >> 24) & 0xFF);
    }
}

int main() {
    // 1) Stat STDIN to find file size
    struct stat st;
    if (fstat(STDIN_FILENO, &st) < 0) {
        std::cerr << "Error: fstat on STDIN failed.\n";
        return 1;
    }

    // Ensure we actually have a "real file" on STDIN
    // (the problem statement says it's a file, not a pipe).
    if (!S_ISREG(st.st_mode)) {
        std::cerr << "Error: STDIN is not a regular file (mmap may fail).\n";
        return 1;
    }

    size_t fsize = static_cast<size_t>(st.st_size);
    if (fsize == 0) {
        // Edge case: empty file => known MD5 of empty is d41d8cd98f00b204e9800998ecf8427e
        std::cout << "d41d8cd98f00b204e9800998ecf8427e\n";
        return 0;
    }

    // 2) mmap the file from STDIN
    void* mapped = mmap(nullptr, fsize, PROT_READ, MAP_SHARED, STDIN_FILENO, 0);
    if (mapped == MAP_FAILED) {
        std::cerr << "Error: mmap failed.\n";
        return 1;
    }

    // 3) Compute MD5
    uint8_t digest[16];
    md5_compute(static_cast<const uint8_t*>(mapped), fsize, digest);

    // 4) Unmap
    munmap(mapped, fsize);

    // 5) Print MD5 result in hex
    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << std::hex << std::setfill('0');
    for (int i = 0; i < 16; i++) {
        std::cout << std::setw(2) << (unsigned)digest[i];
    }
    std::cout << std::endl;
    std::cout.flags(f);

    return 0;
}
