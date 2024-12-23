#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <iostream>

// ---------------------------------------------------------------------------
// number_crc(n):
//    Sum of each decimal digit's ASCII code multiplied by its position,
//    where the leftmost digit has position = 0.
//
//    Example: n = 123
//      decimal string: "123"
//      digits: '1'(ASCII=49), '2'(50), '3'(51)
//      result = 49*0 + 50*1 + 51*2 = 0 + 50 + 102 = 152
// ---------------------------------------------------------------------------
inline uint64_t number_crc(uint32_t x) {
    // Handle zero explicitly (single digit '0')
    // '0' * position0 = 48 * 0 = 0
    if (x == 0) {
        return 0;
    }

    // Temporary buffer for up to 10 digits (max 32-bit decimal has 10 digits)
    // We'll collect digits from right to left.
    // Then we will sum them from left to right.
    char buf[10];
    int len = 0;

    // Repeatedly extract decimal digits from the right
    while (x > 0) {
        buf[len++] = char('0' + (x % 10));
        x /= 10;
    }
    // 'len' is now the count of digits in the number
    // buf[0] is the rightmost digit; buf[len-1] is the leftmost.

    // We need to accumulate s[i]*i with i=0 as the leftmost digit.
    // That means position = (len - 1 - index_in_buf).
    // So if the leftmost digit is at buf[len-1], it gets position = 0.
    // Next digit to the right gets position = 1, etc.

    // Instead of reversing the array to build the string from left to right,
    // we'll directly sum with the reversed position index.
    uint64_t ret = 0;
    for (int i = 0; i < len; i++) {
        // buf[len - 1 - i] is the *leftmost* digit going left to right
        // i is the position: leftmost => position 0, next => position 1, ...
        ret += (static_cast<unsigned char>(buf[len - 1 - i]) * i);
    }
    return ret;
}

int main() {
    // Map stdin into memory
    //   1. fstat() stdin to get size
    //   2. mmap() the data
    //   3. Interpret as array of 32-bit values in little-endian
    //   4. Accumulate result

    // We expect 250,000,000 numbers => 1,000,000,000 bytes total
    // but we'll do a generic approach that uses the file size from fstat.

    struct stat sb;
    if (fstat(STDIN_FILENO, &sb) < 0) {
        std::perror("fstat on stdin");
        return 1;
    }
    if (sb.st_size % 4 != 0) {
        std::cerr << "Input size not a multiple of 4 bytes!\n";
        return 2;
    }

    size_t fileSize = sb.st_size;
    size_t count    = fileSize / sizeof(uint32_t);

    void* mapped = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, STDIN_FILENO, 0);
    if (mapped == MAP_FAILED) {
        std::perror("mmap of stdin");
        return 3;
    }

    uint64_t res = 0;
    // Process all 32-bit values
    const uint32_t* ptr = static_cast<const uint32_t*>(mapped);

    for (size_t i = 0; i < count; i++) {
        // Because the stream is little-endian, direct reinterpret_cast is fine on x86.
        // If you were on a big-endian system, you'd need to byte-swap.
        uint32_t val = ptr[i];
        res += number_crc(val);
    }

    // Clean up
    munmap(mapped, fileSize);

    // Print the final CRC result
    std::cout << res << std::endl;
    return 0;
}
