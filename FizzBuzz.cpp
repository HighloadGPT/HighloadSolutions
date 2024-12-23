#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// A fast routine to convert a 32-bit unsigned integer to decimal string.
// Returns a pointer to the character after the last written digit.
static inline char* u32toa(uint32_t x, char* out) {
    // Temporary buffer to construct the number in reverse
    // 10 digits are enough for 32-bit (max 4294967295).
    char buf[11];
    char* p = buf;
    do {
        *p++ = char('0' + (x % 10));
        x /= 10;
    } while (x != 0);

    // Now reverse it into 'out'
    // p points one past the last digit written
    while (p != buf) {
        *out++ = *--p;
    }
    return out;
}

// Decide how large our output buffer should be.
// We'll flush to stdout once this is near full.
// 8MB is a decent chunk; you can tune as desired.
static const size_t OUTBUF_SIZE = 8UL * 1024UL * 1024UL;

int main() {
    // Get size of stdin via fstat
    struct stat st;
    if (fstat(STDIN_FILENO, &st) < 0) {
        perror("fstat failed on stdin");
        return 1;
    }

    // If size is zero or invalid for some reason, exit early
    if (st.st_size == 0) {
        return 0;
    }

    // Map the entire input into memory
    // Note: if stdin is not seekable (e.g., a pure pipe),
    // fstat may not return a valid size. Adjust accordingly
    // if your environment differs.
    void* mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, STDIN_FILENO, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap on stdin failed");
        return 1;
    }

    // Each number is 4 bytes (little-endian uint32_t)
    size_t numElements = st.st_size / sizeof(uint32_t);
    auto* dataPtr = static_cast<const unsigned char*>(mapped);

    // Prepare a large output buffer
    char* outBuf = new char[OUTBUF_SIZE];
    char* outPtr = outBuf;

    // A lambda to flush the current buffer to stdout
    auto flushOutput = [&](bool finalFlush = false) {
        size_t bytesToWrite = outPtr - outBuf;
        if (bytesToWrite > 0) {
            ::write(STDOUT_FILENO, outBuf, bytesToWrite);
        }
        // Reset outPtr for next chunk
        outPtr = outBuf;
        // If it's a final flush, we won't continue writing
        if (finalFlush) {
            // no-op here, but could close or do anything else if needed
        }
    };

    // We'll parse each 4-byte chunk as a little-endian uint32_t,
    // then do the fizzbuzz logic and store strings in our outBuf.
    // We check for near-full buffer to flush in chunks.
    for (size_t i = 0; i < numElements; i++) {
        // Read 4 bytes as little-endian
        // (On little-endian systems, this reinterpret_cast is fine directly;
        //  to be fully portable to big-endian, you'd reorder bytes.)
        uint32_t n;
        memcpy(&n, dataPtr + i*4, 4);

        // FizzBuzz checks
        if (n % 15 == 0) {
            // "FizzBuzz\n"
            static const char fbStr[] = "FizzBuzz\n";
            memcpy(outPtr, fbStr, sizeof(fbStr) - 1);
            outPtr += sizeof(fbStr) - 1;
        }
        else if (n % 3 == 0) {
            // "Fizz\n"
            static const char fStr[] = "Fizz\n";
            memcpy(outPtr, fStr, sizeof(fStr) - 1);
            outPtr += sizeof(fStr) - 1;
        }
        else if (n % 5 == 0) {
            // "Buzz\n"
            static const char bStr[] = "Buzz\n";
            memcpy(outPtr, bStr, sizeof(bStr) - 1);
            outPtr += sizeof(bStr) - 1;
        }
        else {
            // number -> string + newline
            outPtr = u32toa(n, outPtr);
            *outPtr++ = '\n';
        }

        // If the buffer is almost full, flush
        // (choose a threshold less than OUTBUF_SIZE to avoid overflows).
        if (size_t(outPtr - outBuf) > (OUTBUF_SIZE - 128)) {
            flushOutput();
        }
    }

    // Final flush
    flushOutput(/*finalFlush=*/true);

    // Clean up
    munmap(mapped, st.st_size);
    delete[] outBuf;

    return 0;
}
