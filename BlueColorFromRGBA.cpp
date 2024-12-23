#include <iostream>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>

int main()
{
    // We know the data is 500,000,000 bytes = 125,000,000 RGBA pixels
    static const size_t NUM_PIXELS = 125000000ULL;
    static const size_t IN_SIZE    = NUM_PIXELS * 4ULL; // 500,000,000
    static const size_t OUT_SIZE   = NUM_PIXELS;        // 125,000,000

    //--------------------------------------------------------------------------
    // 1. Memory-map STDIN (fd = 0) for reading
    //--------------------------------------------------------------------------
    {
        // Optional: verify the size from fstat. Not strictly necessary
        // if the data is guaranteed to be exactly 500,000,000 bytes.
        struct stat st;
        if (fstat(STDIN_FILENO, &st) != 0) {
            std::cerr << "fstat on STDIN failed\n";
            return 1;
        }
        if (static_cast<size_t>(st.st_size) < IN_SIZE) {
            std::cerr << "STDIN has fewer bytes than expected.\n";
            return 1;
        }
    }

    void* inPtr = mmap(nullptr, IN_SIZE, PROT_READ, MAP_PRIVATE, /*fd*/0, 0);
    if (inPtr == MAP_FAILED) {
        std::cerr << "mmap failed for input\n";
        return 1;
    }

    // Advise the kernel we will read sequentially.
    madvise(inPtr, IN_SIZE, MADV_SEQUENTIAL);

    //--------------------------------------------------------------------------
    // 2. Memory-map an anonymous region for output of size OUT_SIZE
    //    so we can do one single write at the end
    //--------------------------------------------------------------------------
    void* outPtr = mmap(nullptr, OUT_SIZE, PROT_READ | PROT_WRITE, 
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (outPtr == MAP_FAILED) {
        std::cerr << "mmap failed for output\n";
        munmap(inPtr, IN_SIZE);
        return 1;
    }

    //--------------------------------------------------------------------------
    // 3. SSE-based extraction of the Blue channel
    //    RGBA pixel layout: R=byte0, G=byte1, B=byte2, A=byte3
    //
    //    We process 4 pixels (16 bytes) at a time using _mm_shuffle_epi8.
    //--------------------------------------------------------------------------
    auto*  input  = reinterpret_cast<const uint8_t*>(inPtr);
    auto*  output = reinterpret_cast<uint8_t*>(outPtr);

    // This mask picks out bytes 2, 6, 10, 14 (the Blue channels) from a
    // 16-byte chunk, and places them into the low 4 bytes of the SSE register.
    // Bytes marked 0x80 (or -1) are ignored/zeroed out in the result.
    __m128i shuffle_mask = _mm_setr_epi8(
        2, 6, 10, 14,         // B of each 4-byte pixel
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80
    );

    // We'll iterate in steps of 16 bytes from the input = 4 pixels.
    // Each iteration produces 4 output bytes (the Blue components).
    const size_t stepBytes = 16; // 4 RGBA pixels at a time
    const size_t vecIters  = IN_SIZE / stepBytes; // should be 125000000 / 4 = 31250000
                                                 // but let's compute it systematically.

    size_t i = 0;
    for (size_t it = 0; it < vecIters; ++it)
    {
        // Load 16 bytes from input
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));

        // Shuffle to extract only the Blue bytes into the low 4 bytes of 'res'
        __m128i res = _mm_shuffle_epi8(v, shuffle_mask);

        // Store the low 4 bytes into output
        // _mm_storeu_si32 writes only the first 4 bytes of the SSE register
        _mm_storeu_si32(reinterpret_cast<void*>(output + (i >> 2)), res);

        i += stepBytes;
    }

    // In this problem, the total number of pixels (125e6) is divisible by 4,
    // so there is no leftover to handle. If leftover were possible, you'd
    // handle it here with a scalar loop.

    //--------------------------------------------------------------------------
    // 4. Write everything to STDOUT in a single call
    //--------------------------------------------------------------------------
    // Just do a single write of 125,000,000 bytes from outPtr
    {
        size_t bytesToWrite = OUT_SIZE;
        uint8_t* writePtr   = reinterpret_cast<uint8_t*>(outPtr);
        while (bytesToWrite > 0)
        {
            ssize_t wret = ::write(STDOUT_FILENO, writePtr, bytesToWrite);
            if (wret < 0) {
                std::cerr << "write() to STDOUT failed\n";
                break;
            }
            bytesToWrite -= wret;
            writePtr     += wret;
        }
    }

    //--------------------------------------------------------------------------
    // 5. Clean up
    //--------------------------------------------------------------------------
    munmap(inPtr, IN_SIZE);
    munmap(outPtr, OUT_SIZE);

    return 0;
}
