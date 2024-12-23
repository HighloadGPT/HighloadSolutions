// extract_blue.cpp
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <iostream>  // For perror if desired

// Optional compiler hints:
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC target("avx2")
#endif

int main() {
    // We know the input size: 450,000,000 bytes (150,000,000 pixels * 3 bytes per pixel).
    // and the output size: 150,000,000 bytes (one byte per pixel: the Blue channel).

    static const size_t NUM_PIXELS  = 150000000ULL;
    static const size_t INPUT_SIZE  = NUM_PIXELS * 3ULL; // 450,000,000
    static const size_t OUTPUT_SIZE = NUM_PIXELS;        // 150,000,000

    // Map stdin (file descriptor 0) into memory (read-only).
    // Using MAP_PRIVATE because we're only reading it.
    const int fd_in = 0; // STDIN
    void* in_ptr = mmap(nullptr, INPUT_SIZE, PROT_READ, MAP_PRIVATE, fd_in, 0);
    if (in_ptr == MAP_FAILED) {
        perror("mmap stdin");
        return 1;
    }

    // Advise the OS that we'll read this sequentially.
    if (madvise(in_ptr, INPUT_SIZE, MADV_SEQUENTIAL) != 0) {
        // Not a fatal error if this fails; just continue
    }

    // Allocate output buffer in RAM (we could also consider mmap for stdout,
    // but it's trickier to set up if stdout is a pipe).
    unsigned char* out_buffer = (unsigned char*) std::malloc(OUTPUT_SIZE);
    if (!out_buffer) {
        perror("malloc");
        munmap(in_ptr, INPUT_SIZE);
        return 1;
    }

    // Extract the Blue channel.
    // The Blue component is at offset +2 within each 3-byte pixel (RGB).
    // We'll unroll the loop by 16 for extra speed.

    const unsigned char* in_data = static_cast<const unsigned char*>(in_ptr);
    unsigned char*       out_data = out_buffer;

    size_t i = 0;
    // Process blocks of 16 pixels at a time.
    // Each pixel is 3 bytes, so 16 pixels = 48 bytes.

    for (; i + 16 <= NUM_PIXELS; i += 16) {
        out_data[ 0] = in_data[ 2];
        out_data[ 1] = in_data[ 5];
        out_data[ 2] = in_data[ 8];
        out_data[ 3] = in_data[11];
        out_data[ 4] = in_data[14];
        out_data[ 5] = in_data[17];
        out_data[ 6] = in_data[20];
        out_data[ 7] = in_data[23];
        out_data[ 8] = in_data[26];
        out_data[ 9] = in_data[29];
        out_data[10] = in_data[32];
        out_data[11] = in_data[35];
        out_data[12] = in_data[38];
        out_data[13] = in_data[41];
        out_data[14] = in_data[44];
        out_data[15] = in_data[47];

        in_data  += 16 * 3;  // skip 48 bytes
        out_data += 16;
    }

    // Process any leftover pixels (if the total wasn't a multiple of 16).
    while (i < NUM_PIXELS) {
        // Blue is the 3rd byte of each 3-byte pixel
        *out_data++ = in_data[2];
        in_data += 3;
        i++;
    }

    // Write everything out in one go to stdout.
    // (If your environment doesn't allow such a large single write,
    //  you could loop until fully written, or use buffered I/O.)
    size_t bytes_written = 0;
    unsigned char* out_ptr = out_buffer;
    while (bytes_written < OUTPUT_SIZE) {
        ssize_t w = write(STDOUT_FILENO, out_ptr + bytes_written,
                          OUTPUT_SIZE - bytes_written);
        if (w < 0) {
            perror("write to stdout");
            std::free(out_buffer);
            munmap(in_ptr, INPUT_SIZE);
            return 1;
        }
        bytes_written += static_cast<size_t>(w);
    }

    // Clean up
    std::free(out_buffer);
    munmap(in_ptr, INPUT_SIZE);

    return 0;
}
