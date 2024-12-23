#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <cstdint>

static constexpr size_t N = 100'000'000;

int main() {
    // We will mmap standard input (file descriptor = 0).
    struct stat st;
    if (fstat(STDIN_FILENO, &st) < 0) {
        perror("fstat");
        return 1;
    }

    // Optional sanity check: ensure the file is at least large enough
    // to contain N 32-bit integers (400 million bytes).
    if (static_cast<size_t>(st.st_size) < N * sizeof(uint32_t)) {
        std::cerr << "Error: Not enough data for " << N 
                  << " uint32_t values.\n";
        return 1;
    }

    // Map the entire input file into memory (read+write in private copy).
    // MAP_PRIVATE allows us to do in-place partitioning without affecting
    // the underlying data. On Linux, PROT_READ|PROT_WRITE is permissible here
    // for an in-place nth_element if needed. 
    void* addr = mmap(nullptr,
                      st.st_size,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE,
                      STDIN_FILENO,
                      0);
    if (addr == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // Treat this mapped region as an array of uint32_t.
    auto* dataPtr = static_cast<uint32_t*>(addr);

    // Use nth_element to place the median element at index N/2.
    // This is typically O(N) on average, much faster than full sorting.
    std::nth_element(dataPtr, dataPtr + (N / 2), dataPtr + N);

    // The median is now at position N/2.
    uint32_t medianVal = dataPtr[N / 2];

    // Print it
    std::cout << medianVal << "\n";

    // Unmap the memory before exiting.
    munmap(addr, st.st_size);
    return 0;
}
