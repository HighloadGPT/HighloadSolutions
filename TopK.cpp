#include <iostream>
#include <queue>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <vector>

static const size_t N = 100'000'000; // Number of 32-bit integers to read
static const size_t K = 100;         // We want the sum of the top-100 greatest numbers

int main() {
    // Try to open "/dev/stdin"
    int fd = open("/dev/stdin", O_RDONLY);

    // If that fails, fallback to fd=0 (stdin)
    if (fd < 0) {
        fd = 0; // File descriptor 0 is standard input
    }

    // We know that there should be N * sizeof(uint32_t) bytes,
    // but let's attempt an mmap. 
    // In many environments, mmap of a pipe or certain STDIN streams may fail:
    size_t totalBytes = N * sizeof(uint32_t);

    void* mmappedData = mmap(nullptr, totalBytes, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (mmappedData == MAP_FAILED) {
        // If mmap failed, revert to a read-based approach:
        // 1) Allocate a buffer of size N * 4
        // 2) Read from STDIN in chunks until done
        std::cerr << "[INFO] mmap of STDIN failed, falling back to read()..." << std::endl;

        std::vector<uint32_t> buffer(N);
        size_t bytesRead = 0;
        size_t toRead = totalBytes;
        uint8_t* writePtr = reinterpret_cast<uint8_t*>(buffer.data());

        while (toRead > 0) {
            ssize_t ret = read(fd, writePtr, toRead);
            if (ret <= 0) {
                std::cerr << "Error or EOF on read() before we got all data.\n";
                return 1;
            }
            bytesRead += ret;
            writePtr += ret;
            toRead   -= ret;
        }

        // We now have all N 32-bit numbers in `buffer`.
        // Process them with a min-heap of size K to find the top 100.

        std::priority_queue<
            uint32_t,
            std::vector<uint32_t>,
            std::greater<uint32_t>
        > topK;

        // Populate initial K
        size_t i = 0;
        for (; i < K; ++i) {
            topK.push(buffer[i]);
        }

        // Process the rest
        for (; i < N; ++i) {
            uint32_t val = buffer[i];
            if (val > topK.top()) {
                topK.push(val);
                if (topK.size() > K) {
                    topK.pop();
                }
            }
        }

        // Sum the top K
        uint64_t sum = 0;
        while (!topK.empty()) {
            sum += topK.top();
            topK.pop();
        }

        std::cout << sum << std::endl;
        return 0;
    }

    // If mmap succeeded, treat mmappedData as an array of uint32_t:
    const uint32_t* data = static_cast<const uint32_t*>(mmappedData);

    // Proceed with the min-heap approach:
    std::priority_queue<
        uint32_t,
        std::vector<uint32_t>,
        std::greater<uint32_t>
    > topK;

    // Fill the heap with the first K elements
    size_t i = 0;
    for (; i < K; ++i) {
        topK.push(data[i]);
    }

    // Process the remainder
    for (; i < N; ++i) {
        uint32_t val = data[i];
        if (val > topK.top()) {
            topK.push(val);
            if (topK.size() > K) {
                topK.pop();
            }
        }
    }

    // Sum the top K
    uint64_t sum = 0;
    while (!topK.empty()) {
        sum += topK.top();
        topK.pop();
    }

    std::cout << sum << std::endl;

    munmap(mmappedData, totalBytes);
    close(fd);

    return 0;
}
