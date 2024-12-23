/*
   Compile with:
     g++ -O3 -march=native sum_primes_mmap.cpp -o sum_primes_mmap

   This solution tries to:
     1) mmap stdin if it is a regular file.
     2) Fallback to a normal read loop if stdin is a pipe/other non-regular file.
     3) Precompute primes up to 65536 with a sieve.
     4) Sum up 32-bit numbers that are prime.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

// Hint the compiler we want to use AVX2 on Haswell (GCC/Clang)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx2")
#endif

// Generate all primes up to maxN using Sieve of Eratosthenes
static std::vector<uint32_t> build_prime_table(uint32_t maxN) {
    std::vector<bool> is_prime(maxN + 1, true);
    is_prime[0] = false;
    is_prime[1] = false;
    for(uint32_t i = 2; i * i <= maxN; i++) {
        if (is_prime[i]) {
            for(uint32_t j = i * i; j <= maxN; j += i) {
                is_prime[j] = false;
            }
        }
    }

    std::vector<uint32_t> primes;
    primes.reserve(6542); // # of primes up to 65536 is around 6542
    for(uint32_t i = 2; i <= maxN; i++) {
        if(is_prime[i]) {
            primes.push_back(i);
        }
    }
    return primes;
}

// Check primality by trial division using precomputed primes
inline bool is_prime_32(uint32_t n, const std::vector<uint32_t> &primes) {
    if (__builtin_expect(n < 2, 0))
        return false;
    if (n == 2 || n == 3)
        return true;
    if ((n & 1) == 0)  // even and not 2
        return false;

    uint32_t limit = static_cast<uint32_t>(std::sqrt(n));
    for (uint32_t p : primes) {
        if (p > limit) break;
        if ((n % p) == 0) return false;
    }
    return true;
}

int main() {
    // Build the prime table up to 65536 (sqrt of ~2^32).
    static const uint32_t PRIME_MAX = 65536;
    std::vector<uint32_t> primes = build_prime_table(PRIME_MAX);

    // Get file descriptor for stdin
    int fd = fileno(stdin);
    if (fd < 0) {
        std::cerr << "Error: fileno(stdin) failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    // Use fstat to check if it's a regular file (can be memory-mapped)
    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "Error: fstat failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    // We'll assume 1,000,000 numbers => 4,000,000 bytes total
    // But in some cases it might differ; we'll base reading on st_size or fallback.

    uint64_t result = 0; // sum of primes

    // If it's a regular file, try mmap
    if (S_ISREG(st.st_mode)) {
        // Attempt to mmap
        size_t file_size = static_cast<size_t>(st.st_size);
        // Number of 32-bit integers we can read
        size_t count = file_size / sizeof(uint32_t);

        void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            // If mapping fails, fallback to read
            std::cerr << "Warning: mmap() failed: " << std::strerror(errno)
                      << "\nFalling back to buffered read.\n";

            // Fallback read approach: read all bytes via read() or by a loop
            // (Implementation shown below)
        } else {
            const uint32_t* data = reinterpret_cast<const uint32_t*>(mapped);
            for (size_t i = 0; i < count; i++) {
                uint32_t number = data[i];
                if (is_prime_32(number, primes)) {
                    result += number;
                }
            }
            munmap(mapped, file_size);

            // Print result and exit success
            std::cout << result << std::endl;
            return 0;
        }
    }

    // Fallback approach: read the data from fd in a loop
    // (Works for pipes, or if mmap failed)
    {
        static const size_t BUFSIZE = 1 << 16; // 64 KB chunk
        alignas(16) unsigned char buffer[BUFSIZE];
        
        ssize_t nread;
        // Because each number is 4 bytes, we must handle partial chunks carefully.
        size_t leftover = 0;  // leftover bytes that don't align to a uint32_t boundary
        uint8_t temp[4];      // store partial bytes if needed

        while ((nread = ::read(fd, buffer + leftover, BUFSIZE - leftover)) > 0) {
            nread += leftover;  // total bytes in buffer now
            size_t usable = (nread / 4) * 4; // # of bytes that align to 4
            const uint32_t* data32 = reinterpret_cast<const uint32_t*>(buffer);

            // Process in 4-byte increments
            for (size_t i = 0; i < usable; i += 4) {
                uint32_t number;
                // Copy 4 bytes (little-endian)
                std::memcpy(&number, &buffer[i], 4);
                if (is_prime_32(number, primes)) {
                    result += number;
                }
            }

            // If there's leftover bytes, move them to front
            leftover = nread - usable;
            if (leftover > 0) {
                std::memmove(buffer, buffer + usable, leftover);
            }
        }
        if (nread < 0) {
            std::cerr << "Error: read() failed: " << std::strerror(errno) << "\n";
            return 1;
        }

        // If we still have leftover bytes that might form a partial number,
        // handle them. Normally, we expect multiples of 4 bytes total,
        // so this might not happen in a well-formed stream, but just in case:
        if (leftover > 0) {
            if (leftover == 4) {
                // One last number
                uint32_t number;
                std::memcpy(&number, buffer, 4);
                if (is_prime_32(number, primes)) {
                    result += number;
                }
            }
            // else leftover < 4 => incomplete last number, we can ignore or handle error
        }
    }

    // Done reading in fallback mode
    std::cout << result << std::endl;
    return 0;
}
