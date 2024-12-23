#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#include <cassert>
#include <cerrno>

// We have exactly 500,000 bytes of input:
//   - First  250,000 bytes = integer A, in little-endian
//   - Second 250,000 bytes = integer B, in little-endian
// We must output 500,000 bytes (the product), in little-endian.

static constexpr size_t N_BYTES = 250'000;
static constexpr size_t N_WORDS = N_BYTES / sizeof(uint64_t); // = 31,250
// The result can be up to 500,000 bytes (N_WORDS * 2 words).

int main() {
    // 1) Allocate buffers via mmap
    //    We'll store A and B in separate 64-bit arrays.
    //    We'll store the result R in a 64-bit array of size 2*N_WORDS.
    //    The input is read from STDIN into a temporary buffer in bytes,
    //    and we reinterpret that buffer as two 64-bit arrays.

    // Memory for reading the entire 500,000 bytes (A + B):
    void* inBuf = mmap(nullptr, 2 * N_BYTES,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, 
                       -1, 0);
    if (inBuf == MAP_FAILED) {
        _exit(1);
    }

    // Memory for result (500,000 bytes = 2*N_WORDS * 8):
    void* outBuf = mmap(nullptr, 2 * N_BYTES,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1, 0);
    if (outBuf == MAP_FAILED) {
        _exit(2);
    }

    // Read entire 500,000 bytes from STDIN:
    {
        size_t totalRead = 0;
        while (totalRead < 2 * N_BYTES) {
            ssize_t got = ::read(STDIN_FILENO,
                                 static_cast<uint8_t*>(inBuf) + totalRead,
                                 2 * N_BYTES - totalRead);
            if (got < 0) {
                _exit(3); 
            }
            if (got == 0) {
                // Premature EOF
                break;
            }
            totalRead += got;
        }
        // Not checking if we got exactly 500k, but we assume the task guarantees it.
    }

    // Pointers to A and B in 64-bit form:
    uint64_t* A = static_cast<uint64_t*>(inBuf);
    // B starts right after A in that mapped region:
    uint64_t* B = reinterpret_cast<uint64_t*>(
                      static_cast<uint8_t*>(inBuf) + N_BYTES);

    // Result array (2*N_WORDS) in 64-bit form:
    uint64_t* R = static_cast<uint64_t*>(outBuf);

    // We must zero out the result array:
    // (this is significantly cheaper than re-initializing for each partial sum)
    for (size_t i = 0; i < 2 * N_WORDS; i++) {
        R[i] = 0ULL;
    }

    // 2) Multiply A by B, store into R in little-endian 64-bit words.
    //    We'll do the naive O(N_WORDS^2) approach, but we avoid
    //    re-zeroing partial sums for each i. Instead, we accumulate directly.

    // Some unrolling factor:
    constexpr size_t UNROLL = 4;
    
    for (size_t i = 0; i < N_WORDS; i++) {
        // bVal is the 64-bit chunk of B
        const uint64_t bVal = B[i];
        
        __uint128_t carry = 0;  // 128-bit to hold any overflow
        size_t j = 0;

        // Do multiples of UNROLL:
        for (; j + (UNROLL-1) < N_WORDS; j += UNROLL) {
            // Unroll 4 multiplications
#pragma GCC unroll 4
            for (size_t k = 0; k < UNROLL; k++) {
                __uint128_t mul = ( __uint128_t )A[j + k] * bVal
                                  + R[i + j + k]
                                  + carry;
                R[i + j + k] = (uint64_t) mul;
                carry        = mul >> 64;
            }
        }
        // Remainder of the loop
        for (; j < N_WORDS; j++) {
            __uint128_t mul = ( __uint128_t )A[j] * bVal
                              + R[i + j]
                              + carry;
            R[i + j] = (uint64_t) mul;
            carry    = mul >> 64;
        }

        // Now add final carry to R[i + N_WORDS], if any:
        // i + N_WORDS < 2*N_WORDS, so it's valid.
        if (carry != 0) {
            __uint128_t tmp = ( __uint128_t )R[i + N_WORDS] + carry;
            R[i + N_WORDS] = (uint64_t) tmp;
            // If there's another carry-out, it can only propagate
            // further in extremely rare corner cases, but the max
            // we can have is 2*N_WORDS = 62500 64-bit words of result,
            // which is 500k bytes, so we do not need to go beyond that.
            // The next index is i+N_WORDS+1, but problem states exactly
            // 500k output. So we ignore further carries.
        }
    }

    // 3) Write exactly 500,000 bytes of result to STDOUT:
    {
        size_t totalWritten = 0;
        while (totalWritten < 2 * N_BYTES) {
            ssize_t done = ::write(STDOUT_FILENO,
                                   static_cast<uint8_t*>(outBuf) + totalWritten,
                                   (2 * N_BYTES) - totalWritten);
            if (done < 0) {
                _exit(4);
            }
            totalWritten += done;
        }
    }

    // 4) Cleanup
    munmap(inBuf, 2 * N_BYTES);
    munmap(outBuf, 2 * N_BYTES);

    return 0;
}
