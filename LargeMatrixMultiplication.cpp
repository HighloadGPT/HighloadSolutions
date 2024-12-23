#include <immintrin.h>   // For AVX2 intrinsics
#include <sys/mman.h>    // For mmap
#include <sys/stat.h>    // For fstat
#include <fcntl.h>       // For open
#include <unistd.h>      // For close, write
#include <stdint.h>      // For uint32_t
#include <stdio.h>       // For perror, etc.
#include <stdlib.h>      // For exit

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
static const int N = 2000;         // Matrix dimension
static const int BLOCK_SIZE = 32;  // Block size for tiling (adjust as needed)

// -----------------------------------------------------------------------------
// We'll map the input file (STDIN) which contains two NxN matrices, each
// in row-major order, 32-bit little-endian integers. Then output NxN product.
//
// Memory usage (approx):
//   A, B, C ~ 2000 * 2000 * 4 bytes = 16,000,000 bytes each
//   total ~ 48 MB plus overhead
// -----------------------------------------------------------------------------

// Transpose matrix B (N x N) into Btrans (also N x N).
// This allows us to access B by rows in the multiply step.
static void transposeB(const uint32_t* __restrict B,
                       uint32_t* __restrict Btrans) 
{
    for (int i = 0; i < N; i++) {
        const uint32_t* rowB = &B[i * N];
        for (int j = 0; j < N; j++) {
            Btrans[j * N + i] = rowB[j];
        }
    }
}

// Perform blocked (tiled) matrix multiplication with AVX2 intrinsics.
//   C = A * B  (where B is transposed into Btrans to improve locality).
static void multiplyBlockedAVX2(const uint32_t* __restrict A,
                                const uint32_t* __restrict Btrans,
                                      uint32_t* __restrict C)
{
    // Zero-initialize C, as we'll accumulate sums into it.
    // This large memset is also a bandwidth hog, but straightforward.
    // We could also do partial/blocked zero if we like.
    for (int i = 0; i < N * N; i++) {
        C[i] = 0;
    }

    // We will do 3-level loop over blocks (I, J, K),
    // then do local loops over [I..I+BS) x [J..J+BS].
    // The typical formula is:
    //     C[i,j] += A[i,k] * B[k,j]
    // But we have transposed B => Btrans[j,k] = B[k,j].
    // So we do:
    //     C[i,j] += A[i,k] * Btrans[j,k]

    for (int iBlock = 0; iBlock < N; iBlock += BLOCK_SIZE) {
        for (int jBlock = 0; jBlock < N; jBlock += BLOCK_SIZE) {
            for (int kBlock = 0; kBlock < N; kBlock += BLOCK_SIZE) {

                int iMax = (iBlock + BLOCK_SIZE < N) ? iBlock + BLOCK_SIZE : N;
                int jMax = (jBlock + BLOCK_SIZE < N) ? jBlock + BLOCK_SIZE : N;
                int kMax = (kBlock + BLOCK_SIZE < N) ? kBlock + BLOCK_SIZE : N;

                for (int i = iBlock; i < iMax; i++) {

                    // pointer to A row i
                    const uint32_t* Arow = &A[i * N];

                    for (int j = jBlock; j < jMax; j++) {
                        
                        // We'll accumulate in a 32-bit integer,
                        // but we can do partial sums in a 256-bit vector
                        // for the 'k' dimension in chunks of 8.
                        // We will accumulate horizontally into a sum at the end.
                        __m256i vsum = _mm256_setzero_si256();

                        // For k in [kBlock..kMax), do in steps of 8
                        int k = kBlock;
                        for (; k + 8 <= kMax; k += 8) {
                            // Load 8 elements from Arow (A[i, k..k+7])
                            __m256i va = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(&Arow[k]));

                            // Load 8 elements from Btrans row j 
                            // (which is B[k..k+7, j] in normal orientation),
                            // but in Btrans it is Btrans[j, k..k+7].
                            const uint32_t* Bptr = &Btrans[j * N + k];
                            __m256i vb = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(Bptr));
                            
                            // Multiply 32-bit integers in va and vb, producing
                            // 32-bit results. Then add to vsum.
                            __m256i vmul = _mm256_mullo_epi32(va, vb);
                            vsum = _mm256_add_epi32(vsum, vmul);
                        }

                        // Now reduce vsum horizontally into a single 32-bit integer
                        // sum of the 8 lanes.
                        __m128i vsumHigh = _mm256_extracti128_si256(vsum, 1); // high 128
                        __m128i vsumLow  = _mm256_castsi256_si128(vsum);       // low 128
                        __m128i vsum128  = _mm_add_epi32(vsumHigh, vsumLow);
                        // vsum128 has 4 lanes. reduce further
                        __m128i vsum64 = _mm_hadd_epi32(vsum128, vsum128); // sum pairs
                        vsum64 = _mm_hadd_epi32(vsum64, vsum64); // sum pairs again
                        int partialSum = _mm_cvtsi128_si32(vsum64);

                        // For leftover k
                        for (; k < kMax; k++) {
                            partialSum += Arow[k] * Btrans[j * N + k];
                        }

                        // Add partialSum to C[i,j]
                        C[i * N + j] += partialSum;
                    }
                }
            }
        }
    }
}

int main()
{
    // -------------------------------------------------------------------------
    // 1) Map STDIN which has 2*N*N*sizeof(uint32_t) bytes = 2*2000*2000*4
    // -------------------------------------------------------------------------
    const size_t matrixSizeBytes = (size_t)N * N * sizeof(uint32_t);
    const size_t totalSizeBytes  = 2 * matrixSizeBytes; // A + B

    // Use file descriptor 0 (STDIN)
    int fd = 0; 
    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        exit(1);
    }
    if ((size_t)st.st_size < totalSizeBytes) {
        fprintf(stderr, "ERROR: STDIN not large enough for 2 matrices.\n");
        exit(1);
    }

    // Map the input
    uint32_t* inputData = (uint32_t*)mmap(/*addr=*/nullptr,
                                          /*length=*/totalSizeBytes,
                                          PROT_READ,
                                          MAP_PRIVATE,
                                          fd,
                                          /*offset=*/0);
    if (inputData == MAP_FAILED) {
        perror("mmap for inputData");
        exit(1);
    }

    // A is in [0 .. N*N-1], B is in [N*N .. 2*N*N-1]
    const uint32_t* A = inputData;
    const uint32_t* B = &inputData[N * N];

    // -------------------------------------------------------------------------
    // 2) Allocate memory for Btrans and C
    //    (We could also mmap an output buffer for C, but let's just allocate.)
    // -------------------------------------------------------------------------
    uint32_t* Btrans = (uint32_t*)aligned_alloc(32, matrixSizeBytes);
    if (!Btrans) {
        perror("aligned_alloc for Btrans");
        exit(1);
    }

    uint32_t* C = (uint32_t*)aligned_alloc(32, matrixSizeBytes);
    if (!C) {
        perror("aligned_alloc for C");
        exit(1);
    }

    // -------------------------------------------------------------------------
    // 3) Transpose B -> Btrans
    // -------------------------------------------------------------------------
    transposeB(B, Btrans);

    // -------------------------------------------------------------------------
    // 4) Multiply A * B  => C   (using blocking + AVX2)
    // -------------------------------------------------------------------------
    multiplyBlockedAVX2(A, Btrans, C);

    // -------------------------------------------------------------------------
    // 5) Write C to STDOUT as raw bytes
    // -------------------------------------------------------------------------
    // Because output is large, you may consider using mmap on STDOUT as well.
    // Here we do a single big write call for brevity.
    size_t bytesLeft = matrixSizeBytes;
    const char* outPtr = reinterpret_cast<const char*>(C);
    while (bytesLeft > 0) {
        ssize_t w = write(STDOUT_FILENO, outPtr, bytesLeft);
        if (w < 0) {
            perror("write");
            exit(1);
        }
        bytesLeft -= w;
        outPtr    += w;
    }

    // -------------------------------------------------------------------------
    // 6) Cleanup
    // -------------------------------------------------------------------------
    munmap(inputData, totalSizeBytes);
    free(Btrans);
    free(C);

    return 0;
}
