#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <immintrin.h>

// ================== Configuration ==================

// Maximum token length (plus 1 for null terminator).
// Given in the problem: up to 16 chars.
static const int MAX_TOKEN_LEN = 16;
// We will store tokens in a fixed-size 16-byte buffer, plus we keep the length separately.

// Expected maximum number of unique tokens. 
// We'll choose a table size ~2X to reduce collisions.
static const size_t EXPECTED_MAX_UNIQUE = 1'000'000ULL; 
static const size_t TABLE_SIZE = 1ULL << 21; // ~2 million buckets (2^21)

// ================== FNV-1a 64-bit Hash ==================
// For short tokens (<=16 bytes), FNV-1a is reasonably fast and easy to implement.
inline uint64_t fnv1a_64(const char* data, int len)
{
    static const uint64_t FNV_OFFSET_BASIS = 1469598103934665603ULL;
    static const uint64_t FNV_PRIME        = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET_BASIS;
    for (int i = 0; i < len; ++i) {
        hash ^= (unsigned char)data[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

// ================== Token Table Entry ==================
struct TokenEntry {
    uint64_t hashVal;   // 64-bit hash of the token
    char     token[16]; // Exactly 16 bytes for the token data (no null terminator needed if we store length).
    int      length;    // Actual length of the token, 0 means "empty slot"

    TokenEntry() : hashVal(0), length(0) {}
};

// ================== Global Table ==================
static TokenEntry* g_table = nullptr;

// Returns true if 'needle' (length nLen) matches the token in 'slot'.
inline bool tokensEqualSSE(const TokenEntry& slot, const char* needle, int nLen) {
    if (slot.length != nLen) {
        return false;
    }
    // For up to 16 bytes, we can use SSE/AVX to compare quickly.
    // Load 16 bytes from slot.token and from needle (zero-padded).
    __m128i slotData  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(slot.token));
    
    // We must build a 16-byte buffer from needle (length nLen) zero-padded
    // in case nLen < 16, to safely compare the entire 128 bits.
    alignas(16) char padded[16] = {0};
    if (nLen > 0) {
        memcpy(padded, needle, nLen);
    }
    __m128i needleData = _mm_load_si128(reinterpret_cast<const __m128i*>(padded));

    __m128i cmp = _mm_cmpeq_epi8(slotData, needleData);
    // _mm_movemask_epi8(cmp) will be 0xFFFF if all 16 bytes matched.
    unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(cmp));
    return (mask == 0xFFFFU);
}

// Insert a token into the open-addressing table.
// If it's a duplicate, we do nothing. Otherwise we store it.
inline void insertToken(const char* token, int length)
{
    if (length <= 0) return;

    uint64_t hv = fnv1a_64(token, length);
    size_t idx = static_cast<size_t>(hv) & (TABLE_SIZE - 1);

    // Linear probing
    for (;;) {
        TokenEntry& entry = g_table[idx];
        if (entry.length == 0) {
            // Empty slot
            entry.hashVal = hv;
            memcpy(entry.token, token, length);
            entry.length = length;
            return;
        } 
        if (entry.hashVal == hv) {
            // Potential match, confirm by string compare
            if (tokensEqualSSE(entry, token, length)) {
                // It's already in the table, do nothing
                return;
            }
        }
        // Move to next slot
        idx = (idx + 1) & (TABLE_SIZE - 1);
    }
}

// ================== Parsing & MMap ==================

// Maps STDIN into memory using mmap.
// Returns a pointer to the mapped memory and the file size in 'sizeOut'.
static const char* mmapStdin(size_t& sizeOut) 
{
    // We read from file descriptor 0 (STDIN).
    struct stat sb;
    if (fstat(0, &sb) < 0) {
        perror("fstat on STDIN");
        return nullptr;
    }
    if (sb.st_size == 0) {
        // Nothing to read
        sizeOut = 0;
        return nullptr;
    }

    sizeOut = static_cast<size_t>(sb.st_size);
    void* mapped = mmap(nullptr, sizeOut, PROT_READ, MAP_PRIVATE, 0, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap on STDIN");
        return nullptr;
    }

    return static_cast<const char*>(mapped);
}

// ================== Main ==================
int main() 
{
    // Allocate our open-addressing table
    // We'll do a single 'new' to allocate the entire table. 
    // Each slot is 16 bytes + some overhead for hashVal and length ~ 32 bytes each.
    g_table = new TokenEntry[TABLE_SIZE];

    // Mmap STDIN
    size_t fileSize = 0;
    const char* data = mmapStdin(fileSize);
    if (!data || fileSize == 0) {
        // If there's no data, just output 0
        printf("0\n");
        return 0;
    }

    // Parse tokens from the buffer
    // We'll consider ' ' '\n' '\r' '\t' etc. as delimiters.
    // A straightforward approach is to manually scan for non-delimiter runs.

    const char* ptr = data;
    const char* end = data + fileSize;

    while (ptr < end) {
        // Skip delimiters
        while (ptr < end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\r' || *ptr == '\t')) {
            ++ptr;
        }
        // Now parse a token up to 16 chars or until next delimiter
        if (ptr >= end) break;

        char tokenBuf[16];
        int length = 0;
        while (ptr < end && !(*ptr == ' ' || *ptr == '\n' || *ptr == '\r' || *ptr == '\t')) {
            if (length < MAX_TOKEN_LEN) {
                tokenBuf[length++] = *ptr;
            }
            ++ptr;
        }
        // Insert token
        insertToken(tokenBuf, length);
    }

    // Count how many slots are used
    size_t uniqueCount = 0;
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        if (g_table[i].length > 0) {
            uniqueCount++;
        }
    }

    // Output result
    printf("%zu\n", uniqueCount);

    // Cleanup
    munmap((void*)data, fileSize);
    delete[] g_table;
    return 0;
}
