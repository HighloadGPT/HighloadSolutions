#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>

#include <immintrin.h>  // For _mm_crc32_u64, etc.
#include <x86intrin.h>  // Some compilers put intrinsics here

// -----------------------------------------------------------------------------
// A token of up to 16 chars, stored in a 128-bit (__m128i) container, zero-padded.
// We'll store its length in a single byte. We rely on zero-padding to allow a
// full 16-byte comparison in constant time.
// -----------------------------------------------------------------------------
struct Token128 {
    __m128i data;   // 16 bytes
    uint8_t length; // Actual length (1..16). Zero if unused slot in the hash table.
};

// -----------------------------------------------------------------------------
// Convert a raw C-string [ptr, ptr+len) into a zero-padded __m128i
// -----------------------------------------------------------------------------
static inline __m128i make_128bit(const char* ptr, int len) {
    // We'll copy up to 16 bytes into a buffer, zero out the rest, and load it
    char temp[16] = {0};
    // len is guaranteed <= 16 as per problem statement
    memcpy(temp, ptr, len);
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(temp));
}

// -----------------------------------------------------------------------------
// Hardware-accelerated 64-bit hash.  We'll split the 128-bit token into
// two 64-bit chunks and feed them through CRC instructions.
// -----------------------------------------------------------------------------
static inline uint64_t hash_token(const __m128i &v) {
    union {
        __m128i v;
        uint64_t w[2];
    } x;
    x.v = v;
    // Start from a 0 seed. On Haswell+ we can use _mm_crc32_u64
    uint64_t h = 0;
    h = _mm_crc32_u64(h, x.w[0]);
    h = _mm_crc32_u64(h, x.w[1]);
    // You can optionally mix the result again, but for short strings CRC is fairly solid.
    return h;
}

// -----------------------------------------------------------------------------
// Compare two tokens for full equality (both length and 16-byte content).
// Since we zero-pad the remainder, we can just compare all 16 bytes if lengths match.
// -----------------------------------------------------------------------------
static inline bool equal_tokens(const Token128 &a, const Token128 &b) {
    if (a.length != b.length) return false;

    __m128i cmp = _mm_cmpeq_epi8(a.data, b.data);
    // movemask will be 0xFFFF if all bytes matched
    // However, we only care if all 16 bytes are identical. 
    // If they are, _mm_movemask_epi8(cmp) == 0xFFFF (i.e., 65535 or 0xFFFF).
    uint32_t mask = static_cast<uint32_t>(_mm_movemask_epi8(cmp));
    return (mask == 0xFFFF);
}

// -----------------------------------------------------------------------------
// A simple open-addressing hash set specialized to our Token128 type.
// We won't do dynamic resizing here to keep code simpler, but you could implement it.
// -----------------------------------------------------------------------------
class TokenHashSet {
public:
    // For 1,000,000 unique tokens, pick something comfortably larger (2^21 = 2,097,152).
    // You can tweak this based on memory constraints.
    static constexpr size_t DEFAULT_CAPACITY = 1ULL << 21; 

    TokenHashSet() 
        : capacity_(DEFAULT_CAPACITY),
          size_(0)
    {
        tokens_ = static_cast<Token128*>(std::aligned_alloc(64, capacity_ * sizeof(Token128)));
        hashes_ = static_cast<uint64_t*>(std::aligned_alloc(64, capacity_ * sizeof(uint64_t)));
        if (!tokens_ || !hashes_) {
            std::fprintf(stderr, "Allocation failed\n");
            std::exit(1);
        }
        // Mark empty slots
        for (size_t i = 0; i < capacity_; i++) {
            tokens_[i].length = 0; // indicates unused
            hashes_[i] = 0;
        }
    }

    ~TokenHashSet() {
        std::free(tokens_);
        std::free(hashes_);
    }

    // Insert a token if it doesn't already exist.
    // Returns true if inserted, false if already present.
    bool insert(const __m128i &data, uint8_t length) {
        uint64_t h = hash_token(data);
        size_t idx = static_cast<size_t>(h) & (capacity_ - 1);

        // linear probing
        while (true) {
            if (tokens_[idx].length == 0) {
                // unused slot, store new token
                tokens_[idx].data   = data;
                tokens_[idx].length = length;
                hashes_[idx]        = h;
                size_++;
                return true;
            } 
            else if (hashes_[idx] == h && equal_tokens(tokens_[idx], Token128{data, length})) {
                // same token already in the set
                return false;
            }
            idx = (idx + 1) & (capacity_ - 1);
        }
    }

    size_t size() const { return size_; }

private:
    Token128*  tokens_;
    uint64_t*  hashes_;
    size_t     capacity_;
    size_t     size_;
};

// -----------------------------------------------------------------------------
// Read entire STDIN via mmap. (On some systems, this can fail if stdin is not
// a file or if the size is unknown. For large data, using read() in big buffers
// is also a good approach.)
// -----------------------------------------------------------------------------
char* mmap_stdin(size_t &fileSize) {
    // We get the FD for stdin:
    int fd = ::fileno(stdin);

    // Stat the FD to get size:
    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::perror("fstat");
        std::exit(1);
    }
    if (st.st_size == 0) {
        // No data, just return nullptr.
        fileSize = 0;
        return nullptr;
    }
    fileSize = static_cast<size_t>(st.st_size);

    // Map it:
    void* ptr = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        std::perror("mmap");
        std::exit(1);
    }
    return static_cast<char*>(ptr);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main() {
    // Make stdin buffered if not already (optional)
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    size_t fileSize = 0;
    char* data = mmap_stdin(fileSize);

    // If mmap failed or the file is empty, fallback to a trivial approach:
    if (!data || fileSize == 0) {
        // Fallback: read line by line
        TokenHashSet set;
        std::string line;
        while (true) {
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            if (line.size() > 16) {
                // Problem statement implies max length 16, but just in case
                line.resize(16);
            }
            __m128i t = make_128bit(line.data(), (int)line.size());
            set.insert(t, (uint8_t)line.size());
        }
        std::cout << set.size() << "\n";
        return 0;
    }

    // We have the entire stdin file in [data, data+fileSize).
    // We'll parse lines manually, building tokens.
    TokenHashSet set;

    const char* ptr = data;
    const char* end = data + fileSize;

    while (ptr < end) {
        // Each token is one line => read until '\n' or EOF
        const char* lineStart = ptr;
        // Find newline:
        while (ptr < end && *ptr != '\n') {
            ptr++;
        }
        // Now [lineStart, ptr) is one line. If ptr == end, then we got to EOF
        size_t len = ptr - lineStart;
        if (len > 16) len = 16; // enforce max 16

        if (len > 0) {
            __m128i t = make_128bit(lineStart, (int)len);
            set.insert(t, (uint8_t)len);
        }

        // skip the newline char if we're not at the end
        if (ptr < end) {
            ptr++;
        }
    }

    // Unmap
    munmap(data, fileSize);

    // Finally, output the number of unique tokens
    std::cout << set.size() << "\n";

    return 0;
}
