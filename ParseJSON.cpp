#include <iostream>
#include <cstring>       // for std::memcpy
#include <cstdint>
#include <cstdio>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <immintrin.h>

//----------------------------------------------------------
// Minimal function to parse unsigned integers from a string
static inline uint32_t parse_uint(const char*& p) {
    uint32_t val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    return val;
}

//----------------------------------------------------------
// Minimal function to parse booleans (expects "true" or else false)
static inline bool parse_bool(const char*& p) {
    // Skip possible whitespace, quotes, commas, colons, etc.
    while (*p && (*p == ' ' || *p == ':' || *p == '\t' || *p == '"' || *p == ',')) {
        p++;
    }
    // If we see "true", consume and return true
    if (p[0] == 't' && p[1] == 'r' && p[2] == 'u' && p[3] == 'e') {
        p += 4;
        return true;
    }
    // Otherwise assume false
    return false;
}

//----------------------------------------------------------
// Skip a naive string: from leading double-quote to closing double-quote
static inline void skip_string(const char*& p) {
    if (*p != '"') return;
    p++; // skip opening quote
    while (*p && *p != '"') {
        p++;
    }
    if (*p == '"') p++; // skip closing quote
}

//----------------------------------------------------------
// A naive AVX2-based routine to skip non-structural characters quickly
// "Structural" = { } [ ] " : , or whitespace/newlines
static inline void skip_non_structural(const char*& p, const char* end) {
#if defined(__AVX2__)
    while (p + 32 <= end) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));

        // Compare chunk to each interesting character
        __m256i c0  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('{'));
        __m256i c1  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('}'));
        __m256i c2  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('['));
        __m256i c3  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(']'));
        __m256i c4  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('"'));
        __m256i c5  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(':'));
        __m256i c6  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(','));
        __m256i c7  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(' '));
        __m256i c8  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\t'));
        __m256i c9  = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\n'));
        __m256i c10 = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\r'));

        // Combine with OR
        __m256i or0  = _mm256_or_si256(c0, c1);
        __m256i or1  = _mm256_or_si256(c2, c3);
        __m256i or2  = _mm256_or_si256(c4, c5);
        __m256i or3  = _mm256_or_si256(c6, c7);
        __m256i or4  = _mm256_or_si256(c8, c9);
        __m256i or5  = _mm256_or_si256(c10, or0);

        __m256i t1   = _mm256_or_si256(or1, or2);
        __m256i t2   = _mm256_or_si256(or3, or4);
        __m256i t3   = _mm256_or_si256(t1, t2);
        __m256i mask = _mm256_or_si256(t3, or5);

        int bits = _mm256_movemask_epi8(mask);
        if (bits == 0) {
            // No structural chars found in these 32 bytes; skip them
            p += 32;
        } else {
            // We found at least one structural char
#if defined(__GNUC__) || defined(__clang__)
            // Use built-in to find first set bit
            int idx = __builtin_ctz(bits);
#else
            // Fallback for MSVC
            int idx = 0;
            while (((bits >> idx) & 1) == 0) {
                idx++;
            }
#endif
            p += idx;
            return;
        }
    }
#endif
    // If < 32 bytes left or no AVX2, fallback to char-by-char
    while (p < end) {
        char c = *p;
        if (c == '{' || c == '}' || c == '[' || c == ']' ||
            c == '"' || c == ':' || c == ',' ||
            c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            return;
        }
        p++;
    }
}

//----------------------------------------------------------
// A function to check if p points to a JSON key like `"user_id"`
static inline bool match_key(const char*& p, const char* key) {
    // Expect leading quote
    if (*p != '"') return false;
    p++;
    const char* k = key;
    while (*k && *k == *p) {
        p++;
        k++;
    }
    // If we matched the entire string and the next char is the closing quote
    if (*k == '\0' && *p == '"') {
        p++; // skip the closing quote
        return true;
    }
    return false;
}

//----------------------------------------------------------
// The main parsing function: parse the entire JSON input [p, end)
// to find sum of external USD transactions.
static uint64_t parse_records(const char* p, const char* end) {
    uint64_t sum_usd_external = 0;

    while (true) {
        // Skip until we see an object starting with '{'
        while (p < end && *p != '{') {
            p++;
        }
        if (p >= end) break;
        p++; // skip '{'

        bool isUSD       = false;
        uint32_t user_id = 0;
        bool have_user_id = false;
        int brace_depth   = 1;

        // Parse one JSON object
        while (p < end && brace_depth > 0) {
            skip_non_structural(p, end);
            if (p >= end) break;
            char c = *p;

            if (c == '"') {
                // Possibly "user_id", "currency", "transactions", ...
                if (match_key(p, "user_id")) {
                    // parse user_id
                    while (p < end && *p != ':') p++;
                    if (p < end) p++;
                    user_id = parse_uint(p);
                    have_user_id = true;
                }
                else if (match_key(p, "currency")) {
                    // parse currency string
                    while (p < end && *p != ':') p++;
                    if (p < end) p++;
                    // skip until opening quote
                    while (p < end && *p != '"') p++;
                    if (p < end) p++; // skip the '"'
                    // check if "USD"
                    static const char* USD = "USD";
                    isUSD = false;
                    int i = 0;
                    while (p < end && *p != '"' && i < 3) {
                        if (USD[i] != *p) break;
                        p++;
                        i++;
                    }
                    if (i == 3) {
                        // matched "USD"
                        isUSD = true;
                    }
                    // skip rest of string until closing quote
                    while (p < end && *p != '"') p++;
                    if (p < end) p++;
                }
                else if (match_key(p, "transactions")) {
                    // parse array of transactions
                    while (p < end && *p != '[') p++;
                    if (p >= end) break;
                    p++; // skip '['

                    // Loop over transaction objects until ']'
                    while (p < end && *p != ']') {
                        // skip until we see '{' (start of a transaction) or ']'
                        while (p < end && *p != '{' && *p != ']') {
                            p++;
                        }
                        if (p >= end) break;
                        if (*p == ']') {
                            break;
                        }
                        p++; // skip '{'

                        bool canceled = false;
                        uint32_t amount = 0;
                        uint32_t to_user_id = 0;
                        int tx_brace_depth = 1;

                        // Parse one transaction object
                        while (p < end && tx_brace_depth > 0) {
                            skip_non_structural(p, end);
                            if (p >= end) break;
                            char cc = *p;
                            if (cc == '"') {
                                if (match_key(p, "amount")) {
                                    while (p < end && *p != ':') p++;
                                    if (p < end) p++;
                                    amount = parse_uint(p);
                                }
                                else if (match_key(p, "to_user_id")) {
                                    while (p < end && *p != ':') p++;
                                    if (p < end) p++;
                                    to_user_id = parse_uint(p);
                                }
                                else if (match_key(p, "canceled")) {
                                    while (p < end && *p != ':') p++;
                                    if (p < end) p++;
                                    canceled = parse_bool(p);
                                } else {
                                    // unknown key
                                    skip_string(p);
                                }
                            }
                            else if (cc == '{') {
                                tx_brace_depth++;
                                p++;
                            }
                            else if (cc == '}') {
                                tx_brace_depth--;
                                p++;
                            }
                            else {
                                p++;
                            }
                        }

                        // Check if we add to sum
                        if (isUSD && have_user_id) {
                            // External (user_id != to_user_id) and not canceled
                            if (!canceled && user_id != to_user_id) {
                                sum_usd_external += amount;
                            }
                        }
                    }
                    // skip ']'
                    if (p < end && *p == ']') {
                        p++;
                    }
                }
                else {
                    // Some other key, skip the string
                    skip_string(p);
                }
            }
            else if (c == '{') {
                // Nested object
                brace_depth++;
                p++;
            }
            else if (c == '}') {
                // End of the current object
                brace_depth--;
                p++;
            }
            else if (c == '[' || c == ']' || c == ':' || c == ',') {
                p++;
            }
            else {
                // Possibly whitespace or other irrelevant chars
                p++;
            }
        } // end while brace_depth>0
    } // end while true

    return sum_usd_external;
}

//----------------------------------------------------------
// main: handle reading from stdin (mmap or fallback), parse, output result
int main() {
    int fd = fileno(stdin);
    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::perror("fstat");
        return 1;
    }

    bool is_regular = S_ISREG(st.st_mode); 
    size_t size = static_cast<size_t>(st.st_size);

    const char* data = nullptr;
    void* mapped = MAP_FAILED;
    std::vector<char> buffer;

    if (is_regular && size > 0) {
        // If it's a regular file with non-zero size, mmap directly
        mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            std::perror("mmap");
            return 1;
        }
        data = reinterpret_cast<const char*>(mapped);
    } else {
        // Fallback: read all data from stdin into buffer, then mmap that buffer
        constexpr size_t CHUNK = 65536;
        buffer.reserve(CHUNK);
        char tmp[CHUNK];
        ssize_t r;

        while ((r = ::read(fd, tmp, CHUNK)) > 0) {
            buffer.insert(buffer.end(), tmp, tmp + r);
        }
        // Now we have all data in buffer
        size = buffer.size();
        if (size == 0) {
            // No data => sum=0
            std::cout << 0 << std::endl;
            return 0;
        }
        // Create an anonymous mmap of that size
        mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped == MAP_FAILED) {
            std::perror("mmap");
            return 1;
        }
        std::memcpy(mapped, buffer.data(), size);
        data = reinterpret_cast<const char*>(mapped);
    }

    // Now parse the entire memory range
    const char* p   = data;
    const char* end = data + size;
    uint64_t sum_usd_external = parse_records(p, end);

    // Cleanup
    if (mapped != MAP_FAILED) {
        munmap(mapped, size);
    }

    // Print result
    std::cout << sum_usd_external << std::endl;
    return 0;
}
