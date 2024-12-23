#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <algorithm>  // for std::search, std::memchr

// ----------------------------------------------------------------------------
// Minimal helpers for fast I/O
// ----------------------------------------------------------------------------

static inline void fastWrite(const char* s, size_t len) {
    ::write(STDOUT_FILENO, s, len);
}

static inline void fastWrite(const char* s) {
    fastWrite(s, std::strlen(s));
}

// ----------------------------------------------------------------------------
// SSE/AVX2-accelerated whitespace skip (if available). Otherwise fallback.
// ----------------------------------------------------------------------------

#ifdef __AVX2__
#include <immintrin.h>
static inline const char* skipSpaces(const char* ptr, const char* end) {
    // We'll do 32 bytes at a time, checking if all are whitespace.
    while (ptr + 32 <= end) {
        bool allWhitespace = true;
        for (int i = 0; i < 32; i++) {
            char c = ptr[i];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                allWhitespace = false;
                break;
            }
        }
        if (!allWhitespace) {
            break;
        }
        ptr += 32;
    }
    // scalar finishing
    while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')) {
        ptr++;
    }
    return ptr;
}
#else
static inline const char* skipSpaces(const char* ptr, const char* end) {
    while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')) {
        ptr++;
    }
    return ptr;
}
#endif

// ----------------------------------------------------------------------------
// Data structures
// ----------------------------------------------------------------------------

struct Phone {
    std::string code;
    uint64_t number;
};

struct Person {
    uint32_t id;
    bool hasAge;
    uint8_t age;
    bool hasHeight;
    double height;
    bool hasMarried;
    bool married;
    int phoneCount;
    Phone phones[3];
};

// ----------------------------------------------------------------------------
// Fast parse helpers
// ----------------------------------------------------------------------------

static inline uint32_t fastAtoi32(const char* start, const char* end) {
    uint32_t val = 0;
    for (const char* p = start; p < end; p++) {
        val = val * 10u + (uint32_t)(*p - '0');
    }
    return val;
}

static inline uint64_t fastAtoi64(const char* start, const char* end) {
    uint64_t val = 0;
    for (const char* p = start; p < end; p++) {
        val = val * 10ULL + (uint64_t)(*p - '0');
    }
    return val;
}

static inline double fastAtof(const char* start, const char* end) {
    // For speed, we do a small local copy:
    char buf[64];
    size_t len = (size_t)(end - start);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, start, len);
    buf[len] = '\0';
    return atof(buf);
}

// ----------------------------------------------------------------------------
// Floats: remove trailing zeros (and the decimal point if not needed)
// ----------------------------------------------------------------------------

static char* formatFloatNoTrailingZeros(double value, char* buf, size_t bufSize) {
    // 1) Print with a fixed decimal precision (6) to ensure enough digits
    snprintf(buf, bufSize, "%.6f", value);

    // 2) Strip trailing zeros
    char* end = buf + std::strlen(buf) - 1;
    while (end > buf && *end == '0') {
        --end;
    }
    // 3) If we end on a '.', remove it
    if (*end == '.') {
        --end;
    }
    *(end + 1) = '\0'; // re-terminate
    return end + 1;
}

// ----------------------------------------------------------------------------
// We define small constant arrays for searches
// ----------------------------------------------------------------------------
static const char CLOSE_AGE[]       = "</age>";
static const char CLOSE_HEIGHT[]    = "</height>";
static const char CLOSE_MARRIED[]   = "</married>";
static const char CLOSE_PHONE[]     = "</phone>";
static const char CLOSE_NUMBER[]    = "</number>";
static const char OPEN_NUMBER[]     = "<number>";
static const char OPEN_PHONE[]      = "<phone ";
static const char PERSON_TAG[]      = "<person";
static const char CLOSE_PERSON[]    = "</person>";
static const char ID_ATTR[]         = "id=\"";
static const char CODE_ATTR[]       = "code=\"";
static const char OPEN_AGE[]        = "<age>";
static const char OPEN_HEIGHT[]     = "<height>";
static const char OPEN_MARRIED[]    = "<married>";

// ----------------------------------------------------------------------------
// parsePhone: parse <phone code="..."><number>...</number></phone>
// ----------------------------------------------------------------------------

static const char* parsePhone(const char* p, const char* end, Phone& phone) {
    // find 'code="'
    {
        auto codePos = std::search(p, end, CODE_ATTR, CODE_ATTR + (sizeof(CODE_ATTR) - 1));
        if (codePos == end) return end;
        codePos += (sizeof(CODE_ATTR) - 1); // skip code="

        // read until the next '"'
        auto codeEnd = codePos;
        while (codeEnd < end && *codeEnd != '"') {
            codeEnd++;
        }
        if (codeEnd == end) return end;

        phone.code.assign(codePos, codeEnd);
        p = codeEnd + 1; // skip the ending quote
    }

    // parse <number>...</number>
    {
        auto numberPos = std::search(p, end, OPEN_NUMBER, OPEN_NUMBER + (sizeof(OPEN_NUMBER) - 1));
        if (numberPos == end) return end;
        numberPos += (sizeof(OPEN_NUMBER) - 1);

        auto numberEnd = std::search(numberPos, end, CLOSE_NUMBER, CLOSE_NUMBER + (sizeof(CLOSE_NUMBER) - 1));
        if (numberEnd == end) return end;

        phone.number = fastAtoi64(numberPos, numberEnd);
        p = numberEnd + (sizeof(CLOSE_NUMBER) - 1); // skip </number>
    }

    // skip past </phone>
    {
        auto phoneClose = std::search(p, end, CLOSE_PHONE, CLOSE_PHONE + (sizeof(CLOSE_PHONE) - 1));
        if (phoneClose < end) {
            p = phoneClose + (sizeof(CLOSE_PHONE) - 1);
        }
    }
    return p;
}

// ----------------------------------------------------------------------------
// parsePerson: parse <person id="...">...</person>
// ----------------------------------------------------------------------------

static const char* parsePerson(const char* p, const char* end, Person& person) {
    // zero out
    person.id = 0;
    person.hasAge = false;
    person.age = 0;
    person.hasHeight = false;
    person.height = 0.0;
    person.hasMarried = false;
    person.married = false;
    person.phoneCount = 0;

    // parse id="..."
    {
        auto idPos = std::search(p, end, ID_ATTR, ID_ATTR + (sizeof(ID_ATTR) - 1));
        if (idPos < end) {
            idPos += (sizeof(ID_ATTR) - 1); // skip id="
            const char* idEnd = idPos;
            while (idEnd < end && *idEnd != '"') {
                idEnd++;
            }
            if (idEnd < end) {
                person.id = fastAtoi32(idPos, idEnd);
                p = idEnd + 1; 
            }
        }
    }

    // parse sub-tags until we see </person>
    while (true) {
        // find next '<'
        const char* lt = (const char*)memchr(p, '<', end - p);
        if (!lt) {
            p = end;
            break;
        }
        p = lt;

        // check if it is </person>
        if ((size_t)(end - p) >= (sizeof(CLOSE_PERSON) - 1) &&
            memcmp(p, CLOSE_PERSON, (sizeof(CLOSE_PERSON) - 1)) == 0)
        {
            // done
            p += (sizeof(CLOSE_PERSON) - 1);
            break;
        }

        // otherwise check for known tags
        if ((size_t)(end - p) >= (sizeof(OPEN_AGE) - 1) &&
            memcmp(p, OPEN_AGE, (sizeof(OPEN_AGE) - 1)) == 0)
        {
            // parse <age>...</age>
            p += (sizeof(OPEN_AGE) - 1);
            auto close = std::search(p, end, CLOSE_AGE, CLOSE_AGE + (sizeof(CLOSE_AGE) - 1));
            if (close < end) {
                person.age = (uint8_t)fastAtoi32(p, close);
                person.hasAge = true;
                p = close + (sizeof(CLOSE_AGE) - 1);
            } else {
                p = end;
            }
            continue;
        }
        else if ((size_t)(end - p) >= (sizeof(OPEN_HEIGHT) - 1) &&
                 memcmp(p, OPEN_HEIGHT, (sizeof(OPEN_HEIGHT) - 1)) == 0)
        {
            p += (sizeof(OPEN_HEIGHT) - 1);
            auto close = std::search(p, end, CLOSE_HEIGHT, CLOSE_HEIGHT + (sizeof(CLOSE_HEIGHT) - 1));
            if (close < end) {
                person.height = fastAtof(p, close);
                person.hasHeight = true;
                p = close + (sizeof(CLOSE_HEIGHT) - 1);
            } else {
                p = end;
            }
            continue;
        }
        else if ((size_t)(end - p) >= (sizeof(OPEN_MARRIED) - 1) &&
                 memcmp(p, OPEN_MARRIED, (sizeof(OPEN_MARRIED) - 1)) == 0)
        {
            p += (sizeof(OPEN_MARRIED) - 1);
            auto close = std::search(p, end, CLOSE_MARRIED, CLOSE_MARRIED + (sizeof(CLOSE_MARRIED) - 1));
            if (close < end) {
                // naive trim
                const char* subBeg = p;
                const char* subEnd = close;
                while (subBeg < subEnd && (*subBeg == ' ' || *subBeg == '\t' ||
                                           *subBeg == '\r' || *subBeg == '\n')) 
                {
                    subBeg++;
                }
                while (subEnd > subBeg && (subEnd[-1] == ' ' || subEnd[-1] == '\t' ||
                                           subEnd[-1] == '\r' || subEnd[-1] == '\n')) 
                {
                    subEnd--;
                }
                std::string val(subBeg, subEnd);
                person.married = (val == "true");
                person.hasMarried = true;
                p = close + (sizeof(CLOSE_MARRIED) - 1);
            } else {
                p = end;
            }
            continue;
        }
        else if ((size_t)(end - p) >= (sizeof(OPEN_PHONE) - 1) &&
                 memcmp(p, OPEN_PHONE, (sizeof(OPEN_PHONE) - 1)) == 0)
        {
            // parse phone
            if (person.phoneCount < 3) {
                p = parsePhone(p + (sizeof(OPEN_PHONE) - 1), end, person.phones[person.phoneCount]);
                person.phoneCount++;
            } else {
                // skip if more than 3
                auto ph = std::search(p, end, CLOSE_PHONE, CLOSE_PHONE + (sizeof(CLOSE_PHONE) - 1));
                if (ph < end) {
                    p = ph + (sizeof(CLOSE_PHONE) - 1);
                } else {
                    p = end;
                }
            }
            continue;
        }
        else {
            // skip unknown tag: find '>'
            const char* gt = (const char*)memchr(p, '>', end - p);
            if (!gt) {
                p = end;
                break;
            }
            p = gt + 1;
        }
    }

    return p;
}

// ----------------------------------------------------------------------------
// printPersonJSON
// ----------------------------------------------------------------------------

static void printPersonJSON(const Person& person) {
    // We'll manually build JSON in a thread-local buffer to reduce overhead
    static thread_local std::string buf;
    buf.clear();
    buf.reserve(256);

    buf += "{";

    // "id":
    {
        buf += "\"id\": ";
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%u", person.id);
        buf += tmp;
    }

    if (person.hasAge) {
        buf += ", \"age\": ";
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%u", person.age);
        buf += tmp;
    }
    if (person.hasHeight) {
        buf += ", \"height\": ";
        char tmp[64];
        formatFloatNoTrailingZeros(person.height, tmp, sizeof(tmp));
        buf += tmp;
    }
    if (person.hasMarried) {
        buf += ", \"married\": ";
        buf += (person.married ? "true" : "false");
    }
    if (person.phoneCount > 0) {
        buf += ", \"phones\": [";
        for (int i = 0; i < person.phoneCount; i++) {
            const auto& ph = person.phones[i];
            if (i > 0) {
                buf += ", ";
            }
            buf += "{\"code\": \"";
            buf += ph.code;     // e.g. "+3"
            buf += "\", \"number\": ";
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long)ph.number);
            buf += tmp;         // e.g. 1322908759
            buf += "}";
        }
        buf += "]";
    }

    buf += "}\n";
    fastWrite(buf.data(), buf.size());
}

// ----------------------------------------------------------------------------
// main driver
// ----------------------------------------------------------------------------

int main(int /*argc*/, char** /*argv*/) {
    // 1) Memory map stdin if possible
    struct stat sb;
    if (fstat(STDIN_FILENO, &sb) < 0) {
        perror("fstat");
        return 1;
    }

    size_t fileSize = (size_t)sb.st_size;
    const char* data = nullptr;

    if (S_ISREG(sb.st_mode)) {
        // It's a regular file -> mmap
        void* p = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, STDIN_FILENO, 0);
        if (p == MAP_FAILED) {
            perror("mmap");
            return 1;
        }
        data = (const char*)p;
    } else {
        // fallback: read into a buffer
        size_t allocSize = fileSize ? fileSize : (1UL << 31); // 2GB fallback
        char* buf = (char*)malloc(allocSize);
        if (!buf) {
            fprintf(stderr, "Failed to allocate buffer.\n");
            return 1;
        }
        ssize_t offset = 0;
        while (true) {
            ssize_t rd = ::read(STDIN_FILENO, buf + offset, allocSize - offset);
            if (rd <= 0) break;
            offset += rd;
            if ((size_t)offset >= allocSize) break;
        }
        data = buf;
        fileSize = offset;
    }

    const char* ptr = data;
    const char* end = data + fileSize;

    // skip possible header, etc. parse <person> objects
    while (true) {
        // skip whitespace
        ptr = skipSpaces(ptr, end);
        if (ptr >= end) break;

        // find next <person
        auto personPos = std::search(ptr, end, PERSON_TAG, PERSON_TAG + (sizeof(PERSON_TAG) - 1));
        if (personPos == end) {
            break;
        }
        // parse it
        ptr = personPos + (sizeof(PERSON_TAG) - 1);
        Person person;
        ptr = parsePerson(ptr, end, person);
        printPersonJSON(person);
    }

    // cleanup
    if (S_ISREG(sb.st_mode)) {
        munmap((void*)data, fileSize);
    } else {
        free((void*)data);
    }

    return 0;
}
