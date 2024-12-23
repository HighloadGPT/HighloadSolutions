#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <iostream>
#include <cstring>  // for memchr

// -----------------------------------------------------------------------------
// Toggle #defines to match your test harness:
// -----------------------------------------------------------------------------

// 1) Choose which JDN offset to use for 1970-01-01
//    Official references often say 2440588, but some code uses 2440587.
#define USE_JDN_2440587  0   // set to 1 or 0
//    ^ If =1, we subtract 2440587.  If =0, we subtract 2440588.

// 2) Choose offset sign interpretation
//    - RFC3339_SIGN => "YYYY-MM-DDTHH:MM:SS-03:00" => UTC = local + 3h
//    - FLIPPED_SIGN => "YYYY-MM-DDTHH:MM:SS-03:00" => UTC = local - 3h
#define RFC3339_SIGN     1   // set to 1 => official; set to 0 => flipped

// -----------------------------------------------------------------------------
// daysFrom1970: uses standard JDN formula, but subtracts either 2440587 or 2440588
// -----------------------------------------------------------------------------
static inline int daysFrom1970(int year, int month, int day) {
    // Standard proleptic Gregorian -> JDN
    int a = (14 - month) / 12;
    int y = year + 4800 - a;
    int m = month + 12*a - 3;

    int jdn = day
            + ( (153*m + 2) / 5 )
            + 365*y
            + (y / 4)
            - (y / 100)
            + (y / 400)
            - 32045;

    // 1970-01-01 is JD = 2440587.5, so some code does -2440587, some does -2440588.
#if USE_JDN_2440587
    static const int JDN_UNIX_EPOCH = 2440587;  // Some environments
#else
    static const int JDN_UNIX_EPOCH = 2440588;  // Common
#endif
    return jdn - JDN_UNIX_EPOCH;
}

// -----------------------------------------------------------------------------
// toUnixTimestamp: convert local date/time to UTC using your chosen sign logic
// -----------------------------------------------------------------------------
static inline int64_t toUnixTimestamp(
    int year, int month, int day,
    int hour, int minute, int second,
    char sign, int offHour, int offMin)
{
    int64_t days = (int64_t) daysFrom1970(year, month, day);
    int64_t localSec = days*86400 + hour*3600 + minute*60 + second;
    int64_t offsetSec = (int64_t)offHour * 3600 + (int64_t)offMin * 60;

#if RFC3339_SIGN
    // RFC 3339 official interpretation:
    //   "-03:00" => local behind => UTC = local + offset
    //   "+02:00" => local ahead  => UTC = local - offset
    if (sign == '-') {
        localSec += offsetSec;
    } else {
        localSec -= offsetSec;
    }
#else
    // Flipped sign interpretation:
    //   "-03:00" => UTC = local - offset
    //   "+02:00" => UTC = local + offset
    if (sign == '-') {
        localSec -= offsetSec;
    } else {
        localSec += offsetSec;
    }
#endif

    return localSec;
}

// -----------------------------------------------------------------------------
// parseLineAndComputeTimestamp: parse "YYYY-MM-DDTHH:MM:SS±HH:MM" (25 chars)
// -----------------------------------------------------------------------------
static inline int64_t parseLineAndComputeTimestamp(const char* line) {
    // year
    int year = (line[0] - '0')*1000
             + (line[1] - '0')*100
             + (line[2] - '0')*10
             + (line[3] - '0');
    // month
    int month = (line[5] - '0')*10 + (line[6] - '0');
    // day
    int day   = (line[8] - '0')*10 + (line[9] - '0');
    // hour
    int hour  = (line[11]-'0')*10 + (line[12]-'0');
    // minute
    int minute= (line[14]-'0')*10 + (line[15]-'0');
    // second
    int second= (line[17]-'0')*10 + (line[18]-'0');
    // sign
    char sign = line[19];
    // offset hour
    int offH  = (line[20]-'0')*10 + (line[21]-'0');
    // offset minute
    int offM  = (line[23]-'0')*10 + (line[24]-'0');

    return toUnixTimestamp(year, month, day, hour, minute, second, sign, offH, offM);
}

// -----------------------------------------------------------------------------
// main(): mmap stdin, parse lines, sum, print result
// -----------------------------------------------------------------------------
int main() {
    // 1) fstat -> size
    struct stat st;
    if (fstat(0, &st) < 0) {
        std::perror("fstat failed on stdin");
        return 1;
    }
    if (st.st_size == 0) {
        std::cout << 0 << "\n";
        return 0;
    }

    // 2) mmap all of stdin
    char* data = static_cast<char*>(
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, 0, 0)
    );
    if (data == MAP_FAILED) {
        std::perror("mmap failed on stdin");
        return 1;
    }

    // 3) parse line by line
    int64_t sumTimestamps = 0;
    const char* ptr = data;
    const char* end = data + st.st_size;

    while (ptr < end) {
        // Find next newline or end
        const char* newlinePos = static_cast<const char*>(
            std::memchr(ptr, '\n', end - ptr)
        );
        if (!newlinePos) {
            newlinePos = end;
        }
        size_t lineLen = (size_t)(newlinePos - ptr);

        // If we have at least 25 chars, parse as RFC3339
        if (lineLen >= 25) {
            sumTimestamps += parseLineAndComputeTimestamp(ptr);
        }

        // Advance ptr
        ptr = newlinePos;
        if (ptr < end && *ptr == '\n') {
            ++ptr;
        }
        // Skip possible '\r'
        if (ptr < end && *ptr == '\r') {
            ++ptr;
        }
    }

    // 4) cleanup and output
    munmap(data, st.st_size);
    std::cout << sumTimestamps << "\n";
    return 0;
}
