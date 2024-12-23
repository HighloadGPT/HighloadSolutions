#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <immintrin.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

// We assume each line is exactly 36 chars of UUID + 1 char newline = 37 bytes
static constexpr size_t UUID_LEN = 36;
static constexpr size_t LINE_SIZE = 37;

//-----------------------------------------------------------------
// AVX2-based comparator for two 36-byte UUID strings.
//
// Returns:
//   negative if a < b (lexicographically)
//   zero     if a == b
//   positive if a > b
//-----------------------------------------------------------------
inline int avx2_cmp_36(const char* a, const char* b)
{
    // Compare first 32 bytes
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    __m256i x  = _mm256_xor_si256(va, vb);

    int mask = _mm256_movemask_epi8(x);
    if (mask != 0) {
        // The least significant set bit in 'mask' indicates the first differing byte
        int idx = __builtin_ctz(mask);
        return (unsigned char)a[idx] - (unsigned char)b[idx];
    }

    // If first 32 bytes are identical, compare the remaining 4 bytes
    return std::memcmp(a + 32, b + 32, UUID_LEN - 32);
}

// Comparator for sorting each chunk in ascending order
struct UuidComparator {
    bool operator()(const std::string& lhs, const std::string& rhs) const {
        return avx2_cmp_36(lhs.data(), rhs.data()) < 0;
    }
};

//-----------------------------------------------------------------
// Sorts a chunk of UUID strings in ascending order, writes them
// to a temporary file, and returns that filename.
//-----------------------------------------------------------------
std::string sortAndWriteChunk(std::vector<std::string>& chunk, int fileIndex)
{
    // Sort in ascending order
    std::sort(chunk.begin(), chunk.end(), UuidComparator());

    // Create temp filename
    std::string filename = "/tmp/uuid_sort_chunk_" + std::to_string(fileIndex);

    // Write sorted data (one UUID per line)
    {
        std::ofstream ofs(filename, std::ios::binary);
        for (auto& s : chunk) {
            ofs.write(s.data(), s.size());
            ofs.put('\n');
        }
    }
    return filename;
}

//-----------------------------------------------------------------
// Represents a line being merged plus its file of origin.
//-----------------------------------------------------------------
struct MergeItem {
    std::string currentLine;
    size_t fileIdx;
    MergeItem(const std::string& line, size_t idx)
        : currentLine(line), fileIdx(idx) {}
};

//-----------------------------------------------------------------
// Comparator for the multi-way merge priority queue.
//
// std::priority_queue is a max-heap by default. We want a min-heap
// (i.e. the smallest item is popped first). If `cmp(a, b)` > 0 => a > b,
// that means `a` is "lower priority" => `b` should rise to the top => return true.
//
// In simpler terms: "return (a > b)" => this flips it into a min-heap.
//
// So if `avx2_cmp_36(a, b) > 0`, a is lexicographically bigger => we want b on top => return true.
//-----------------------------------------------------------------
struct MergeComparator {
    bool operator()(const MergeItem& a, const MergeItem& b) const {
        int c = avx2_cmp_36(a.currentLine.data(), b.currentLine.data());
        // If `a > b` => c > 0 => return true => 'b' is on top => smaller is popped first
        return (c > 0);
    }
};

//-----------------------------------------------------------------
// Utility: read one UUID line from ifstream into outLine. 
// Returns false if EOF or error, true if success.
//-----------------------------------------------------------------
bool readUuidLine(std::ifstream& ifs, std::string& outLine)
{
    outLine.clear();
    if (std::getline(ifs, outLine)) {
        return true;
    }
    return false;
}

//-----------------------------------------------------------------
// Multi-way merge of sorted chunk files (ascending).
//  1) Open each file
//  2) Read first line from each => push into priority_queue (min-heap style)
//  3) Pop smallest => print => read next line from same file => push
//  4) Repeat until empty
//-----------------------------------------------------------------
void multiWayMerge(const std::vector<std::string>& chunkFilenames)
{
    // Open all sorted chunk files
    std::vector<std::ifstream> files;
    files.reserve(chunkFilenames.size());
    for (auto& fn : chunkFilenames) {
        files.emplace_back(fn, std::ios::binary);
    }

    // Our min-heap (via custom comparator)
    std::priority_queue<MergeItem, std::vector<MergeItem>, MergeComparator> pq;

    // Initialize heap with the first line from each file
    for (size_t i = 0; i < files.size(); ++i) {
        std::string line;
        if (readUuidLine(files[i], line)) {
            pq.push(MergeItem(line, i));
        }
    }

    // Pop smallest line, print, read next from same file
    while (!pq.empty()) {
        MergeItem topItem = pq.top();
        pq.pop();

        // Print to stdout (plus newline)
        std::fwrite(topItem.currentLine.data(),
                    1,
                    topItem.currentLine.size(),
                    stdout);
        std::fputc('\n', stdout);

        // Read next line from same file
        size_t idx = topItem.fileIdx;
        std::string nextLine;
        if (readUuidLine(files[idx], nextLine)) {
            pq.push(MergeItem(nextLine, idx));
        }
    }
    std::fflush(stdout);
}

//-----------------------------------------------------------------
// Main: single-threaded external mergesort with chunking.
//
//  1) Reads up to CHUNK_LINES from stdin
//  2) Sorts chunk in-memory (using AVX2 comparator), writes to temp file
//  3) Repeat until EOF
//  4) Multi-way merge => global ascending order => stdout
//-----------------------------------------------------------------
int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Tune this chunk size based on your memory/time constraints.
    static const size_t CHUNK_LINES = 2000000;

    std::vector<std::string> chunk;
    chunk.reserve(CHUNK_LINES);

    std::vector<std::string> chunkFiles;
    chunkFiles.reserve(32);

    size_t fileIndex = 0;
    while (true) {
        chunk.clear();
        chunk.reserve(CHUNK_LINES);

        // Read up to CHUNK_LINES from stdin
        for (size_t i = 0; i < CHUNK_LINES; ++i) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF
                break;
            }
            chunk.push_back(std::move(line));
        }
        if (chunk.empty()) {
            // No more data
            break;
        }

        // Sort + write chunk to temp file
        std::string filename = sortAndWriteChunk(chunk, fileIndex++);
        chunkFiles.push_back(filename);
    }

    // If no data read at all, done
    if (chunkFiles.empty()) {
        return 0;
    }

    // Merge all sorted chunks => ascending order => stdout
    multiWayMerge(chunkFiles);

    return 0;
}
