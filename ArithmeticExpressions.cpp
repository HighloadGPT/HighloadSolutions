#include <bits/stdc++.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// This solution uses a shunting-yard parser to convert each line into RPN, then evaluates the RPN.
// We also memory-map stdin to avoid overhead of repeated I/O calls.

// A small SIMD-based function to skip whitespace. It uses AVX2 to check 32 bytes at a time.
// Whitespace characters: ' ', '\t', '\n', '\r'. We skip them in chunks if possible.
// This is a micro-optimization demonstration. For large expressions, it can help.
#ifdef __AVX2__
#include <immintrin.h>
#endif

static inline void skipWhitespaceSIMD(const char *&ptr, const char *end)
{
#ifdef __AVX2__
    static const __m256i spaces = _mm256_setr_epi8(
        ' ','\t','\n','\r',  // set of whitespace chars
        0,0,0,0, 0,0,0,0,    // we will fill up to 32
        0,0,0,0, 0,0,0,0,
        0,0,0,0, 0,0,0,0,
        0,0,0,0
    );
#endif

    // First skip any leftover scalar spaces to reach a 32-byte aligned boundary or until done.
    while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')) {
        ptr++;
    }

#ifdef __AVX2__
    // Then, while we have at least 32 bytes remaining, skip in AVX2 chunks if they are all whitespace.
    while (end - ptr >= 32) {
        // Load 32 bytes from ptr
        __m256i chunk = _mm256_loadu_si256((const __m256i*)ptr);

        // We'll check if chunk has any non-whitespace by checking if each
        // character is in the set { ' ', '\t', '\n', '\r' }. A quick approach
        // is to compare each byte with each candidate and OR them. If we find
        // a match, we set bits to 0xFF. Then we check if the result covers all
        // 32 bytes. But for a truly robust approach, you'd use a range check or
        // a specialized technique. For simplicity, let's do a naive approach
        // in scalar fallback. Doing a 'perfect' set membership test in AVX2
        // can be more elaborate than this code snippet. We'll do a partial
        // demonstration, then break to scalar.
        //
        // Realistically, because expressions contain parentheses, digits, etc.,
        // it's unlikely that big 32-byte blocks are *all* whitespace. This is
        // more demonstration than guaranteed speedup for every scenario.

        // We'll do a quick compare with ' '.
        __m256i eqSpace = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(' '));
        __m256i eqTab   = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\t'));
        __m256i eqNl    = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\n'));
        __m256i eqCr    = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\r'));

        // Combine them
        __m256i combined = _mm256_or_si256(_mm256_or_si256(eqSpace, eqTab),
                                           _mm256_or_si256(eqNl, eqCr));

        // Now check if combined is all 1 bits => 0xFF in every byte for the entire 256 bits
        int mask = _mm256_movemask_epi8(combined);
        if (mask == -1) {
            // All bytes matched one of the whitespace chars => skip the entire 32 bytes
            ptr += 32;
        } else {
            // Found a non-whitespace => break and handle the rest in scalar
            break;
        }
    }
#endif

    // Finally, skip in scalar mode whatever remains
    while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')) {
        ptr++;
    }
}

// Simple token types
enum class TokenType {
    Number,
    Plus,
    Minus,
    Mul,
    Div,
    LParen,
    RParen,
    End
};

// A token is either an operator or a 64-bit number
struct Token {
    TokenType type;
    int64_t value;
};

class ExpressionParser {
public:
    ExpressionParser(const char* start, const char* end)
        : ptr(start), endPtr(end)
    {
    }

    // Convert entire line to Reverse Polish Notation (RPN) using Shunting Yard
    // and store in rpnTokens
    void toRPN(std::vector<Token> &rpnTokens)
    {
        // Operator stack
        std::stack<Token> st;

        Token t;
        while (true) {
            t = nextToken();
            if (t.type == TokenType::End) {
                break;
            }
            switch (t.type) {
            case TokenType::Number:
                // goes directly to output
                rpnTokens.push_back(t);
                break;
            case TokenType::Plus:
            case TokenType::Minus:
                while (!st.empty() && st.top().type != TokenType::LParen) {
                    TokenType topT = st.top().type;
                    if (topT == TokenType::Mul || topT == TokenType::Div ||
                        topT == TokenType::Plus || topT == TokenType::Minus) {
                        rpnTokens.push_back(st.top());
                        st.pop();
                    } else {
                        break;
                    }
                }
                st.push(t);
                break;
            case TokenType::Mul:
            case TokenType::Div:
                // higher precedence than + and -
                while (!st.empty() && st.top().type != TokenType::LParen) {
                    TokenType topT = st.top().type;
                    if (topT == TokenType::Mul || topT == TokenType::Div) {
                        rpnTokens.push_back(st.top());
                        st.pop();
                    } else {
                        break;
                    }
                }
                st.push(t);
                break;
            case TokenType::LParen:
                st.push(t);
                break;
            case TokenType::RParen:
                // pop until matching (
                while (!st.empty() && st.top().type != TokenType::LParen) {
                    rpnTokens.push_back(st.top());
                    st.pop();
                }
                if (!st.empty() && st.top().type == TokenType::LParen) {
                    st.pop(); // remove '('
                }
                break;
            default:
                break;
            }
        }
        // pop any remaining operators
        while (!st.empty()) {
            rpnTokens.push_back(st.top());
            st.pop();
        }
    }

private:
    const char *ptr;
    const char *endPtr;

    // Return next token from the string
    Token nextToken()
    {
        skipWhitespaceSIMD(ptr, endPtr);
        if (ptr >= endPtr) {
            return Token{ TokenType::End, 0 };
        }

        char c = *ptr;
        // Check operator
        switch (c) {
        case '+': ++ptr; return Token{ TokenType::Plus, 0 };
        case '-': {
            // Could be a unary minus => we handle as part of number if next is digit or second minus ...
            // But simpler to always return 'Minus' token here, and rely on grammar to interpret.
            // We'll handle negative numbers by detecting them in getNumber() if it sees a sign.
            // But if we do that, we must look ahead. Another approach: if minus is followed by digit,
            // parse as negative number. Let's do that for performance (less parser overhead).
            if ((ptr+1) < endPtr && isdigit((unsigned char)*(ptr+1))) {
                // It's probably a negative number, parse that.
                return getNumber();
            } else {
                ++ptr;
                return Token{ TokenType::Minus, 0 };
            }
        }
        case '*': ++ptr; return Token{ TokenType::Mul, 0 };
        case '/': ++ptr; return Token{ TokenType::Div, 0 };
        case '(': ++ptr; return Token{ TokenType::LParen, 0 };
        case ')': ++ptr; return Token{ TokenType::RParen, 0 };
        default:
            // It's presumably a number
            if (isdigit((unsigned char)c)) {
                return getNumber();
            } else if (c == '\n' || c == '\r') {
                // end of line
                // treat as End
                return Token{ TokenType::End, 0 };
            } else {
                // Possibly a leading minus was already handled. If there's
                // e.g. a second minus or some other sign, parse as number or fallback.
                // We'll do one more check: if there's a sign, parse number. Otherwise, End.
                if ((c == '+' || c == '-') && (ptr+1) < endPtr && isdigit((unsigned char)*(ptr+1))) {
                    return getNumber();
                }
                // Unexpected char or whitespace
                ++ptr; 
                return Token{ TokenType::End, 0 };
            }
        }
    }

    // Parse an integer (possibly with leading sign) from the current pointer
    Token getNumber()
    {
        skipWhitespaceSIMD(ptr, endPtr);
        if (ptr >= endPtr) {
            return Token{TokenType::End, 0};
        }

        // We handle optional sign
        bool neg = false;
        if (*ptr == '-') {
            neg = true;
            ++ptr;
        } else if (*ptr == '+') {
            ++ptr;
        }

        // parse digits
        int64_t val = 0;
        while (ptr < endPtr && isdigit((unsigned char)*ptr)) {
            val = val * 10 + (*ptr - '0');
            ++ptr;
        }
        if (neg) val = -val;

        return Token{TokenType::Number, val};
    }
};

// Evaluate RPN tokens
static inline int64_t evalRPN(const std::vector<Token> &rpn)
{
    std::stack<int64_t> st;
    for (auto &tk : rpn) {
        switch (tk.type) {
        case TokenType::Number:
            st.push(tk.value);
            break;
        case TokenType::Plus: {
            int64_t b = st.top(); st.pop();
            int64_t a = st.top(); st.pop();
            st.push(a + b);
            break;
        }
        case TokenType::Minus: {
            int64_t b = st.top(); st.pop();
            int64_t a = st.top(); st.pop();
            st.push(a - b);
            break;
        }
        case TokenType::Mul: {
            int64_t b = st.top(); st.pop();
            int64_t a = st.top(); st.pop();
            st.push(a * b);
            break;
        }
        case TokenType::Div: {
            int64_t b = st.top(); st.pop();
            int64_t a = st.top(); st.pop();
            // integer division by zero is not handled explicitly here. We assume valid input.
            // C++ integer division truncates toward zero.
            st.push(a / b);
            break;
        }
        default:
            break;
        }
    }
    return st.top();
}

int main()
{
    // 1) Memory-map stdin
    //    On some systems "/dev/stdin" works. Alternatively, we could read from 0 (fileno(stdin)).
    struct stat sb;
    if (fstat(STDIN_FILENO, &sb) < 0) {
        perror("fstat");
        return 1;
    }
    size_t len = sb.st_size;
    if (len == 0) {
        // No input
        return 0;
    }

    void* data = mmap(nullptr, len, PROT_READ, MAP_PRIVATE | MAP_POPULATE, STDIN_FILENO, 0);
    if (data == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    const char* base = static_cast<const char*>(data);
    const char* end = base + len;

    // 2) We know we have 100 lines. We'll parse line by line. Each line can be huge.
    //    We'll find newline boundaries. For each line, parse it and evaluate.

    // A simple approach: we find each line boundary (or the end of file).
    // Because the problem statement says "The number of rows is 100".
    // We'll do exactly 100 expressions.

    for (int i = 0; i < 100; i++) {
        // Find line boundary
        const char* lineStart = base;
        if (lineStart >= end) {
            // no more data
            break;
        }

        // Move until newline or end
        while (base < end && *base != '\n') {
            base++;
        }
        // base points to newline or end
        const char* lineEnd = base;
        // skip the newline if present
        if (base < end && *base == '\n') {
            base++;
        }

        // Parse lineStart..lineEnd
        ExpressionParser parser(lineStart, lineEnd);
        std::vector<Token> rpnTokens;
        rpnTokens.reserve(65536); // avoid some re-allocs for large expressions
        parser.toRPN(rpnTokens);

        int64_t result = evalRPN(rpnTokens);
        std::cout << result << "\n";
    }

    munmap(data, len);
    return 0;
}
