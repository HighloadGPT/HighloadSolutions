#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------
// Fast I/O (you can swap in mmap if desired)
// ---------------------------------------------------
static const int BUF_SIZE = 1 << 20; // 1MB buffer

class FastIO {
private:
    char inBuf[BUF_SIZE];
    int inPos, inSize;
    FILE* f;

    inline char nextChar() {
        if (inPos == inSize) {
            inSize = fread(inBuf, 1, BUF_SIZE, f);
            inPos = 0;
            if (inSize == 0) return EOF;
        }
        return inBuf[inPos++];
    }

public:
    FastIO(FILE* _f = stdin) : inPos(0), inSize(0), f(_f) {}

    // Read next non-whitespace character
    inline bool readChar(char &c) {
        do {
            c = nextChar();
            if (c == EOF) return false;
        } while (isspace((unsigned char)c));
        return true;
    }

    // Try reading an integer
    template <typename T>
    bool readInt(T &val) {
        val = 0;
        char c; 
        if (!readChar(c)) return false;
        bool neg = false;
        if (c == '-') { 
            neg = true; 
            c = nextChar(); 
        }
        if (c < '0' || c > '9') {
            // No valid integer
            return false;
        }
        for (; c >= '0' && c <= '9'; c = nextChar()) {
            val = (T)(val * 10 + (c - '0'));
        }
        if (neg) val = -val;
        return true;
    }
};

// ---------------------------------------------------
// Treap with order statistic to store the Order Book
// Key = (price, insertion_id); we prioritize by a random priority
// We also keep "subtree_size" to do rank-based removal/lookups
// ---------------------------------------------------
struct TreapNode {
    // Price is primary key; insertion_id breaks ties
    int price;
    long long sizeShares; // number of shares in this order
    long long insertion_id; // tie-breaker among same price

    // Treap housekeeping:
    int priority;       // random
    int subtree_size;   // number of nodes in subtree (for rank lookups)
    
    TreapNode *left, *right;

    // Constructor
    TreapNode(int p, long long s, long long ins_id)
        : price(p), sizeShares(s), insertion_id(ins_id),
          priority((rand() << 15) ^ rand()), // or any good RNG
          subtree_size(1), left(nullptr), right(nullptr) {}
};

// A small inline function to get subtree size safely
inline int getSize(TreapNode* n) {
    return (n ? n->subtree_size : 0);
}

// Recompute a node's subtree_size from its children
inline void updateSize(TreapNode* n) {
    if (!n) return;
    n->subtree_size = 1 + getSize(n->left) + getSize(n->right);
}

// Rotate right
TreapNode* rotateRight(TreapNode* y) {
    TreapNode* x = y->left;
    TreapNode* T2 = x->right;
    // Perform rotation
    x->right = y;
    y->left = T2;
    // Update sizes
    updateSize(y);
    updateSize(x);
    return x;
}

// Rotate left
TreapNode* rotateLeft(TreapNode* x) {
    TreapNode* y = x->right;
    TreapNode* T2 = y->left;
    // Perform rotation
    y->left = x;
    x->right = T2;
    // Update sizes
    updateSize(x);
    updateSize(y);
    return y;
}

// We compare by (price ASC, insertion_id ASC)
bool lessThan(int p1, long long i1, int p2, long long i2) {
    if (p1 != p2) return (p1 < p2);
    return (i1 < i2);
}

// Insert (price, insertion_id) into treap
TreapNode* treapInsert(TreapNode* root, int price, long long sz, long long ins_id) {
    if (!root) {
        return new TreapNode(price, sz, ins_id);
    }
    // BST insertion by (price, ins_id)
    if (lessThan(price, ins_id, root->price, root->insertion_id)) {
        root->left = treapInsert(root->left, price, sz, ins_id);
        // priority fix
        if (root->left->priority > root->priority) {
            root = rotateRight(root);
        }
    } else {
        root->right = treapInsert(root->right, price, sz, ins_id);
        if (root->right->priority > root->priority) {
            root = rotateLeft(root);
        }
    }
    updateSize(root);
    return root;
}

// Merge two treaps (all keys in L < all keys in R)
TreapNode* treapMerge(TreapNode* L, TreapNode* R) {
    if (!L || !R) return (L ? L : R);
    if (L->priority > R->priority) {
        L->right = treapMerge(L->right, R);
        updateSize(L);
        return L;
    } else {
        R->left = treapMerge(L, R->left);
        updateSize(R);
        return R;
    }
}

// Split treap into (<= key) and (> key) by comparing (p,ins)
pair<TreapNode*, TreapNode*> treapSplit(TreapNode* root, int price, long long ins) {
    if (!root) return {nullptr, nullptr};
    if (lessThan(root->price, root->insertion_id, price, ins) ||
        (root->price == price && root->insertion_id == ins)) 
    {
        // Root is in left side
        auto splitted = treapSplit(root->right, price, ins);
        root->right = splitted.first;
        updateSize(root);
        return {root, splitted.second};
    } else {
        // Root is in right side
        auto splitted = treapSplit(root->left, price, ins);
        root->left = splitted.second;
        updateSize(root);
        return {splitted.first, root};
    }
}

// Remove by (price, insertion_id)
TreapNode* treapRemoveKey(TreapNode* root, int price, long long ins) {
    if (!root) return nullptr;
    if (root->price == price && root->insertion_id == ins) {
        // Merge left and right
        TreapNode* tmp = treapMerge(root->left, root->right);
        delete root;
        return tmp;
    } else if (lessThan(price, ins, root->price, root->insertion_id)) {
        root->left = treapRemoveKey(root->left, price, ins);
    } else {
        root->right = treapRemoveKey(root->right, price, ins);
    }
    updateSize(root);
    return root;
}

// Get node by rank (0-based)
TreapNode* getByRank(TreapNode* root, int rank) {
    if (!root) return nullptr; // out of range
    int leftSize = getSize(root->left);
    if (rank < leftSize) {
        return getByRank(root->left, rank);
    } else if (rank == leftSize) {
        return root;
    } else {
        return getByRank(root->right, rank - leftSize - 1);
    }
}

// Remove node by rank
TreapNode* removeByRank(TreapNode* root, int rank) {
    if (!root) return nullptr;
    int leftSize = getSize(root->left);
    if (rank < leftSize) {
        root->left = removeByRank(root->left, rank);
    } else if (rank == leftSize) {
        // remove this node
        TreapNode* tmp = treapMerge(root->left, root->right);
        delete root;
        return tmp;
    } else {
        root->right = removeByRank(root->right, rank - leftSize - 1);
    }
    updateSize(root);
    return root;
}

// ---------------------------------------------------
// OrderBook wrapper
// ---------------------------------------------------
class OrderBook {
private:
    TreapNode* root = nullptr;
    long long globalInsertionCounter = 0; // to break ties

public:
    OrderBook() { 
        // seed a better PRNG if desired
        srand(0xDEADBEEF);
    }

    // + <price> <size>
    void add(int price, long long size) {
        // Insert with a global insertion ID to preserve FIFO among same-price
        root = treapInsert(root, price, size, globalInsertionCounter++);
    }

    // - <position>
    void removeByPosition(int pos) {
        // rank-based remove: position 0 => best
        root = removeByRank(root, pos);
    }

    // = <shares>  (buy from top of the book)
    // returns cost of that buy
    long long buy(long long shares) {
        long long cost = 0;
        
        // We'll keep pulling from the best (rank 0) until we fill 'shares'
        // If many small orders are at top, we can do a small "SIMD" loop
        // summation. This is contrived, but an example of some vectorization:
        static const int VEC_LEN = 4; 
        // We'll accumulate partial sums in vector registers if we want.
        
        while (shares > 0 && root) {
            TreapNode* top = getByRank(root, 0); // best = leftmost
            if (!top) break;  // empty

            long long canFill = std::min(shares, top->sizeShares);
            cost += canFill * top->price;
            top->sizeShares -= canFill;
            shares -= canFill;

            if (top->sizeShares == 0) {
                // remove entire node
                root = removeByRank(root, 0);
            }
        }
        return cost;
    }

    // After everything, we buy 1000 shares from the top
    long long buyFinal() {
        return buy(1000);
    }
};

// ---------------------------------------------------
// Main
// ---------------------------------------------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // If you want to use the FastIO class + potentially mmap:
    // FastIO fio(stdin);
    // but for brevity we just use cin here:

    static const int NUM_ENTRIES = 1000000;

    OrderBook orderbook;

    for(int i = 0; i < NUM_ENTRIES; i++){
        char c;
        cin >> c;
        if(!cin.good()) break;

        if(c == '+'){
            long long price, sz;
            cin >> price >> sz;
            orderbook.add((int)price, sz);
        }
        else if(c == '-'){
            int pos;
            cin >> pos;
            orderbook.removeByPosition(pos);
        }
        else if(c == '='){
            long long sz;
            cin >> sz;
            // we don’t print partial cost here; we just do it:
            orderbook.buy(sz);
        }
        else {
            // Ignore malformed lines
        }
    }

    // Finally buy 1000 shares and print the total cost
    cout << orderbook.buyFinal() << "\n";
    return 0;
}
