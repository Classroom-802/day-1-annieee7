//BAVNEET KAUR - 22BCS14121

// 1) Majority Elements
#include <iostream>
#include <vector>
using namespace std;
int majorityElement(vector<int>& nums) {
    int count = 0, candidate;
    for (int num : nums) {
        if (count == 0) candidate = num;
        count += (num == candidate) ? 1 : -1;
    }
    return candidate;
}

// 2) Single Number
#include <iostream>
#include <vector>
using namespace std;
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int num : nums) result ^= num;
    return result;
}

// 3) Convert Sorted Array to Binary Search Tree
#include <iostream>
#include <vector>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
TreeNode* helper(vector<int>& nums, int left, int right) {
    if (left > right) return nullptr;
    int mid = left + (right - left) / 2;
    TreeNode* node = new TreeNode(nums[mid]);
    node->left = helper(nums, left, mid - 1);
    node->right = helper(nums, mid + 1, right);
    return node;
}
TreeNode* sortedArrayToBST(vector<int>& nums) {
    return helper(nums, 0, nums.size() - 1);
}

// 4) Merge Two Sorted Lists
#include <iostream>
using namespace std;
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    if (!l1) return l2;
    if (!l2) return l1;
    if (l1->val < l2->val) {
        l1->next = mergeTwoLists(l1->next, l2);
        return l1;
    } else {
        l2->next = mergeTwoLists(l1, l2->next);
        return l2;
    }
}

// 5) Linked List Cycle
#include <iostream>
using namespace std;
bool hasCycle(ListNode *head) {
    if (!head || !head->next) return false;
    ListNode *slow = head, *fast = head->next;
    while (slow != fast) {
        if (!fast || !fast->next) return false;
        slow = slow->next;
        fast = fast->next->next;
    }
    return true;
}

// 6) Pascal's Triangle
#include <iostream>
#include <vector>
using namespace std;
vector<vector<int>> generate(int numRows) {
    vector<vector<int>> triangle;
    for (int i = 0; i < numRows; ++i) {
        vector<int> row(i + 1, 1);
        for (int j = 1; j < i; ++j) {
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
        }
        triangle.push_back(row);
    }
    return triangle;
}

// 7) Remove Linked List Elements
ListNode* removeElements(ListNode* head, int val) {
    if (!head) return nullptr;
    head->next = removeElements(head->next, val);
    return head->val == val ? head->next : head;
}

// 8) Reversed Linked List
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    while (head) {
        ListNode* nextTemp = head->next;
        head->next = prev;
        prev = head;
        head = nextTemp;
    }
    return prev;
}

// 9) Populating Next Right Pointers in Each Node
#include <queue>
using namespace std;
struct Node {
    int val;
    Node* left, *right, *next;
    Node(int x) : val(x), left(nullptr), right(nullptr), next(nullptr) {}
};
Node* connect(Node* root) {
    if (!root) return nullptr;
    queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            Node* node = q.front(); q.pop();
            if (i < size - 1) node->next = q.front();
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return root;
}

// 10) Design Circular Queue
class MyCircularQueue {
private:
    vector<int> data;
    int head, tail, size;
public:
    MyCircularQueue(int k) : data(k), head(-1), tail(-1), size(k) {}
    bool enQueue(int value) {
        if (isFull()) return false;
        if (isEmpty()) head = 0;
        tail = (tail + 1) % size;
        data[tail] = value;
        return true;
    }
    bool deQueue() {
        if (isEmpty()) return false;
        if (head == tail) {
            head = -1; tail = -1;
        } else {
            head = (head + 1) % size;
        }
        return true;
    }
    int Front() {
        return isEmpty() ? -1 : data[head];
    }
    int Rear() {
        return isEmpty() ? -1 : data[tail];
    }
    bool isEmpty() {
        return head == -1;
    }
    bool isFull() {
        return (tail + 1) % size == head;
    }
};

//11) Maximum Number of Groups Getting Fresh Donuts

#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int maxHappyGroups(int batchSize, vector<int>& groups) {
        vector<int> count(batchSize, 0);
        for (int group : groups) {
            count[group % batchSize]++;
        }
        
        // Handle groups with remainder 0 directly
        int happyGroups = count[0];
        
        // Pair remainders to maximize happy groups
        for (int i = 1; i <= batchSize / 2; ++i) {
            if (i == batchSize - i) { // Special case for exact middle
                happyGroups += count[i] / 2;
            } else {
                int pairs = min(count[i], count[batchSize - i]);
                happyGroups += pairs;
                count[i] -= pairs;
                count[batchSize - i] -= pairs;
            }
        }
        
        // Dynamic Programming to handle remaining groups
        unordered_map<string, int> memo;
        return happyGroups + dp(batchSize, count, memo);
    }

private:
    int dp(int batchSize, vector<int>& count, unordered_map<string, int>& memo) {
        string key = serialize(count);
        if (memo.count(key)) return memo[key];
        
        int remaining = 0;
        for (int i = 0; i < batchSize; ++i) {
            remaining += count[i];
        }
        if (remaining == 0) return 0;
        
        int maxHappy = 0;
        for (int i = 0; i < batchSize; ++i) {
            if (count[i] > 0) {
                count[i]--;
                int extraHappy = (i == 0) ? 1 : 0;
                maxHappy = max(maxHappy, extraHappy + dp(batchSize, count, memo));
                count[i]++;
            }
        }
        
        return memo[key] = maxHappy;
    }
    
    string serialize(const vector<int>& count) {
        string key = "";
        for (int x : count) {
            key += to_string(x) + ",";
        }
        return key;
    }
};

//12) Cherry Pickup II

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int cherryPickup(vector<vector<int>>& grid) {
        int rows = grid.size(), cols = grid[0].size();
        
        // Create a 3D DP array
        vector<vector<vector<int>>> dp(rows, vector<vector<int>>(cols, vector<int>(cols, 0)));
        
        // Base case: last row
        for (int j1 = 0; j1 < cols; ++j1) {
            for (int j2 = 0; j2 < cols; ++j2) {
                if (j1 == j2) {
                    dp[rows - 1][j1][j2] = grid[rows - 1][j1];
                } else {
                    dp[rows - 1][j1][j2] = grid[rows - 1][j1] + grid[rows - 1][j2];
                }
            }
        }
        
        // Fill the DP table bottom-up
        for (int i = rows - 2; i >= 0; --i) {
            vector<vector<int>> curr(cols, vector<int>(cols, 0));
            for (int j1 = 0; j1 < cols; ++j1) {
                for (int j2 = 0; j2 < cols; ++j2) {
                    int maxCherries = 0;
                    for (int dj1 = -1; dj1 <= 1; ++dj1) {
                        for (int dj2 = -1; dj2 <= 1; ++dj2) {
                            int nj1 = j1 + dj1, nj2 = j2 + dj2;
                            if (nj1 >= 0 && nj1 < cols && nj2 >= 0 && nj2 < cols) {
                                maxCherries = max(maxCherries, dp[i + 1][nj1][nj2]);
                            }
                        }
                    }
                    if (j1 == j2) {
                        curr[j1][j2] = grid[i][j1] + maxCherries;
                    } else {
                        curr[j1][j2] = grid[i][j1] + grid[i][j2] + maxCherries;
                    }
                }
            }
            dp[i] = curr;
        }
        
        // The result is the maximum cherries collected starting from the top
        return dp[0][0][cols - 1];
    }
};


//13 Maximum Number of Darts Inside of a Circular Dartboard

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

class Solution {
public:
    int numPoints(vector<vector<int>>& darts, int r) {
        int n = darts.size();
        int maxDarts = 1;
        double radius = static_cast<double>(r);
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double d = distance(darts[i], darts[j]);
                if (d > 2 * radius) continue;

                auto centers = getCircleCenters(darts[i], darts[j], radius);
                for (auto& center : centers) {
                    int count = 0;
                    for (int k = 0; k < n; ++k) {
                        if (distance(center, darts[k]) <= radius) {
                            count++;
                        }
                    }
                    maxDarts = max(maxDarts, count);
                }
            }
        }
        return maxDarts;
    }

private:
    double distance(vector<int>& p1, vector<int>& p2) {
        return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
    }

    vector<vector<double>> getCircleCenters(vector<int>& p1, vector<int>& p2, double r) {
        vector<vector<double>> centers;
        double d = distance(p1, p2);
        double midX = (p1[0] + p2[0]) / 2.0;
        double midY = (p1[1] + p2[1]) / 2.0;

        double h = sqrt(r * r - (d / 2.0) * (d / 2.0));
        double dx = (p2[1] - p1[1]) / d;
        double dy = (p2[0] - p1[0]) / d;

        centers.push_back({midX + h * dx, midY - h * dy});
        centers.push_back({midX - h * dx, midY + h * dy});
        return centers;
    }
};

//14)All O`one Data Structure

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <list>
using namespace std;

class AllOne {
private:
    // Node structure to store count and associated keys
    struct Node {
        int count;
        unordered_set<string> keys;
        Node(int c) : count(c) {}
    };

    unordered_map<string, list<Node>::iterator> keyNodeMap; // Maps keys to their Node
    list<Node> nodeList; // Doubly-linked list of Nodes

public:
    AllOne() {}

    void inc(string key) {
        if (keyNodeMap.find(key) == keyNodeMap.end()) {
            // New key
            if (nodeList.empty() || nodeList.front().count != 1) {
                nodeList.push_front(Node(1));
            }
            nodeList.front().keys.insert(key);
            keyNodeMap[key] = nodeList.begin();
        } else {
            // Existing key
            auto curNode = keyNodeMap[key];
            auto nextNode = next(curNode);
            curNode->keys.erase(key);
            if (curNode->keys.empty()) {
                nodeList.erase(curNode);
            }
            if (nextNode == nodeList.end() || nextNode->count != curNode->count + 1) {
                nextNode = nodeList.insert(nextNode, Node(curNode->count + 1));
            }
            nextNode->keys.insert(key);
            keyNodeMap[key] = nextNode;
        }
    }

    void dec(string key) {
        auto curNode = keyNodeMap[key];
        curNode->keys.erase(key);
        if (curNode->keys.empty()) {
            nodeList.erase(curNode);
        }
        if (curNode->count > 1) {
            auto prevNode = prev(curNode);
            if (curNode == nodeList.begin() || prevNode->count != curNode->count - 1) {
                prevNode = nodeList.insert(curNode, Node(curNode->count - 1));
            }
            prevNode->keys.insert(key);
            keyNodeMap[key] = prevNode;
        } else {
            keyNodeMap.erase(key);
        }
    }

    string getMaxKey() {
        return nodeList.empty() ? "" : *nodeList.back().keys.begin();
    }

    string getMinKey() {
        return nodeList.empty() ? "" : *nodeList.front().keys.begin();
    }
};


//15)find minimum time to finish all the jobs 

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minimumTimeRequired(vector<int>& jobs, int k) {
        int left = *max_element(jobs.begin(), jobs.end());
        int right = accumulate(jobs.begin(), jobs.end(), 0);
        int result = right;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (canDistribute(jobs, k, mid)) {
                result = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return result;
    }

private:
    bool canDistribute(vector<int>& jobs, int k, int maxWorkTime) {
        vector<int> workers(k, 0);
        return backtrack(jobs, workers, 0, maxWorkTime);
    }

    bool backtrack(vector<int>& jobs, vector<int>& workers, int jobIndex, int maxWorkTime) {
        if (jobIndex == jobs.size()) {
            return true;
        }

        for (int i = 0; i < workers.size(); ++i) {
            if (workers[i] + jobs[jobIndex] <= maxWorkTime) {
                workers[i] += jobs[jobIndex];
                if (backtrack(jobs, workers, jobIndex + 1, maxWorkTime)) {
                    return true;
                }
                workers[i] -= jobs[jobIndex];
            }

            // Avoid assigning the same job to multiple workers in equivalent states
            if (workers[i] == 0) break;
        }

        return false;
    }
};

//16)Minimum number of people to teach 

#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minimumTeachings(int n, vector<vector<int>>& languages, vector<vector<int>>& friendships) {
        unordered_map<int, unordered_set<int>> langMap;
        for (int i = 0; i < languages.size(); ++i) {
            langMap[i + 1] = unordered_set<int>(languages[i].begin(), languages[i].end());
        }

        unordered_set<int> needTeach;
        for (const auto& f : friendships) {
            int u = f[0], v = f[1];
            bool canCommunicate = false;
            for (int lang : langMap[u]) {
                if (langMap[v].count(lang)) {
                    canCommunicate = true;
                    break;
                }
            }
            if (!canCommunicate) {
                needTeach.insert(u);
                needTeach.insert(v);
            }
        }

        unordered_map<int, int> langFreq;
        for (int user : needTeach) {
            for (int lang : langMap[user]) {
                langFreq[lang]++;
            }
        }

        int maxCommon = 0;
        for (const auto& [lang, freq] : langFreq) {
            maxCommon = max(maxCommon, freq);
        }

        return needTeach.size() - maxCommon;
    }
};


//17)Count ways to make Array with Product 

#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

const int MOD = 1e9 + 7;

class Solution {
public:
    vector<int> waysToFillArray(vector<vector<int>>& queries) {
        // Precompute factorials and modular inverses
        const int MAX = 10015;
        vector<long long> fact(MAX), invFact(MAX);
        fact[0] = invFact[0] = 1;
        for (int i = 1; i < MAX; ++i) {
            fact[i] = fact[i - 1] * i % MOD;
            invFact[i] = modInverse(fact[i], MOD);
        }

        vector<int> result;
        for (const auto& q : queries) {
            int n = q[0], k = q[1];
            result.push_back(countWays(n, k, fact, invFact));
        }

        return result;
    }

private:
    long long modPow(long long base, long long exp, long long mod) {
        long long result = 1;
        while (exp > 0) {
            if (exp % 2 == 1) {
                result = result * base % mod;
            }
            base = base * base % mod;
            exp /= 2;
        }
        return result;
    }

    long long modInverse(long long x, long long mod) {
        return modPow(x, mod - 2, mod);
    }

    int countWays(int n, int k, const vector<long long>& fact, const vector<long long>& invFact) {
        unordered_map<int, int> primeFactors;
        int num = k;

        // Prime factorization of k
        for (int p = 2; p * p <= num; ++p) {
            while (num % p == 0) {
                primeFactors[p]++;
                num /= p;
            }
        }
        if (num > 1) {
            primeFactors[num]++;
        }

        long long ways = 1;

        // Calculate combinations for distributing prime factors
        for (const auto& [prime, freq] : primeFactors) {
            ways = ways * comb(freq + n - 1, n - 1, fact, invFact) % MOD;
        }

        return ways;
    }

    long long comb(int n, int r, const vector<long long>& fact, const vector<long long>& invFact) {
        if (n < r) return 0;
        return fact[n] * invFact[r] % MOD * invFact[n - r] % MOD;
    }
};

//18)Maximum  twin Sum of a linked list 

#include <iostream>
using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

class Solution {
public:
    int pairSum(ListNode* head) {
        // Step 1: Find the middle of the linked list using the slow and fast pointers
        ListNode* slow = head;
        ListNode* fast = head;

        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        // Step 2: Reverse the second half of the linked list
        ListNode* prev = nullptr;
        while (slow) {
            ListNode* nextNode = slow->next;
            slow->next = prev;
            prev = slow;
            slow = nextNode;
        }

        // Step 3: Compute the maximum twin sum
        int maxSum = 0;
        ListNode* firstHalf = head;
        ListNode* secondHalf = prev;

        while (secondHalf) {
            maxSum = max(maxSum, firstHalf->val + secondHalf->val);
            firstHalf = firstHalf->next;
            secondHalf = secondHalf->next;
        }

        return maxSum;
    }
};




