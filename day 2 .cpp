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



// 2) Pascal's Triangle
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


//3) Maximum Number of Groups Getting Fresh Donuts

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



//4)find minimum time to finish all the jobs 

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

//5) Container with most water

#include <iostream>
using namespace std;

int maxArea(int height[], int n) {
    int left = 0, right = n - 1;
    int maxArea = 0;

    while (left < right) {
        int width = right - left;
        int currentHeight = min(height[left], height[right]);
        maxArea = max(maxArea, width * currentHeight);

        // Move the pointer of the shorter line
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }

    return maxArea;
}

int main() {
    int height[] = {1, 8, 6, 2, 5, 4, 8, 3, 7};
    int n = sizeof(height) / sizeof(height[0]);

    cout << "Maximum area of water container: " << maxArea(height, n) << endl;

    return 0;
}



