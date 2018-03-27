#include <iostream>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <vector>
#include <cmath>
#include <limits.h>
#include "DataStructure.hpp"

using namespace std;

// Minimum Path Sum: given a m-by-n grid with nonnegative entries, find a path from top left to bottom right with minimal of the sum of all numbers along the path
class P64 {
public:
	int minPathSum(vector<vector<int>>& grid){
		if (grid.size() == 0)
			return 0;
		vector<vector<int>> dp(grid);
		for (int j = 1; j < dp[0].size(); j++)
			dp[0][j] += dp[0][j-1];
		for (int i = 1; i < dp.size(); i++)
			dp[i][0] += dp[i-1][0];
		for (int i = 1; i < dp.size(); i++){
			for (int j = 1; j < dp[0].size(); j++){
				dp[i][j] += min(dp[i-1][j], dp[i][j-1]);
			}
		}
		return dp[dp.size()-1][dp[0].size()-1];
	}
	void test(){
		vector<vector<int>> grid = {{1,3,1}, {1,5,1}, {4,2,1}};
		cout << minPathSum(grid) << endl;
		return;
	}
};


// Given n points on a 2D plane, find the maximum number of points that lie on the same straight line
class P149 {
public:
	int gcd(int a, int b){
		while(b != 0){
			int t = b;
			b = a % b;
			a = t;
		}
		return a;
	}
	int maxPoints(vector<Point>& points){
		if (points.size() < 3)
			return points.size();
		int res = 0;
		for(int i = 0; i < points.size(); i++){
			map<pair<int, int>, int> slope_map;
			int vertical = 0, same_point = 1;
			for(int j = i + 1; j < points.size(); j++){
				if (points[i].x == points[j].x){ // Check for same point
					if (points[i].y == points[j].y)
						same_point++;
					else
						vertical++;
				}
				else{
					int divisor = gcd(points[i].y - points[j].y, points[i].x - points[j].x);
					pair<int, int> slope = make_pair((points[i].y - points[j].y)/divisor, (points[i].x - points[j].x)/divisor);
					slope_map[slope]++;
				}
			}
			res = max(res, vertical + same_point);
			for(map<pair<int, int>, int>::iterator it = slope_map.begin(); it != slope_map.end(); it++)
				res = max(res, it->second + same_point);
		}
		return res;
	}
	void test(){
		vector<Point> points;
		points.push_back(Point(0, 0));
		points.push_back(Point(1, 1));
		points.push_back(Point(1, -1));
		cout << "Number of points on the same line is " << maxPoints(points) << endl;
		return;
	}
};

// Implement Min Stack
class MinStack{
public:
	MinStack(){
	}
	void push(int x){
		s1.push(x);
		if (s2.empty() || x <= s2.top())
			s2.push(x);
	}
	void pop(){
		if (s1.top() == s2.top())
			s2.pop();
		s1.pop();
	}
	int top(){
		return s1.top();
	}
	int getMin(){
		return s2.top();
	}
private:
	stack<int> s1, s2;
};
class P155 {
public:
	void test(){
		MinStack obj;
		obj.push(-2);
		obj.push(0);
		obj.push(-3);
		cout << obj.getMin() << endl;
		obj.pop();
		cout << obj.top() << endl;
		cout << obj.getMin() << endl;
		return;
	}
};

// Implement Queue using Stacks
class MyQueue {
public:
	MyQueue(){
	}
	void enqueue(int x){
		s1.push(x);
	}
	int dequeue(){
		int value = peek();
		s2.pop();
		return value;
	}
	int peek() {
		if (s2.empty()){
			while (!s1.empty()){
				s2.push(s1.top());
				s1.pop();
			}
		}
		return s2.top();
	}
	bool empty(){
		return s1.empty() && s2.empty();
	}
private:
	stack<int> s1, s2;
};

class P232 {
public:
	void test(){
		MyQueue obj;
		obj.enqueue(1);
		obj.enqueue(2);
		obj.enqueue(3);
		while (!obj.empty())
			cout << obj.dequeue() << ", ";
		cout << endl;
		return;
	}
};

// Reverse a string
class P344 {
public:
	string reverseString(string s){
		string ret;
		for (int i = s.size() - 1; i >= 0; i--)
			ret.push_back(s[i]);
		return ret;
	}
	void test() {
		string s = "hello";
		string rs = reverseString(s);
		cout << rs << endl;
	}
};

// A non-negative number is repesented as a singly linked list of digits, plus one to the number

class P369 {
public:
	ListNode* plusOne(ListNode* head){
		if (!head) 
			return head;
		if(DFS(head) == 0)
			return head;
		ListNode* p = new ListNode(1);
		p->next = head;
		return p;
	}
	int DFS(ListNode* head){
		int flag = 0;
		if (!head->next)
			flag = 1;
		else
			flag = DFS(head->next);
		int val = head->val + flag;
		head->val = val%10;
		flag = val/10;
		cout << flag << ";";
		return flag;
	}
	void test() {
		ListNode* head = new ListNode(1);
		ListNode* p = head;
		p->next = new ListNode(3);
		p->next->next = new ListNode(5);
		//ListNode* res = plusOne(head);
		printListNode(head);
		return;
	}
};

// Fizz BuzzWrite a program that outputs the string representation of numbers from 1 to n.
// But for multiples of three it should output “Fizz” instead of the number and for the multiples of five output “Buzz”. For numbers which are multiples of both three and five output “FizzBuzz”.

class P412 {
public:
	vector<string> fizzBuzz(int n) {
        vector<string> res;
        string record;
        if (n == 1){
            res.push_back("1");
            return res;
        }
        if(n % 3 == 0){
            if (n % 5 == 0)
                record = "FizzBuzz";
            else
                record = "Fizz";
        }
        else if (n % 5 == 0)
            record = "Buzz";
        else
            record = to_string(n);
        res = fizzBuzz(n-1);
        res.push_back(record);
        return res;
    }
    void test(){
    	vector<string> res = fizzBuzz(15);
    	for(vector<string>::iterator it = res.begin(); it != res.end(); it++)
    		cout << *it << ", " << endl;
    	return;
    }
};



// In LOL, Teemo attacks and make his enemy poisoned, with a poison duration. Given attacking ascending time and poison duration. Compute the total time of enemy in poison duration.
// For example, [1, 4] and 2 gives 4, with poison time being 1, 2, 4, 5; and [1, 2] and 2 gives 3, with poison time being 1, 2, 3.
class P495 {
public:
	int findPoisonedDurationRecursively(vector<int>& timeSeries, int duration) {
		if (timeSeries.size() == 0)
			return 0;
		if (timeSeries.size() == 1)
			return duration;
		else{
			vector<int> pre = timeSeries;
			int last = timeSeries.back();
			pre.pop_back();
			int adj = last - pre.back() >= duration ? duration : last - pre.back();
			return findPoisonedDurationRecursively(pre, duration) + adj;
		}
	}
	int findPoisonedDuration(vector<int>& timeSeries, int duration) {
		if (timeSeries.size() == 0)
			return 0;
		int sum = duration;
		for (int i = 1; i < timeSeries.size(); i++){
			sum += timeSeries[i] - timeSeries[i-1] >= duration ? duration : timeSeries[i] - timeSeries[i-1];
		}
		return sum;
	}
	void test() {
		int array[] = {1, 2, 6};
		vector<int> timeSeries(array, array + sizeof(array)/sizeof(int));
		int duration = 2;
		int poisoned = findPoisonedDuration(timeSeries, duration);
		cout << "Time duration in poison status is " << poisoned << endl;
		return;
	}
};

// Diagonal Traverse: given a matrix, teturn all elements in diagonal order
// Example, matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]] should return [1, 2, 4, 7, 5, 3, 6, 8, 9]
class P498 {
public:
	vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {
		vector<int> res;
		bool twist_from_same_row = true;
		int row, col, current_row = 0, current_col = 0, num_cols = matrix.size(), num_rows = matrix[0].size();
		if (num_cols == 0 || num_rows == 0)
			return res;
		res.push_back(matrix[current_row][current_col]);
		for(int sum = 1; sum < num_cols + num_rows; sum++){
			cout << "start for sum: " << sum << endl;
			if (twist_from_same_row){
				current_col++;
				for(row = current_row, col = current_col; row <= sum; row++, col--){
					res.push_back(matrix[row][col]);
					cout << "Vsist Point (" << row << ", " << col << "): " << matrix[row][col] << endl;
				}
				current_row = row--;
				current_col = col++;
			}
			else{
				current_row++;
				for(row = current_row, col = current_col; col <= sum; row--, col++){
					res.push_back(matrix[row][col]);
					cout << "Vsist Point (" << row << ", " << col << "): " << matrix[row][col] << endl;
				}
				current_row = row++;
				current_col = col--;
			}
			if (sum < (num_cols + num_rows)/2)
				twist_from_same_row = twist_from_same_row == false;
			else{
				if (col == num_cols - 1 || row == 0)
					twist_from_same_row = false;
				else if (col == 0 || row == num_rows - 1)
					twist_from_same_row = true;
			}
			if (twist_from_same_row)
				cout << "Pivoting at Point (" << current_row << ", " << current_col << "):  twist from same row" << endl;
			else
				cout << "Pivoting at Point (" << current_row << ", " << current_col << "):  twist from same col" << endl;
		}
		return res;
	}
	void test() {
		vector<vector<int>> mat {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		vector<int> res = findDiagonalOrder(mat);
		for(vector<int>::iterator it = res.begin(); it != res.end(); it++)
				cout << *it << ",";
		return;
	}
};


// Given a binary tree, compute the length of the diameter of the tree, defined as the length of the longest path between any two nodes in a tree.

class P543 {
public:
	int diameterOfBinaryTree(TreeNode* root) {
		int res;
		maxDepth(root, res);
		return res;
	}
	int maxDepth(TreeNode* node, int& res) {
		if (!node) return 0;
		int left = maxDepth(node->left, res);
		int right = maxDepth(node->right, res);
		res = max(res, left + right);
		return max(left, right) + 1;
	}
	void test(){
		TreeNode* tree = new TreeNode(1);
		tree->left = new TreeNode(2);
		tree->left->left = new TreeNode(4);
		tree->left->right = new TreeNode(5);
		tree->right = new TreeNode(3);
		int diam = diameterOfBinaryTree(tree);
		cout << "Diameter of the tree is " << diam << endl;
		return;
	}
};

// Given two binary trees and merge them with the rule that if tow nodes overlap, then sum node values up as the new value of the merged node. If null node, then use the value for Not null node.

class P617 {
public:
	TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
		if (!t1) return t2;
		if (!t2) return t1;
		t1->val += t2->val;
		t1->left = mergeTrees(t1->left, t2->left);
		t1->right = mergeTrees(t1->right, t2->right);
		return t1;
	}
	void test(){
		TreeNode* tree1 = new TreeNode(1);
		tree1->left = new TreeNode(3);
		tree1->left->left = new TreeNode(5);
		tree1->right = new TreeNode(2);
		TreeNode* tree2 = new TreeNode(2);
		tree2->left = new TreeNode(1);
		tree2->left->right = new TreeNode(4);
		tree2->right = new TreeNode(3);
		tree2->right->right = new TreeNode(7);
		cout << "Print Tree 1:" << endl;
		LevelOrder(tree1);
		cout << "Print Tree 2:" << endl;
		LevelOrder(tree2);
		TreeNode* tree = mergeTrees(tree1, tree2);
		cout << "Print Merged Tree" << endl;
		LevelOrder(tree);
		return;
	}
};

// Given a class of employee information, with id, importance value and id of direct subordinates, return the total importance of this branch
// For example, employee = [id, importance, [subordinates]], [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], id = 1 gives importance 5 + 3 + 3 = 11
class P690 {
public:
	int getImportance(vector<Employee*> employees, int id){
		if (employees.size() == 0)
			return 0;
		vector<Employee*> subordinates;
		int importance;
		for (int it = 0; it < employees.size(); it++){
			Employee* employee = employees[it];
			if (employee->id == id){
				importance = employee->importance;
				//cout << "marked " << id << ", importance " << importance <<endl;
				for (vector<int>::iterator jt = employee->subordinates.begin(); jt != employee->subordinates.end(); jt++)
					importance += getImportance(employees, *jt);
				return importance;
				break;
			}
		}
		return 0;
	}
	void test(){
		Employee* Martha = new Employee(1, 5, {2, 3});
		Employee* Sandeep = new Employee(2, 5, {4});
		Employee* Nagulan = new Employee(3, 4, {5});
		Employee* Ramesh = new Employee(4, 4, {});
		Employee* Ian = new Employee(5, 3, {6});
		Employee* Tianyi = new Employee(6, 2, {});
		vector<Employee*> FRC;
		FRC.push_back(Martha);
		FRC.push_back(Sandeep);
		FRC.push_back(Nagulan);
		FRC.push_back(Ramesh);
		FRC.push_back(Ian);
		FRC.push_back(Tianyi);
		cout << "Importance for Martha is " << getImportance(FRC, 1) << endl;
		return;
	}
};





// Given an array of integers, return the pivot index, which is defined as the index where the sum to the left is equal to the sum to the right.
// For example, nums = {1, 7, 3, 6, 5, 6} should return 3 as nums[3] = 6; nums = {1, 2, 3} should return -1
class P724 {
public:
	int pivotIndex(vector<int>& nums) {
		int res = -1;
        if (nums.size() == 0)
            return -1;
		vector<int> left = nums, right = nums;
		for(int i = 1; i != nums.size(); i++)
			left[i] += left[i-1];
		for(int i = nums.size() - 2; i != -1; i--)
			right[i] += right[i+1];
        for(int i = 0; i != nums.size(); i++){
            if (left[i] == right[i])
                return i;
        }
		return res;
	}
	void test() {
		int array[] = {1, 7, 3, 6, 5, 6};
		vector<int> nums(array, array + sizeof(array)/sizeof(int));
		int pivot = pivotIndex(nums);
		cout << "Pivot index is " << pivot << endl;
		return;
	}
};



// Given strings J representing the types of stones that are jewels(guaranteed distinct), and S representing the stones you have. Compute how many stones are also jewels.
// For exmaple, J = "aA" and S = "aAAbbbb" should return 3; J = "z" and S = "ZZ" should return 0.

class P771 {
public:
	int numJewelsInStones(string J, string S) {
		map<char, int> jewel_map;
		for (string::iterator i = J.begin(); i != J.end(); i++) {
			jewel_map[*i] = 0;
		}
		int res = 0;
		for (string::iterator i = S.begin(); i != S.end(); i++) {
			if (jewel_map.find(*i) != jewel_map.end())
				res++;
		}
		return res;
	}
	void test(){
		string J = "z";
		string S = "ZZ";
		int num = numJewelsInStones(J, S);
		cout << "Number of jewels in stones is " << num << endl;
		return;
	}
};

// Escape the Ghost: Pacman game, your starting point (0,0), ghosts position at ghots, your target at target, determine whether you can reach your target without being caught
class P789 {
public:
    bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
        int min_dist = INT_MAX;
		for (int i = 0; i < ghosts.size(); i++){
			int dist = 0;
			for (int j = 0; j < target.size(); j++)
				dist += abs(ghosts[i][j] - target[j]);
			min_dist = min(min_dist, dist);
		}
		int my_dist = 0;
		for (int j = 0; j < target.size(); j++)
			my_dist += abs(target[j]);
		return my_dist < min_dist;
    }
	void test(){
		vector<vector<int>> ghosts = {{1, 0}, {0, 3}};
		vector<int> target = {0, 1};
		cout << escapeGhosts(ghosts, target) << endl;
		return;
	}
};

class P804 {
public:
	int uniqueMorseRepresentations(vector<string>& words){
		vector<string> morse = {".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};
		map<char, string> morse_map;
		for (int i = 0; i < morse.size(); i++){
			morse_map[char('a' + i)] = morse[i];
		}
		set<string> res;
		for (vector<string>::iterator it = words.begin(); it != words.end(); it++){
			string code = "";
			string word = *it;
			for (int i = 0; i < word.size(); i++)
				code += morse_map[word[i]];
			res.insert(code);
		}
		return res.size();
	}
	void test() {
		vector<string> words = {"gin", "zen", "gig", "msg"};
		cout << uniqueMorseRepresentations(words) << endl;
		return;
	}
};