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

// 最小路径和
/*给定一个只含非负整数的 m x n 网格，找到一条从左上角到右下角的可以使数字之和最小的路径。

注意: 每次只能向下或者向右移动一步。

示例 1:

[[1,3,1],
 [1,5,1],
 [4,2,1]]

根据上面的数组，返回 7. 因为路径 1→3→1→1→1 总和最小。*/
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


// 直线上最多的点数
/*给定二维平面上有 n 个点，求最多有多少点在同一条直线上。*/

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

// 最小栈
/*设计一个支持 push，pop，top 操作，并能在常量时间内检索最小元素的栈。

    push(x) -- 将元素x推入栈中。
    pop() -- 删除栈顶的元素。
    top() -- 获取栈顶元素。
    getMin() -- 检索栈中的最小元素。

示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
*/

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

// 用栈来实现队列
/*使用栈来实现队列的如下操作：

    push(x) -- 将一个元素放入队列的尾部。
    pop() -- 从队列首部移除元素。
    peek() -- 返回队列首部的元素。
    empty() -- 返回队列是否为空。

注意:

    你只能使用标准的栈操作-- 也就是只有push to top, peek/pop from top, size, 和 is empty 操作是可使用的。
    你所使用的语言也许不支持栈。你可以使用 list 或者 deque (双端队列)来模拟一个栈，只要你仅使用栈的标准操作就可以。
    假设所有操作都是有效的，比如 pop 或者 peek 操作不会作用于一个空队列上。
*/
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

// Find the lowest common ancester of a tree
/*class P236{
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        
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
}*/


// 反转字符串
/*请编写一个函数，其功能是将输入的字符串反转过来。

示例：

输入：s = "hello"
返回："olleh"
*/

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

// 字符串解码
/*给定一个经过编码的字符串，返回它解码后的字符串。
编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
示例:
s = "3[a]2[bc]", 返回 "aaabcbc".
s = "3[a2[c]]", 返回 "accaccacc".
s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".
*/

class P394 {
public:
    string decodeString(string s) {
        stack<int> nums;
		stack<string> strs;
		string str = "";
		int cnt = 0;
		for(string::iterator it = s.begin(); it != s.end(); it++){
			if (*it >= '0' && *it <= '9')
				cnt = 10*cnt + *it - '0';
			else if (*it == '['){
				strs.push(str);
				nums.push(cnt);
				cnt = 0;
				str = "";
			}
			else if (*it == ']'){
				for(int i = 0; i < nums.top(); i++)
					strs.top() += str;
				str = strs.top();
				nums.pop();
				strs.pop();
			}
			else
				str += *it;
		}
		return strs.empty() ? str : strs.top();
    }
	void test() {
		string s = "3[a2[c]]";
		cout << "encoded string is :" << s << endl;
		cout << "decoded string is :" << decodeString(s) << endl;
		return;
	}
};


// Fizz Buzz
/*写一个程序，输出从 1 到 n 数字的字符串表示。
1. 如果 n 是3的倍数，输出“Fizz”；
2. 如果 n 是5的倍数，输出“Buzz”；
3.如果 n 同时是3和5的倍数，输出 “FizzBuzz”。*/
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

//汉明距离总和
/*两个整数的 汉明距离 指的是这两个数字的二进制数对应位不同的数量。
计算一个数组中，任意两个数之间汉明距离的总和。
示例:
输入: 4, 14, 2
输出: 6
解释: 在二进制表示中，4表示为0100，14表示为1110，2表示为0010。（这样表示是为了体现后四位之间关系）
所以答案为：
HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.*/
class P477 {
public:
    int totalHammingDistance(vector<int>& nums) {
        int res = 0, digits = 32;
		for (int i = 0; i < digits; i++){
			int cnt = 0;
			for(int num : nums) {
				if (num & (1 << i))
					++cnt;
			}
			res += cnt * (nums.size() - cnt);
		}
		return res;
    }
	void test() {
		vector<int> nums = {4, 14, 2};
		cout << "Sum of Hamming distance is " << totalHammingDistance(nums) << endl;
		return;
	}
};


// 提莫攻击
/*在《英雄联盟》的世界中，有一个叫“提莫”的英雄，他的攻击可以让敌方英雄艾希（编者注：寒冰射手）进入中毒状态。现在，给出提莫对艾希的攻击时间序列和提莫攻击的中毒持续时间，你需要输出艾希的中毒状态总时长。
你可以认为提莫在给定的时间点进行攻击，并立即使艾希处于中毒状态。
示例1:
输入: [1,4], 2
输出: 4
原因: 在第1秒开始时，提莫开始对艾希进行攻击并使其立即中毒。中毒状态会维持2秒钟，直到第2秒钟结束。
在第4秒开始时，提莫再次攻击艾希，使得艾希获得另外2秒的中毒时间。
所以最终输出4秒。*/
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

// Diagonal Traverse
/*Given a matrix of M x N elements (M rows, N columns), return all elements of the matrix in diagonal order as shown in the below image.
Example:
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output:  [1,2,4,7,5,3,6,8,9]*/
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


// 二叉树的直径
/*给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。
示例 :
给定二叉树

          1
         / \
        2   3
       / \     
      4   5    

返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。*/
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

//朋友圈
/*班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，那么我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。
给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果M[i][j] = 1，表示已知第 i 个和 j 个学生互为朋友关系，否则为不知道。你必须输出所有学生中的已知的朋友圈总数。
示例 1:
输入: 
[[1,1,0],
 [1,1,0],
 [0,0,1]]
输出: 2 
说明：已知学生0和学生1互为朋友，他们在一个朋友圈。
第2个学生自己在一个朋友圈。所以返回2。

示例 2:
输入: 
[[1,1,0],
 [1,1,1],
 [0,1,1]]
输出: 1
说明：已知学生0和学生1互为朋友，学生1和学生2互为朋友，所以学生0和学生2也是朋友，所以他们三个在一个朋友圈，返回1。*/

class P547 {
public:
    int findCircleNum(vector<vector<int>>& M) {
        vector<int> visited(M.size(), 0);
		int cnt = 0;
		for(int i = 0; i < M.size(); i++){
			for (int num : visited)
				cout << num << ",";
			cout << endl;
			if (visited[i] == 0){
				dfs(M, visited, i);
				cnt++;
			}
		}
		return cnt;
    }
	void dfs(vector<vector<int>>&M, vector<int>& visited, int target){
		for (int i = target + 1; i < M.size(); i++){
			if (M[target][i] == 1 && visited[i] == 0){
				cout << "visiting node " << i << endl;
				visited[i] = 1;
				dfs(M, visited, i);
			}
		}
	}
	void test() {
		vector<vector<int>> M = {{1, 1, 0}, {1, 1, 0}, {0, 0, 1}};
		cout << "number of circles is " << findCircleNum(M) << endl;
	}
};

// 合并二叉树
/*给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。
示例 1:
输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7

注意: 合并必须从两个树的根节点开始。*/
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

// 单词替换
/*在英语中，我们有一个叫做 词根(root)的概念，它可以跟着其他一些词组成另一个较长的单词——我们称这个词为 继承词(successor)。例如，词根an，跟随着单词 other(其他)，可以形成新的单词 another(另一个)。
现在，给定一个由许多词根组成的词典和一个句子。你需要将句子中的所有继承词用词根替换掉。如果继承词有许多可以形成它的词根，则用最短的词根替换它。
你需要输出替换之后的句子。

示例 1:
输入: dict(词典) = ["cat", "bat", "rat"]
sentence(句子) = "the cattle was rattled by the battery"
输出: "the cat was rat by the bat"*/

class P648 {
public:
    string replaceWords(vector<string>& dict, string sentence) {
        return sentence;
    }
};

// 员工的重要性
/*定一个保存员工信息的数据结构，它包含了员工唯一的id，重要度 和 直系下属的id。
比如，员工1是员工2的领导，员工2是员工3的领导。他们相应的重要度为15, 10, 5。那么员工1的数据结构是[1, 15, [2]]，员工2的数据结构是[2, 10, [3]]，员工3的数据结构是[3, 5, []]。注意虽然员工3也是员工1的一个下属，但是由于并不是直系下属，因此没有体现在员工1的数据结构中。
现在输入一个公司的所有员工信息，以及单个员工id，返回这个员工和他所有下属的重要度之和。

示例 1:
输入: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
输出: 11
解释:
员工1自身的重要度是5，他有两个直系下属2和3，而且2和3的重要度均为3。因此员工1的总重要度是 5 + 3 + 3 = 11。*/

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





// 寻找数组的中心索引
/*给定一个整数类型的数组 nums，请编写一个能够返回数组“中心索引”的方法。
我们是这样定义数组中心索引的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。

示例 1:
输入: 
nums = [1, 7, 3, 6, 5, 6]
输出: 3
解释: 
索引3 (nums[3] = 6) 的左侧数之和(1 + 7 + 3 = 11)，与右侧数之和(5 + 6 = 11)相等。
同时, 3 也是第一个符合要求的中心索引。

示例 2:
输入: 
nums = [1, 2, 3]
输出: -1
解释: 
数组中不存在满足此条件的中心索引。*/
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

// 托普利茨矩阵
/*如果一个矩阵的每一方向由左上到右下的对角线上具有相同元素，那么这个矩阵是托普利茨矩阵。
给定一个 M x N 的矩阵，当且仅当它是托普利茨矩阵时返回 True。

示例 1:
输入: matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
输出: True
解释:
1234
5123
9512
在上面这个矩阵中, 对角线分别是 "[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]", 各条对角线上的所有元素都相同, 因此答案是True。

示例 2:
输入: matrix = [[1,2],[2,2]]
输出: False
解释: 
对角线, 比如: "[1, 2]" 上有不同的元素。*/

class P766 {
public:
    bool isToeplitzMatrix(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        for (int j = 0, i = 0; j < cols; j++) {
            int x = i, y = j;
            while (x < rows - 1 && y < cols - 1) {
                if (matrix[x][y] != matrix[x + 1][y + 1])
                    return false;
                x++;
                y++;
            }
        }
        for (int i = 1, j = 0; i < rows; i++) {
            int x = i, y = j;
            while (x < rows - 1 && y < cols - 1) {
                if (matrix[x][y] != matrix[x + 1][y + 1])
                    return false;
                x++;
                y++;
            }
        }
        return true;
    }
	void test() {
		vector<vector<int>>matrix = {{1,2,3,4},{5,1,2,3},{9,5,1,2}};
		if (isToeplitzMatrix(matrix))
			cout << "The matrix is Toeplitz" << endl;
		else
			cout << "The matrix is not Toeplitz" << endl;
		return;
	}
};

// 重构字符串
/*给定一个字符串S，检查是否能重新排布其中的字母，使得两相邻的字符不同。
若可行，输出任意可行的结果。若不可行，返回空字符串。

示例 1:
输入: S = "aab"
输出: "aba"

示例 2:
输入: S = "aaab"
输出: ""*/

class P767 {
public:
    string reorganizeString(string S) {
        return S;
    }
	void test() {
		string s = "aab";
		cout << "original: " << s << ", converted: " << reorganizeString(s) << endl;
		return;
	}
};

// 宝石与石头
/* 给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。
J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。

示例 1:
输入: J = "aA", S = "aAAbbbb"
输出: 3

示例 2:
输入: J = "z", S = "ZZ"
输出: 0*/

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

// Escape The Ghosts
/*You are playing a simplified Pacman game. You start at the point (0, 0), and your destination is (target[0], target[1]). There are several ghosts on the map, the i-th ghost starts at (ghosts[i][0], ghosts[i][1]).
Each turn, you and all ghosts simultaneously *may* move in one of 4 cardinal directions: north, east, west, or south, going from the previous point to a new point 1 unit of distance away.
You escape if and only if you can reach the target before any ghost reaches you (for any given moves the ghosts may take.)  If you reach any square (including the target) at the same time as a ghost, it doesn't count as an escape.
Return True if and only if it is possible to escape.

Example 1:
Input: 
ghosts = [[1, 0], [0, 3]]
target = [0, 1]
Output: true
Explanation: 
You can directly reach the destination (0, 1) at time 1, while the ghosts located at (1, 0) or (0, 3) have no way to catch up with you.

Example 2:
Input: 
ghosts = [[1, 0]]
target = [2, 0]
Output: false
Explanation: 
You need to reach the destination (2, 0), but the ghost at (1, 0) lies between you and the destination.

Example 3:
Input: 
ghosts = [[2, 0]]
target = [1, 0]
Output: false
Explanation: 
The ghost can reach the target at the same time as you.*/

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


// 阶乘函数后K个零
/* f(x) 是 x! 末尾是0的数量。（回想一下 x! = 1 * 2 * 3 * ... * x，且0! = 1）
例如， f(3) = 0 ，因为3! = 6的末尾没有0；而 f(11) = 2 ，因为11!= 39916800末端有2个0。给定 K，找出多少个非负整数x ，有 f(x) = K 的性质。
示例 1:
输入:K = 0
输出:5
解释: 0!, 1!, 2!, 3!, and 4! 均符合 K = 0 的条件。

示例 2:
输入:K = 5
输出:0
解释:没有匹配到这样的 x!，符合K = 5 的条件。*/
class P793 {
public:
    int preimageSizeFZF(int K) {
        int left = 0, right = 5L*(K+1);
		cout << right << endl;
		while (left <= right){
			int mid = (left + right)/2;
			long num = numOfTrailingZeroes(mid);
			if (num > K)
				right = mid - 1;
			else if (num < K)
				left = mid + 1;
			else
				return 5;
		}
		return 0;
    }
	long numOfTrailingZeroes(long x){
		long res = 0;
		while(x > 0){
			res += x/5;
			x /= 5;
		}
		return res;
	}
	void test() {
		int K = 24, x = 100;
		cout << "Factorial of " << x << " has " << numOfTrailingZeroes(x) << " trailing zeroes at tail." << endl;
		cout << "There are " << preimageSizeFZF(K) << " numbers which have " << K << " zeroes at tail." << endl;
		return;
	}
};

// 区间子数组个数
/*给定一个元素都是正整数的数组A ，正整数 L 以及 R (L <= R)。
求连续、非空且其中最大元素满足大于等于L 小于等于R的子数组个数。
例如 :
输入: 
A = [2, 1, 4, 3]
L = 2
R = 3
输出: 3
解释: 满足条件的子数组: [2], [2, 1], [3].*/
class P795{
public:
    int numSubarrayBoundedMax(vector<int>& A, int L, int R) {
		int res = 0;
		for(int i = 0; i < A.size(); i++){
			int a = A[i];
			for(int j = i; j < A.size(); j++){
				int tmp = max(A[j], a);
				a = tmp;
				if (tmp >= L && tmp <= R)
					res++;
			}
		}
		return res;
    }
	void test() {
		vector<int> A = {2, 1, 4, 3};
		int L = 2, R = 3;
		int res = numSubarrayBoundedMax(A, L, R);
		cout << "numSubarrayBoundedMax is " << res << endl;
		return;
	}
};


// 旋转字符串
/*给定两个字符串, A 和 B。
A 的旋转操作就是将 A 最左边的字符移动到最右边。 例如, 若 A = 'abcde'，在移动一次之后结果就是'bcdea' 。如果在若干次旋转操作之后，A 能变成B，那么返回True。

示例 1:
输入: A = 'abcde', B = 'cdeab'
输出: true

示例 2:
输入: A = 'abcde', B = 'abced'
输出: false*/

class P796 {
public:
    bool rotateString(string A, string B) {
        if (A.size() != B.size())
			return false;
		if (A.size() == 0)
			return true;
		for (int i = 0; i != A.size(); i++){
			string left = A.substr(0, i), right = A.substr(i, A.size() - i);
			if (right + left == B)
				return true;
		}
		return false;
    }
	void test(){
		string A = "", B = "";
		if (rotateString(A, B))
			cout << A << " and " << B << " is a rotation" << endl;
		else
			cout << A << " and " << B << " is not a rotation" << endl;
		return;
	}
};

// All Paths From Source to Target
/*Given a directed, acyclic graph of N nodes.  Find all possible paths from node 0 to node N-1, and return them in any order.
The graph is given as follows:  the nodes are 0, 1, ..., graph.length - 1.  graph[i] is a list of all nodes j for which the edge (i, j) exists.

Example:
Input: [[1,2], [3], [3], []] 
Output: [[0,1,3],[0,2,3]] 
Explanation: The graph looks like this:
0--->1
|    |
v    v
2--->3
There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.*/

class P797 {
public:
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        return graph;
    }
	void test(){
		vector<vector<int>> graph = {{1, 2}, {3}, {3}, {}};
		vector<vector<int>> paths = allPathsSourceTarget(graph);
		for (int i = 0; i < paths.size(); i++){
			for (int j = 0; j < paths[i].size(); j++)
				cout << paths[i][j] << "->";
			cout << endl;
		}
		return;
	}
};

// 唯一摩尔斯密码词
/*国际摩尔斯密码定义一种标准编码方式，将每个字母对应于一个由一系列点和短线组成的字符串， 比如: "a" 对应 ".-", "b" 对应 "-...", "c" 对应 "-.-.", 等等。
为了方便，所有26个英文字母对应摩尔斯密码表如下：
[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
给定一个单词列表，每个单词可以写成每个字母对应摩尔斯密码的组合。例如，"cab" 可以写成 "-.-.-....-"，(即 "-.-." + "-..." + ".-"字符串的结合)。我们将这样一个连接过程称作单词翻译。
返回我们可以获得所有词不同单词翻译的数量。
例如:
输入: words = ["gin", "zen", "gig", "msg"]
输出: 2
解释: 
各单词翻译如下:
"gin" -> "--...-."
"zen" -> "--...-."
"gig" -> "--...--."
"msg" -> "--...--."

共有 2 种不同翻译, "--...-." 和 "--...--.".*/
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


// 写字符串需要的行数
/*我们要把给定的字符串 S 从左到右写到每一行上，每一行的最大宽度为100个单位，如果我们在写某个字母的时候会使这行超过了100 个单位，那么我们应该把这个字母写到下一行。我们给定了一个数组 widths ，这个数组 widths[0] 代表 'a' 需要的单位， widths[1] 代表 'b' 需要的单位，...， widths[25] 代表 'z' 需要的单位。
现在回答两个问题：至少多少行能放下S，以及最后一行使用的宽度是多少个单位？将你的答案作为长度为2的整数列表返回。
示例 1:
输入: 
widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
S = "abcdefghijklmnopqrstuvwxyz"
输出: [3, 60]
解释: 
所有的字符拥有相同的占用单位10。所以书写所有的26个字母，
我们需要2个整行和占用60个单位的一行。
示例 2:
输入: 
widths = [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
S = "bbbcccdddaaa"
输出: [2, 4]
解释: 
除去字母'a'所有的字符都是相同的单位10，并且字符串 "bbbcccdddaa" 将会覆盖 9 * 10 + 2 * 4 = 98 个单位.
最后一个字母 'a' 将会被写到第二行，因为第一行只剩下2个单位了。
所以，这个答案是2行，第二行有4个单位宽度。*/

class P806 {
public:
    vector<int> numberOfLines(vector<int>& widths, string S) {
		if (S.length() == 0){
			return {0, 0};
		}
		int digits = 0, line = 1;
		for (int i = 0; i != S.size(); i++){
			digits += widths[S[i] - 'a'];
			if (digits > 100){
				digits = 0;
				line++;
				i--;
			}
		}
		return {line, digits};
    }
	void test(){
		vector<int> widths = {4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10};
		string S = "bbbcccdddaaa";
		vector<int> res = numberOfLines(widths, S);
		cout << "number of lines is " << res[0] << ", number of units in last line is " << res[1] << endl;
		return;
	}
};

// 分汤
/*
有 A 和 B 两种类型的汤。一开始每种类型的汤有 N 毫升。有四种分配操作：
提供 100ml 的汤A 和 0ml 的汤B。
提供 75ml 的汤A 和 25ml 的汤B。
提供 50ml 的汤A 和 50ml 的汤B。
提供 25ml 的汤A 和 75ml 的汤B。
当我们把汤分配给某人之后，汤就没有了。每个回合，我们将从四种概率同为0.25的操作中进行分配选择。如果汤的剩余量不足以完成某次操作，我们将尽可能分配。当两种类型的汤都分配完时，停止操作。
注意不存在先分配100 ml汤B的操作。
需要返回的值： 汤A先分配完的概率 + 汤A和汤B同时分配完的概率 / 2。
示例:
输入: N = 50
输出: 0.625
解释:
如果我们选择前两个操作，A将首先变为空。对于第三个操作，A和B会同时变为空。对于第四个操作，B将首先变为空。
所以A变为空的总概率加上A和B同时变为空的概率的一半是 0.25 *(1 + 1 + 0.5 + 0)= 0.625。*/
class P808 {
public:
    double soupServings(int N) {
		vector<pair<bool, bool>> soup;
		pair<bool, bool> tmp;
		tmp.first = (N <= 100);

        return 0.0;
    }
	void test() {
		int N = 50;
		cout << soupServings(N) << endl;
		return;
	}
};

// 子域名访问计数
/*一个网站域名，如"discuss.leetcode.com"，包含了多个子域名。作为顶级域名，常用的有"com"，下一级则有"leetcode.com"，最低的一级为"discuss.leetcode.com"。当我们访问域名"discuss.leetcode.com"时，也同时访问了其父域名"leetcode.com"以及顶级域名 "com"。
给定一个带访问次数和域名的组合，要求分别计算每个域名被访问的次数。其格式为访问次数+空格+地址，例如："9001 discuss.leetcode.com"。
接下来会给出一组访问次数和域名组合的列表cpdomains 。要求解析出所有域名的访问次数，输出格式和输入格式相同，不限定先后顺序。
示例 1:
输入: 
["9001 discuss.leetcode.com"]
输出: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
说明: 
例子中仅包含一个网站域名："discuss.leetcode.com"。按照前文假设，子域名"leetcode.com"和"com"都会被访问，所以它们都被访问了9001次。
示例 2
输入: 
["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
输出: 
["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
说明: 
按照假设，会访问"google.mail.com" 900次，"yahoo.com" 50次，"intel.mail.com" 1次，"wiki.org" 5次。
而对于父域名，会访问"mail.com" 900+1 = 901次，"com" 900 + 50 + 1 = 951次，和 "org" 5 次。*/
class P811 {
public:
	vector<string> subdomainVisits(vector<string>& cpdomains) {
		map<string, int> domains;
		vector<string> res;
		for (int i = 0; i < cpdomains.size(); i++) {
			string cpdomain = cpdomains[i];
			vector<string> addresses;
			size_t index = cpdomain.find_first_of(" ", 0);
			string temp = cpdomain.substr(index + 1, cpdomain.size() - 1);
			int cnt = stoi(cpdomain.substr(0, index));
			size_t last = 0, ind = temp.find_first_of(".", last);;
			while (ind != string::npos) {
				addresses.push_back(temp.substr(last, temp.size()));
				last = ind + 1;
				ind = temp.find_first_of(".", last);
			}
			if (ind - last > 0) 
				addresses.push_back(temp.substr(last, ind - last));
			for (int k = 0; k < addresses.size(); k++) {
				string address = addresses[k];
				if (domains.find(address) == domains.end())
					domains[address] = cnt;
				else
					domains[address] += cnt;
			}
		}
		for (map<string, int>::iterator it = domains.begin(); it != domains.end(); it++)
			res.push_back(to_string(it->second) + " " + it->first);
		return res;
	}
	void test() {
		vector<string> cpdomains = { "900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org" };
		vector<string> visitdomains = subdomainVisits(cpdomains);
		for (vector<string>::iterator it = visitdomains.begin(); it != visitdomains.end(); it++)
			cout << *it << "; ";
		cout << endl;
		return;
	}
};