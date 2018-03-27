#include <iostream>
#include <queue>
#include <vector>

using namespace std;
//Define Data Strucutre used for OJ

//Definition of a point
struct Point {
	int x;
	int y;
	Point() : x(0), y(0) {}
	Point(int a, int b) : x(a), y(b) {}
};

//Definition of a single linked list
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};

//Definition of a binary tree node
struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

// Definition of an employee
class Employee {
public:
	int id;
	int importance;
	vector<int> subordinates;
	Employee(int a, int i, vector<int> s) : id(a), importance(i), subordinates(s) {}
};


void LevelOrder(TreeNode* root) {
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty()) {
		TreeNode* tmp = q.front();
		q.pop();
		cout << tmp->val << ",";
		if (tmp->left != NULL)
			q.push(tmp->left);
		if (tmp->right != NULL)
			q.push(tmp->right);
	}
	cout << endl;
};

void printListNode(ListNode* head){
	ListNode* p = head;
	while (p){
		cout << p->val << "->";
		p = p->next;
	}
	cout << "NULL" << endl;
}