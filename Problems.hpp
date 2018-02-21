#include <iostream>
#include "DataStructure.hpp"

using namespace std;


// Given a binary tree, compute the length of the diameter of the tree, defined as the length of the longest path between any two nodes in a tree.

class P543{
public:
	int diameterOfBinaryTree(TreeNode* root){
		int res;
		maxDepth(root, res);
		return res;
	}
	int maxDepth(TreeNode* node, int& res){
		if (!node) return 0;
		int left = maxDepth(node->left, res);
		int right = maxDepth(node->right, res);
		res = max(res, left + right);
		return max(left, right) + 1;
	}
};
