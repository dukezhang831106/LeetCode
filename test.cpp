#include <iostream>
#include <vector>
#include "Problems.hpp"

using namespace std;

// Test Case for P543
int main(){
	P543 prob;
	TreeNode* tree = new TreeNode(1);
	tree->left = new TreeNode(2);
	tree->left->left = new TreeNode(4);
	tree->left->right = new TreeNode(5);
	tree->right = new TreeNode(3);
	int diam = prob.diameterOfBinaryTree(tree);
	cout << "Diameter of the tree is " << diam << endl;
	return 0;	
}
