#include <stdio.h>

class Node {
	int dept;
	double heuristic;
public:
	bool status;
	int depth;
	int xPos;
	int yPos;
	Node* root;
	Node* parent;
	Node* childen[9];
	int key;
	double getHeuristic();
	void Node::setHeuristic(double d);
	void createRoot(Node node);
	Node createNode(Node* parent, double heuristic, int x, int y);
	void setCoordinates(int x, int y);
	void Node::setStatus(bool sta);
	int Node::numChildren(Node node);
	int Node::getKey();
};

int Node::getKey() {
	return key;
}

int Node::numChildren(Node node) {
	int count;
	for (Node* child : node.childen) {
		count++;
	}
	return count;
}

void Node::setHeuristic(double d) {
	heuristic = d;
}

void Node::setStatus(bool sta) {
	status = sta;
}

void Node::setCoordinates(int x, int y) {
	xPos = x;
	yPos = y;
}

double Node::getHeuristic() {
	return heuristic;
}

void Node::createRoot(Node node) {
	root = &node;
	for (int i = 0; i < sizeof(root->childen); i++) {
		root->childen[i] = NULL;
	}
}

Node Node::createNode(Node* parent, double heuristic, int x, int y) {
	Node newNode;
	newNode.xPos = x;
	newNode.yPos = y;
	for (int i = 0; i < sizeof(&parent->childen); i++) {
		if (&parent->childen[i] == NULL) {
			newNode.heuristic = heuristic;
			parent->childen[i] = &newNode;
		}
	}
	return newNode;
}