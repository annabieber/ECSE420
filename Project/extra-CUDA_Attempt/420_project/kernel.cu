
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Node.cpp"
#include "Player.cpp"
#include "Board.cpp"

#include<math.h>

#include<time.h>

#define RAND_MAX 9;
#define GRID_SIZE 3;

bool currentTurn;

__global__ void simulate(Node node, bool currentTurn, Board board);

__global__ void simulate(Node node, bool currentTurn, Board board) {
	int start = rand() % node.numChildren + 1;

	double maxHeuristic = rand() / double(1000);
	int nodeIndex = 0;

	Node* tempNode;

	//go to depth 3
	for (Node* n : node.childen) {
		for (Node* n1 : n->childen) {
			for (Node* n2 : n1->childen) {
				double random = rand()/double(1000);
				n2->setHeuristic(random);
				if (n2->getHeuristic > maxHeuristic) {
					maxHeuristic = n2->getHeuristic;
					nodeIndex = n2->getKey;
					tempNode = n2;
				}
			}
		}
	}

	for (int i = 0; i < tempNode->depth-1; i++) {
		tempNode = tempNode->parent;
	}
	int index = tempNode->key;
	board.getNodeByIndex(index).setStatus(currentTurn);
}

int main() {
	int G = GRID_SIZE;
	currentTurn = 0;
	Node currentNode;
	Board board;

	Player p1;	//p1 will play 0s instead of Xs
	Player p2;	//p2 will play 1s instead of Os

	p1.setPlayerNumder(0);
	p1.setPlayerName("p1");
	p2.setPlayerNumder(1);
	p2.setPlayerName("p2");

	board.populateBoard();
	Node* randomNode;

	int start = rand() % 9 + 1;
	board.setNodeByIndex(start, 0);
	currentNode = board.getNodeByIndex(start);
	currentNode.createRoot(currentNode);
	currentTurn = 1;

	cudaMalloc((void**)&randomNode, G * G * sizeof(randomNode));
	cudaMemcpy(randomNode, board.moves, G * G * sizeof(randomNode), cudaMemcpyHostToDevice);

	play(currentTurn, board);

	cudaFree(randomNode);

	return 0;
}

void play(bool currentTurn, Board board) {
	if (checkWin) {
		printf("game ended");
	}
	double bestHeuristic = 0;
	Node currentNode;
	Node bestNode;

	bestNode.setStatus(currentTurn);

	simulate << <1, 4 >> > (currentNode, currentTurn, board);

	checkWin(board, currentTurn);
	if (currentTurn == 0) {
		currentTurn = 1;
	}
	else {
		currentTurn = 0;
	}
	play(currentTurn, board);
}

bool checkWin(Board board, bool turn) {
	for (Node n : board.moves) {
		if (board.getNodeByCoord(n.xPos + 1, n.yPos).status == currentTurn ||
			board.getNodeByCoord(n.xPos - 1, n.yPos).status == currentTurn ||
			board.getNodeByCoord(n.xPos, n.yPos + 1).status == currentTurn ||
			board.getNodeByCoord(n.xPos, n.yPos + 1).status == currentTurn ||
			board.getNodeByCoord(n.xPos + 1, n.yPos + 1).status == currentTurn ||
			board.getNodeByCoord(n.xPos + 1, n.yPos - 1).status == currentTurn ||
			board.getNodeByCoord(n.xPos - 1, n.yPos + 1).status == currentTurn ||
			board.getNodeByCoord(n.xPos - 1, n.yPos - 1).status == currentTurn) {
			if (board.getNodeByCoord(n.xPos + 2, n.yPos).status == currentTurn ||
				board.getNodeByCoord(n.xPos - 2, n.yPos).status == currentTurn ||
				board.getNodeByCoord(n.xPos, n.yPos + 2).status == currentTurn ||
				board.getNodeByCoord(n.xPos, n.yPos + 2).status == currentTurn ||
				board.getNodeByCoord(n.xPos + 2, n.yPos + 2).status == currentTurn ||
				board.getNodeByCoord(n.xPos + 2, n.yPos - 2).status == currentTurn ||
				board.getNodeByCoord(n.xPos - 2, n.yPos + 2).status == currentTurn ||
				board.getNodeByCoord(n.xPos - 2, n.yPos - 2).status == currentTurn) {
				return true;
			}
		}
	}
}
