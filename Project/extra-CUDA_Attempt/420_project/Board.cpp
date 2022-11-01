#include <stdio.h>

#include "Node.cpp"

class Board {
public:
	Node moves[9];
	void populateBoard();
	Node setNodeByIndex(int i, bool status);
	Node getNodeByIndex(int i);
	void setNodeByCoord(int x, int y, bool status);
	Node getNodeByCoord(int x, int y);
};

void Board::populateBoard() {
	moves[0].setCoordinates(0, 0);
	moves[1].setCoordinates(1, 0);
	moves[2].setCoordinates(2, 0);
	moves[3].setCoordinates(0, 1);
	moves[4].setCoordinates(1, 1);
	moves[5].setCoordinates(2, 1);
	moves[6].setCoordinates(0, 2);
	moves[7].setCoordinates(1, 2);
	moves[8].setCoordinates(2, 2);
	moves[0].setStatus(NULL);
	moves[1].setStatus(NULL);
	moves[2].setStatus(NULL);
	moves[3].setStatus(NULL);
	moves[4].setStatus(NULL);
	moves[5].setStatus(NULL);
	moves[6].setStatus(NULL);
	moves[7].setStatus(NULL);
	moves[8].setStatus(NULL);
}

Node Board::getNodeByIndex(int i) {
	return moves[i];
}

Node Board::getNodeByCoord(int x, int y) {
	for (Node n : moves) {
		if (n.xPos == x && n.yPos == y) {
			return n;
		}
	}
}

void Board::setNodeByCoord(int x, int y, bool status) {
	for (Node n : moves) {
		if (n.xPos == x && n.yPos == y) {
			n.status = status;
		}
	}
}

Node Board::setNodeByIndex(int i, bool status) {
	moves[i].setStatus = status;
}