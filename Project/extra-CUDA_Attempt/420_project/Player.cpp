#include <stdio.h>
#include <string>

class Player {
	std::string name;
	int playerNumber;
	bool win = 0;
public:
	int getPlayerNumber();
	void setPlayerNumder(int x);
	void setPlayerName(std::string);
	void setWin();
};

void Player::setPlayerName(std::string s) {
	name = s;
}

void Player::setPlayerNumder(int x) {
	playerNumber = x;
}

int Player::getPlayerNumber() {
	return playerNumber;
}

void Player::setWin() {
	win = 1;
}