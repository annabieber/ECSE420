package tictactoe;

import java.util.concurrent.ConcurrentHashMap;

public class MctsThread implements Runnable {
    Thread t;
    ConcurrentHashMap<String, Integer> movesChosen;
    TicTacToePlayer player;
    TicTacToeGame game;
    String name;

    MctsThread (String name, TicTacToePlayer player, TicTacToeGame game, ConcurrentHashMap<String, Integer> movesChosen){
        this.name = name;
        this.player = player;
        this.game = game.getCopy();
        this.movesChosen = movesChosen;
        t = new Thread(this, name);
        System.out.println(" ~ thread " + t.getName() + " started");
        t.start();
    }

    public void run() {
        try {
            long startTime = System.nanoTime();
            TicTacToeMove move = this.player.getMove(this.game);
            long elapsedTime = System.nanoTime() - startTime;

            String moveStr = "[x = " + move.getPosition().getX() + ", y = " + move.getPosition().getY() + "]";
            System.out.println(" -> thread " + t.getName() + " built a MTCS tree in [" + elapsedTime/1000000
                    + " milliseconds] that chooses move: " + moveStr + "\n");

            String moveStrKey = "(" + (int) move.getPosition().getX() + "," + (int) move.getPosition().getY() + ")";
            movesChosen.put(moveStrKey, 1 + movesChosen.getOrDefault(moveStrKey, 0));
        }catch (Exception e) {
            System.out.println(e);
        }
    }
}
