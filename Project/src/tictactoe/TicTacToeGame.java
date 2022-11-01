package tictactoe;

import java.awt.Point;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static tictactoe.GameSymbols.Player1Symbol;
import static tictactoe.GameSymbols.Player2Symbol;

public class TicTacToeGame {

    protected final TicTacToeBoard board;
    private int playerToMove;
    private final HashMap<Integer, Character> playerSymbols = new HashMap<>();

    public TicTacToeGame(TicTacToeBoard board) {
        this.board = board;
        playerSymbols.put(0, Player1Symbol);
        playerSymbols.put(1, Player2Symbol);
    }

    public boolean gameIsOver() {
        return boardContainsWinForSymbol(Player1Symbol)
            || boardContainsWinForSymbol(Player2Symbol)
            || board.getEmptyPositions().size() == 0;
    }

    public boolean boardContainsWinForSymbol(char symbol) {
        for (WinningLine line : WinningLines.getInstance().getWinningLines()) {
            if (lineContainsWinForSymbol(line, symbol)) {
                return true;
            }
        }
        return false;
    }

    private boolean lineContainsWinForSymbol(WinningLine line, char symbol) {
        for (Point p : line.getPoints()) {
            if (board.getSymbolAtPoint(p) != symbol) {
                return false;
            }
        }
        return true;
    }

    public void play(TicTacToePlayer player1, TicTacToePlayer player2) {
        LinkedList<TicTacToePlayer> players = new LinkedList<>();
        players.add(player1);
        players.add(player2);
        while (!gameIsOver()) {
            TicTacToePlayer player = players.get(playerToMove);
            TicTacToeMove move = player.getMove(this);

            ConcurrentHashMap<String, Integer> moves = new ConcurrentHashMap<>();

            if(player instanceof MctsTicTacToePlayer) {
                if(!Properties.runParallel){
                    // sequential

                    long startTime = System.nanoTime();
                    for (int i = 0; i < Properties.NUM_THREADS_ROOT_PARALLEL; i++) {
                        move = player.getMove(this);
                    }

                    long elapsedTime = System.nanoTime() - startTime;
                    System.out.println("\n *** SEQUENTIAL TIME: " + elapsedTime / 1000000 + "\n");
                } else {
                    System.out.println("\n*** PARALLEL RESULTS ***\n");

                    for (int i = 0; i < Properties.NUM_THREADS_ROOT_PARALLEL; i++) {
                        new MctsThread("" + i, player, this, moves);
                    }

                    try {
                        System.out.println(" +++ SLEEP TIME: " + (3 * this.getAvailableMoves().size() / (Math.pow(Properties.BOARD_DIM, 2)) * Properties.BOARD_DIM));
                        Thread.sleep(1000 * (long) (3 * this.getAvailableMoves().size() / Math.pow(Properties.BOARD_DIM, 2)) * Properties.BOARD_DIM);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    System.out.println(" *** (MOVE, OCCURENCES) MAP: " + moves.toString());

                    int maxSimulations = 0;
                    for (Map.Entry<String, Integer> e : moves.entrySet()) {
                        if (e.getValue() > maxSimulations) {
                            maxSimulations = e.getValue();
                            int x = Integer.parseInt(e.getKey().split(",")[0].substring(1));
                            String yStr = e.getKey().split(",")[1];
                            int y = Integer.parseInt(yStr.substring(0, yStr.length() - 1));
                            move = new TicTacToeMove(new Point(x, y), move.getSymbol());
                        }
                    }

                    System.out.println(" +++ BEST MOVE: " + "[x = " + move.getPosition().getX() + ", y = " + move.getPosition().getY() + "]");
                }
            }

            makeMove(move);
        }
    }

    public void makeMove(TicTacToeMove chosenMove) {
        board.play(chosenMove.position.x, chosenMove.position.y, chosenMove.symbol);
        switchPlayer();
    }

    public void switchPlayer() {
        playerToMove = getEnemyPlayer(playerToMove);
    }

    public int getEnemyPlayer(int playerNumber) {
        return 1 - playerNumber;
    }

    public LinkedList<TicTacToeMove> getAvailableMoves() {
        LinkedList<TicTacToeMove> moves = new LinkedList<>();

        for (Point p : board.getEmptyPositions()) {
            moves.add(new TicTacToeMove(p, playerSymbols.get(playerToMove)));
        }

        return moves;
    }


    public Reward getReward() {
        if (player1Wins()) {
            return new Reward(1, -1);
        } else if (player2Wins()) {
            return new Reward(-1, 1);
        }
        return new Reward(0, 0);
    }

    private boolean player1Wins() {
        return boardContainsWinForSymbol(GameSymbols.Player1Symbol);
    }

    private boolean player2Wins() {
        return boardContainsWinForSymbol(GameSymbols.Player2Symbol);
    }

    public TicTacToeBoard getBoard() {
        return board;
    }

    public int getPlayerToMove() {
        return playerToMove;
    }

    public TicTacToeGame getCopy(){
        TicTacToeGame copy = new TicTacToeGame(board.getCopy());
        copy.playerToMove = playerToMove;
        return copy;
    }
}
