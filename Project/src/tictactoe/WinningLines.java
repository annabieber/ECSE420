package tictactoe;

import java.awt.Point;
import java.util.LinkedList;

public class WinningLines {

    private final LinkedList<WinningLine> lines = new LinkedList<>();
    private static WinningLines winningLines;
    private  WinningLines(){
        addWinningLines();
    }

    private void addWinningLines() {
        addWinningRows();
        addWinningColumns();
        addTopLeftBottomRightWinningLine();
        addTopRightBottomLeftWinningLine();
    }

    private void addWinningRows() {
        for (int row = 0; row < Properties.BOARD_DIM; row++) {
            WinningLine winningLine = new WinningLine();
            for (int col = 0; col < Properties.BOARD_DIM; col++) {
                winningLine.addPoint(new Point(col, row));
            }
            lines.add(winningLine);
        }
    }

    private void addWinningColumns() {
        for (int col = 0; col < Properties.BOARD_DIM; col++) {
            WinningLine winningLine = new WinningLine();
            for (int row = 0; row < Properties.BOARD_DIM; row++) {
                winningLine.addPoint(new Point(col, row));
            }
            lines.add(winningLine);
        }
    }

    private void addTopLeftBottomRightWinningLine() {
        WinningLine line = new WinningLine();
        for(int i = 0; i < Properties.BOARD_DIM; i++){
            line.addPoint(new Point(i, i));
        }
        lines.add(line);
    }

    private void addTopRightBottomLeftWinningLine() {
        WinningLine topRightBottomLeft = new WinningLine();
        for(int i = 0; i < Properties.BOARD_DIM; i++){
            topRightBottomLeft.addPoint(new Point(Properties.BOARD_DIM - i - 1, i));
        }
        lines.add(topRightBottomLeft);
    }

    public static WinningLines getInstance(){
        if (winningLines == null){
            winningLines = new WinningLines();
        }
        return winningLines;
    }

    public LinkedList<WinningLine> getWinningLines(){
        return lines;
    }
}
