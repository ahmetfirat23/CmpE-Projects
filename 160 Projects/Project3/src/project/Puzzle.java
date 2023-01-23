
package project;

import java.util.Arrays;
import java.util.Stack;

public class Puzzle {
	private final int[][] tiles;

	public Puzzle(int[][] tiles) {
		this.tiles = new int[tiles.length][tiles.length];
		for (int i = 0; i < tiles.length; i++) {
			this.tiles[i] = tiles[i].clone();
		}
	}

	public String toString() {
		StringBuilder str = new StringBuilder();
		str.append(tiles.length + "\n");
		for (int i = 0; i < tiles.length; i++) {
			for (int j = 0; j < tiles[i].length; j++) {
				str.append(" " + tiles[i][j]);
			}
			str.append("\n");
		}
		return str.toString();

	}

	// Returns height of square
	public int dimension() {
		return tiles.length;
	}

	// sum of Manhattan distances between tiles and goal
	// The Manhattan distance between a board and the goal board is the sum
	// of the Manhattan distances (sum of the vertical and horizontal distance)
	// from the tiles to their goal positions.
	public int h() {
		int distance = 0;
		for (int y = 0; y < dimension(); y++) {
			for (int x = 0; x < dimension(); x++) {
				int elem = tiles[y][x];
				int vdistance;
				int hdistance;
				if (elem == 0) {
					vdistance = 0;
					hdistance = 0;
				} else {
					vdistance = Math.abs(y - (elem - 1) / dimension());
					hdistance = Math.abs(x - (elem - 1) % dimension());
				}
				distance += vdistance + hdistance;
			}
		}
		return distance;
	}

	// Returns whether target is reached
	public boolean isCompleted() {
		for (int y = 0; y < tiles.length; y++) {
			for (int x = 0; x < tiles.length; x++) {
				int elem = tiles[y][x];
				if (elem == 0) {
					if (y == (tiles.length - 1) && x == (tiles.length - 1)) {
						break;
					}
					return false;
				}
				if (y == (elem - 1) / tiles.length && x == (elem - 1) % tiles.length) {
					continue;
				}
				return false;
			}
		}
		return true;
	}

	// Returns any kind of collection that implements iterable.
	// For this implementation, I choose stack.
	// Locates position of 0 and does appropriate switches
	public Iterable<Puzzle> getAdjacents() {
		Stack<Puzzle> stack = new Stack<>();
		int y = 0, x = 0;
		for (int i = 0; i < dimension(); i++) {
			for (int j = 0; j < dimension(); j++) {
				if (tiles[i][j] == 0) {
					y = i;
					x = j;
					break;
				}
			}
		}

		if (y == 0) {
			if (x == 0) {
				stack.push(switchDown(this, y, x));
				stack.push(switchRight(this, y, x));
			} else if (x == dimension() - 1) {
				stack.push(switchDown(this, y, x));
				stack.push(switchLeft(this, y, x));
			} else {
				stack.push(switchDown(this, y, x));
				stack.push(switchLeft(this, y, x));
				stack.push(switchRight(this, y, x));
			}
		} else if (y == dimension() - 1) {
			if (x == 0) {
				stack.push(switchUp(this, y, x));
				stack.push(switchRight(this, y, x));
			} else if (x == dimension() - 1) {
				stack.push(switchUp(this, y, x));
				stack.push(switchLeft(this, y, x));
			} else {
				stack.push(switchUp(this, y, x));
				stack.push(switchLeft(this, y, x));
				stack.push(switchRight(this, y, x));
			}
		} else {
			if (x == 0) {
				stack.push(switchUp(this, y, x));
				stack.push(switchDown(this, y, x));
				stack.push(switchRight(this, y, x));
			} else if (x == dimension() - 1) {
				stack.push(switchUp(this, y, x));
				stack.push(switchDown(this, y, x));
				stack.push(switchLeft(this, y, x));
			} else {
				stack.push(switchUp(this, y, x));
				stack.push(switchDown(this, y, x));
				stack.push(switchLeft(this, y, x));
				stack.push(switchRight(this, y, x));
			}
		}

		return stack;
	}

	private Puzzle switchDown(Puzzle puzzle, int y, int x) {
		int[][] tiles = copy(puzzle.tiles);
		tiles[y][x] = tiles[y + 1][x];
		tiles[y + 1][x] = 0;
		return new Puzzle(tiles);
	}

	private Puzzle switchUp(Puzzle puzzle, int y, int x) {
		int[][] tiles = copy(puzzle.tiles);
		tiles[y][x] = tiles[y - 1][x];
		tiles[y - 1][x] = 0;
		return new Puzzle(tiles);
	}

	private Puzzle switchRight(Puzzle puzzle, int y, int x) {
		int[][] tiles = copy(puzzle.tiles);
		tiles[y][x] = tiles[y][x + 1];
		tiles[y][x + 1] = 0;
		return new Puzzle(tiles);
	}

	private Puzzle switchLeft(Puzzle puzzle, int y, int x) {
		int[][] tiles = copy(puzzle.tiles);
		tiles[y][x] = tiles[y][x - 1];
		tiles[y][x - 1] = 0;
		return new Puzzle(tiles);
	}

	// Returns board in new array
	private int[][] copy(int[][] source) {
		int[][] tiles = new int[source.length][source.length];
		for (int i = 0; i < source.length; i++) {
			tiles[i] = source[i].clone();
		}
		return tiles;
	}

	// Checks board is equal
	@Override
	public boolean equals(Object o) {
		Puzzle puzzle = (Puzzle) o;
		for (int i = 0; i < dimension(); i++) {
			for (int j = 0; j < dimension(); j++) {
				if (tiles[i][j] != puzzle.tiles[i][j]) {
					return false;
				}
			}
		}
		return true;
	}

	// Hashes board's position
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.deepHashCode(tiles);
		return result;
	}

	// You can use this main method to see your Puzzle structure.
	// Actual solving operations will be conducted in Solver.main method
	/*
	 * public static void main(String[] args) { int[][] array = { { 8, 1, 3 }, { 4,
	 * 0, 2 }, { 7, 6, 5 } }; Puzzle board = new Puzzle(array);
	 * System.out.println(board); System.out.println(board.dimension());
	 * System.out.println(board.h()); System.out.println(board.isCompleted());
	 * Iterable<Puzzle> itr = board.getAdjacents(); for (Puzzle neighbor : itr) {
	 * System.out.println(neighbor); System.out.println(neighbor.equals(board)); } }
	 */
}
