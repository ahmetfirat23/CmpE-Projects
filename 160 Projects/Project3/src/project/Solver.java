
package project;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Scanner;
import java.util.Set;

public class Solver {
	private PriorityObject winner;

	// priority = moves + manhattan
	// if priority is low, it's good.
	// find a solution to the initial board
	public Solver(Puzzle root) {
		System.out.println("Starting the solver...");
		if (root == null)
			throw new IllegalArgumentException();
		solve(root);
		System.out.println("Solving is finished...");
	}

	// Scans through unique nodes of puzzle
	// Utilizes priority queue for optimization
	// When goal board is surely found no further search is done
	private void solve(Puzzle root) {
		PriorityQueue<PriorityObject> stack = new PriorityQueue<>(new CustomComparator());
		PriorityQueue<PriorityObject> solutionQueue = new PriorityQueue<>(new CustomComparator());
		Set<Puzzle> set = new HashSet<>();
		PriorityObject rootObject = new PriorityObject(root, 0, null);
		stack.add(rootObject);
		while (!stack.isEmpty()) {
			PriorityObject currentObject = stack.poll();
			if (set.contains(currentObject.board)) {
				continue;
			}
			set.add(currentObject.board);
			if (!solutionQueue.isEmpty() && solutionQueue.peek().f <= currentObject.g) {
				break;
			}
			if (currentObject.board.isCompleted()) {
				if (!solutionQueue.isEmpty() && solutionQueue.peek().f <= currentObject.f) {
					break;
				}
				solutionQueue.add(currentObject);
				continue;
			}
			Iterator<Puzzle> iterator = currentObject.board.getAdjacents().iterator();
			while (iterator.hasNext()) {
				Puzzle nextPuzzle = iterator.next();
				PriorityObject nextObject = new PriorityObject(nextPuzzle, currentObject.g + 1, currentObject);
				if (!solutionQueue.isEmpty() && nextObject.g > solutionQueue.peek().f) {
					continue;
				}
				if (set.contains(nextObject.board)) {
					continue;
				}
				stack.add(nextObject);
			}
		}
		winner = solutionQueue.peek();
	}

	// Returns minimum number of moves required
	public int getMoves() {
		return winner.g;
	}

	// Returns solution in linked list
	public Iterable<Puzzle> getSolution() {
		LinkedList<Puzzle> solution = new LinkedList<>();
		PriorityObject iterObject = winner;
		solution.addFirst(winner.board);
		while (iterObject.prevObject != null) {
			iterObject = iterObject.prevObject;
			solution.addFirst(iterObject.board);
		}
		return solution;
	}

	private class PriorityObject {

		private Puzzle board;
		private int f;
		private PriorityObject prevObject;
		private int g;

		public PriorityObject(Puzzle board, int g, PriorityObject prev) {
			this.board = board;
			this.g = g;
			this.prevObject = prev;
			f = g + board.h();
		}
	}

	// Order in increasing f order and if f equal g order
	private class CustomComparator implements Comparator<PriorityObject> {
		public int compare(PriorityObject o1, PriorityObject o2) {
			if (o1.board.isCompleted() && o2.board.isCompleted()) {
				return 1;
			} else if (o1.board.isCompleted() && !o2.board.isCompleted()) {
				return -1;
			} else if (o1.f == o2.f) {
				return o1.g - o2.g;
			}
			return o1.f - o2.f;
		}
	}

	// Converts string array into integer array
	private static int[] strArrToInt(String[] strArr) {
		int[] intArr = new int[strArr.length];
		for (int i = 0; i < strArr.length; i++) {
			intArr[i] = Integer.parseInt(strArr[i]);
		}
		return intArr;
	}

	// Checks inversions (count of smaller numbers come after a big number)
	// If total inversions are even puzzle is solvable
	private static boolean isSolvable(int[][] arr) {
		if(arr.length%2==0) {
			return true;
		}
		int check = 0;
		int count = 0;
		int[] arr1d = new int[arr.length * arr.length];
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr.length; j++) {
				arr1d[i * arr.length + j] = arr[i][j];
			}
		}
		for (int i = 0; i < arr1d.length; i++) {
			check = arr1d[i];
			if (check == 0) {
				continue;
			}
			for (int j = i; j < arr1d.length; j++) {
				int other = arr1d[j];
				if (other == 0) {
					continue;
				} else if (check > other) {
					count++;
				}
			}
		}

		return count % 2 == 0;
	}

	// test client
	public static void main(String[] args) throws IOException {

		File input = new File("input.txt");
		// Read this file int by int to create
		// the initial board (the Puzzle object) from the file
		Scanner scanner = new Scanner(input);
		int n = Integer.parseInt(scanner.nextLine());
		int[][] tiles = new int[n][n];
		for (int i = 0; i < n; i++) {
			tiles[i] = strArrToInt(scanner.nextLine().split(" "));
		}
		scanner.close();
		Puzzle initial = new Puzzle(tiles);

		if (!isSolvable(tiles)) {
			File output = new File("output.txt");
			output.createNewFile();
			PrintStream write = new PrintStream(output);
			write.println("Board is unsolvable");
			write.close();
			return;
		}

		// solve the puzzle here. Note that the constructor of the Solver class already
		// calls the
		// solve method. So just create a solver object with the Puzzle Object you
		// created above
		// is given as argument, as follows:
		Solver solver = new Solver(initial); // where initial is the Puzzle object created from input file
		// You can use the following part as it is. It creates the output file and fills
		// it accordingly.
		File output = new File("output.txt");
		output.createNewFile();
		PrintStream write = new PrintStream(output);
		write.println("Minimum number of moves = " + solver.getMoves());
		for (Puzzle board : solver.getSolution())
			write.println(board);
		write.close();
	}
}
