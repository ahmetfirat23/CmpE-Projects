package project.Executable;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Scanner;

import project.Senate.Senate;

/**
 * This is the main class of the project. Desired simulations can be executed
 * here by using Senate functions. In order to execute a simulation required
 * string manipulations and checks are done and inputs are transmitted to the
 * Senate.
 * 
 * @author Firat
 *
 */
public class Main {

	public static void main(String[] args) throws FileNotFoundException {
		// Directs all outputs to the given second file
		File outFile = new File(args[1]);
		PrintStream pStream = new PrintStream(outFile);
		System.setOut(pStream);

		// Scans the first file
		File file = new File(args[0]);
		Scanner scanner = new Scanner(file);

		// Creates the conductor class prepares it for use
		Senate senate = new Senate();
		senate.adjustLists();

		// Generates sectors
		int sectorCount = Integer.parseInt(scanner.nextLine());
		for (int i = 1; i <= sectorCount; i++) {
			senate.createSector(scanner.nextLine(), i);
		}

		// Generates crewmans
		int crewmanCount = Integer.parseInt(scanner.nextLine());
		for (int i = 1; i <= crewmanCount; i++) {
			String[] inputs = scanner.nextLine().split(" ");
			String type = inputs[0];
			String input = String.join(" ", Arrays.copyOfRange(inputs, 1, inputs.length));

			switch (type) {
			case "Officer":
				senate.createOfficer(input, i);
				continue;
			case "Jedi":
				senate.createJedi(input, i);
				continue;
			case "Sith":
				senate.createSith(input, i);
				continue;
			}

		}

		// Generates ships
		int shipCount = Integer.parseInt(scanner.nextLine());
		for (int i = 1; i <= shipCount; i++) {
			senate.createShip(scanner.nextLine(), scanner.nextLine(), i);
		}

		// Simulates given attacks
		int eventCount = Integer.parseInt(scanner.nextLine());
		for (int i = 1; i <= eventCount; i++) {
			String[] inputs = scanner.nextLine().split(" ");
			int eventID = Integer.parseInt(inputs[0]);
			String input = String.join(" ", Arrays.copyOfRange(inputs, 1, inputs.length));
			switch (eventID) {
			case 10:
				senate.simulateAttack(input);
				continue;
			case 11:
				senate.simulateAssault(input);
				continue;
			case 20:
				senate.relocate(input);
				continue;
			case 30:
				senate.visitCorousant(input);
				continue;
			case 40:
				senate.addCrewman(input);
				continue;
			case 41:
				senate.removeCrewman(input);
				continue;
			case 50:
				senate.trainOfficer(input);
				continue;
			case 51:
				senate.upgradeShip(input);
				continue;
			}
		}
		scanner.close();

		senate.logWarships();
		senate.logCrewmans();
	}
}