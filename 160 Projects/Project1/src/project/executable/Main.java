package project.executable;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Scanner;

import project.airline.Airline;

public class Main {

	public static void main(String[] args) throws FileNotFoundException {
		// Directs all outputs to the given second file
		File outFile = new File(args[1]);
		PrintStream pStream = new PrintStream(outFile);
		System.setOut(pStream);

		// Scans the first file and does the necessary creations and assignments
		File file = new File(args[0]);
		Scanner scanner = new Scanner(file);

		String[] firstLine = scanner.nextLine().split(" ");
		int maxAircraftCount = Integer.parseInt(firstLine[0]);
		int airportsCount = Integer.parseInt(firstLine[1]);
		int passengersCount = Integer.parseInt(firstLine[2]);

		String[] secondLine = scanner.nextLine().split(" ");
		double propOpFee = Double.parseDouble(secondLine[0]);
		double widebodyOpFee = Double.parseDouble(secondLine[1]);
		double rapidOpFee = Double.parseDouble(secondLine[2]);
		double jetOpFee = Double.parseDouble(secondLine[3]);
		double airlineOpFee = Double.parseDouble(secondLine[4]);

		Airline airline = new Airline(maxAircraftCount, airlineOpFee);
		createAirports(scanner, airportsCount, airline);
		createPassengers(scanner, passengersCount, airline);

		scanner.close();

		// Executes flight plans and prints profit
		airline.jetFlightPlan(jetOpFee);
		airline.propFlightPlan(propOpFee);
		airline.wideFlightPlan(widebodyOpFee);

		airline.printProfit();
	}

	private static void createAirports(Scanner scanner, int airportsCount, Airline airline) {
		for (int i = 0; i < airportsCount; i++) {
			String[] line = scanner.nextLine().split("\s:\s|,\s");
			String airportType = line[0];
			int ID = Integer.parseInt(line[1]);
			double x = Double.parseDouble(line[2]);
			double y = Double.parseDouble(line[3]);
			double fuelCost = Double.parseDouble(line[4]);
			double operationFee = Double.parseDouble(line[5]);
			int aircraftCapacity = Integer.parseInt(line[6]);
			airline.createAirport(airportType, ID, x, y, fuelCost, operationFee, aircraftCapacity);
		}
	}

	private static void createPassengers(Scanner scanner, int passengersCount, Airline airline) {
		for (int i = 0; i < passengersCount; i++) {
			String[] line = scanner.nextLine().split("\s:\s|(?<!\\[.*),\s");
			String passengerType = line[0];
			long ID = Long.parseLong(line[1]);
			double weight = Double.parseDouble(line[2]);
			int baggageCount = Integer.parseInt(line[3]);
			String[] destStrings = line[4].replaceAll("\\[|\\]", "").split(",\s");
			ArrayList<Integer> destinations = new ArrayList<Integer>();
			for (String dest : destStrings) {
				destinations.add(Integer.parseInt(dest));
			}
			airline.createPassenger(passengerType, ID, weight, baggageCount, destinations);
		}
	}

}
