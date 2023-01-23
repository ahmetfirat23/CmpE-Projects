package project.airport;

import java.util.ArrayList;

import project.airline.aircraft.Aircraft;
import project.passenger.Passenger;

public abstract class Airport {

	private final int ID;
	private final double x, y; // Coordinates of the airport
	protected double fuelCost;
	protected double operationFee;
	protected int aircraftCapacity;
	protected ArrayList<Aircraft> aircrafts = new ArrayList<Aircraft>();
	protected ArrayList<Passenger> passengers = new ArrayList<Passenger>();

	public Airport(int ID, double x, double y, double fuelCost, double operationFee, int aircraftCapacity) {
		this.ID = ID;
		this.x = x;
		this.y = y;
		this.fuelCost = fuelCost;
		this.operationFee = operationFee;
		this.aircraftCapacity = aircraftCapacity;
	}

	public int getID() {
		return ID;
	}

	public double getX() {
		return x;
	}

	public double getY() {
		return y;
	}

	public double getFuelCost() {
		return fuelCost;
	}

	public double getOperationFee() {
		return operationFee;
	}

	public int getAircraftCapacity() {
		return aircraftCapacity;
	}

	private double getAircraftRatio() {
		return aircrafts.size() * 1.0 / getAircraftCapacity();
	}

	protected double getFullnessCoefficient() {
		return 0.6 * Math.pow(Math.E, getAircraftRatio());
	}

	protected double getAircraftWeightRatio(Aircraft aircraft) {
		return aircraft.getWeightRatio();
	}

	// Does the departure operations and returns the departure fee
	public abstract double departAircraft(Aircraft aircraft);

	// Does the landing operations and returns the landing fee
	public abstract double landAircraft(Aircraft aircraft);

	// Returns whether airport's aircraft capacity is full
	public boolean isFull() {
		return aircraftCapacity == aircrafts.size();
	}

	// Returns distance between airports
	public double getDistance(Airport airport) {
		return Math.sqrt(Math.pow(x - airport.getX(), 2) + Math.pow(y - airport.getY(), 2));
	}

	public void addPassenger(Passenger passenger) {
		passengers.add(passenger);
	}

	public void removePassenger(Passenger passenger) {
		passengers.remove(passenger);
	}

	public boolean addAircraft(Aircraft aircraft) {
		if (aircraftCapacity == aircrafts.size()) {
			return false;
		}
		aircrafts.add(aircraft);
		return true;
	}
}
