package project.airline.aircraft;

import project.airport.Airport;
import project.interfaces.AircraftInterface;

public abstract class Aircraft implements AircraftInterface {
	public Aircraft(double operationFee, Airport currentAirport, int ID) {
		super();
		this.operationFee = operationFee;
		this.currentAirport = currentAirport;
		this.ID = ID;
	}

	protected double operationFee;
	protected int ID;
	protected Airport currentAirport;
	protected double weight;
	protected double maxWeight;
	protected final double fuelWeight = 0.7;
	protected double fuel;
	public double fuelCapacity;
	protected double aircraftTypeMultiplier;

	// Adjusts fuel, weight and current airport then returns flight cost
	public double fly(Airport toAirport) {
		double fuelConsump = getFuelConsumption(currentAirport.getDistance(toAirport));
		double flightCost = getFlightCost(toAirport);
		fuel -= fuelConsump;
		weight -= fuelConsump * fuelWeight;
		currentAirport = toAirport;
		return flightCost;
	}

	// Check for range limitations
	public boolean checkFlyRange(Airport toAirport) {
		return fuelCapacity >= getFuelConsumption(currentAirport.getDistance(toAirport));
	}

	public boolean checkFly(Airport toAirport) {
		return currentAirport != toAirport && hasFuel(getFuelConsumption(currentAirport.getDistance(toAirport)))
				&& !toAirport.isFull();
	}

	// Calculates required fuel to be loaded for the distance and adds 700 offset
	// for the weight of fuel
	public double getRequiredFuel(Airport toAirport) {
		double value = getFuelConsumption(currentAirport.getDistance(toAirport)) - fuel;
		if (value > 0) {
			value += (fuelCapacity - value - fuel - 700) > 0 ? 700 : (fuelCapacity - value - fuel);
			return value;
		} else {
			return 0;
		}
	}

	// Returns sum of departure fee, landing fee and operation cost
	protected abstract double getFlightCost(Airport toAirport);

	// Does necessary operations for fuel consumption
	protected abstract double getFuelConsumption(double distance);

	protected double getBathtubCoefficient(double distanceRatio) {
		return 25.9324 * Math.pow(distanceRatio, 4) - 50.5633 * Math.pow(distanceRatio, 3)
				+ 35.0554 * Math.pow(distanceRatio, 2) - 9.90346 * distanceRatio + 1.97413;
	}

	// Calculates price of the fuel amount
	public double refuelExpense(double fuelAmount) {
		if (fuelAmount > 0)
			return fuelAmount * getCurrentAirport().getFuelCost();
		return 0;
	}

	@Override
	public boolean addFuel(double fuel) {
		if (this.fuel + fuel <= fuelCapacity && weight + fuel * fuelWeight <= maxWeight) {
			weight += fuel * fuelWeight;
			this.fuel += fuel;
			return true;
		}
		return false;
	}

	@Override
	public double fillUp() {
		double emptyTank = fuelCapacity - fuel;
		weight += emptyTank * fuelWeight;
		if (weight > maxWeight) {
			weight -= emptyTank * fuelWeight;
			emptyTank = (int) (maxWeight - weight) / fuelWeight;
			fuel += emptyTank;
			weight += emptyTank * fuelWeight;
			return emptyTank;
		}
		fuel = fuelCapacity;
		return emptyTank;
	}

	@Override
	public boolean hasFuel(double fuel) {
		return this.fuel >= fuel;
	}

	@Override
	public double getWeightRatio() {
		return weight / maxWeight;
	}

	public Airport getCurrentAirport() {
		return currentAirport;
	}
}
