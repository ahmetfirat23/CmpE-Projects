package project.airline.aircraft.concrete;

import project.airline.aircraft.PassengerAircraft;
import project.airport.Airport;

public class PropPassengerAircraft extends PassengerAircraft {
	public PropPassengerAircraft(double operationFee, Airport currentAirport, int ID) {
		super(operationFee, currentAirport, ID);
		weight = 14000;
		maxWeight = 23000;
		floorArea = 60;
		fuelCapacity = 6000;
		aircraftTypeMultiplier = 0.9;
	}

	private final double fuelConsumption = 0.6;
	private final double flightOpConst = 0.1;
	private final int distantRatioConst = 2000;
	private final double takeOffConst = 0.08;
	private final double efficientDistance = 538.2;
	private final double secondEffDistance = 1261.6;
	private final double maxRange = 2039.7;
	/*
	 * Check the following links for efficient distance and max range calculations
	 * https://www.desmos.com/calculator/7mm3dhmoqf?lang=tr
	 * https://www.desmos.com/calculator/9fn0znpeb5?lang=tr
	 */

	public double[] getEfficientInterval() {
		return new double[] { efficientDistance, secondEffDistance };
	}

	public double getMaxRange() {
		return maxRange;
	}

	protected double getFlightCost(Airport toAirport) {
		resetPassengersTransfer();
		double departureFee = currentAirport.departAircraft(this);
		double landingFee = toAirport.landAircraft(this);
		double operationCost = currentAirport.getDistance(toAirport) * getFullness() * flightOpConst;
		return landingFee + departureFee + operationCost;
	}

	@Override
	protected double getFuelConsumption(double distance) {
		double takeOffFC = weight * takeOffConst / fuelWeight;
		double distanceRatio = distance / distantRatioConst;
		double cruiseFC = fuelConsumption * getBathtubCoefficient(distanceRatio) * distance;
		return takeOffFC + cruiseFC;
	}
}
