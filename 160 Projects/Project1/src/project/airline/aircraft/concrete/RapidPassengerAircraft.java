package project.airline.aircraft.concrete;

import project.airline.aircraft.PassengerAircraft;
import project.airport.Airport;

public class RapidPassengerAircraft extends PassengerAircraft {
	public RapidPassengerAircraft(double operationFee, Airport currentAirport, int ID) {
		super(operationFee, currentAirport, ID);
		weight = 80000;
		maxWeight = 185000;
		floorArea = 120;
		fuelCapacity = 120000;
		aircraftTypeMultiplier = 1.9;
	}

	private final double fuelConsumption = 5.3;
	private final double flightOpConst = 0.2;
	private final int distantRatioConst = 7000;
	private final double takeOffConst = 0.1;
	private final double efficientDistance = 1883.7;
	private final double secondEffDistance = 4415.6;
	private final double maxRange = 7012;

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
		double cruiseFC = fuelConsumption * getBathtubCoefficient(distanceRatio);
		return takeOffFC + cruiseFC;
	}
}
