package project.airline.aircraft.concrete;

import project.airline.aircraft.PassengerAircraft;
import project.airport.Airport;

public class JetPassengerAircraft extends PassengerAircraft {
	public JetPassengerAircraft(double operationFee, Airport currentAirport, int ID) {
		super(operationFee, currentAirport, ID);
		weight = 10000;
		maxWeight = 18000;
		floorArea = 30;
		fuelCapacity = 10000;
		aircraftTypeMultiplier = 5;
	}

	private final double fuelConsumption = 0.7;
	private final double flightOpConst = 0.08;
	private final int distantRatioConst = 5000;
	private final double takeOffConst = 0.1;
	private final double efficientDistance = 1345.5;
	private final double secondEffDistance = 3154;
	private final double maxRange = 4860.7;

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
