package project.airline.aircraft.concrete;

import project.airline.aircraft.PassengerAircraft;
import project.airport.Airport;

public class WidebodyPassengerAircraft extends PassengerAircraft {
	public WidebodyPassengerAircraft(double operationFee, Airport currentAirport, int ID) {
		super(operationFee, currentAirport, ID);
		weight = 135000;
		maxWeight = 250000;
		floorArea = 450;
		fuelCapacity = 140000;
		aircraftTypeMultiplier = 0.7;
	}

	private final double fuelConsumption = 3.0;
	private final double flightOpConst = 0.15;
	private final int distantRatioConst = 14000;
	private final double takeOffConst = 0.1;
	private final double efficientDistance = 3767.4;
	private final double secondEffDistance = 8831.2;
	private final double maxRange = 13988;

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
