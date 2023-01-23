package project.airport;

import project.airline.aircraft.Aircraft;

public class HubAirport extends Airport {

	private final double departConst = 0.7;
	private final double landConst = 0.8;

	public HubAirport(int ID, double x, double y, double fuelCost, double operationFee, int aircraftCapacity) {
		super(ID, x, y, fuelCost, operationFee, aircraftCapacity);
	}

	@Override
	public double departAircraft(Aircraft aircraft) {
		double departureFee = operationFee * getAircraftWeightRatio(aircraft) * getFullnessCoefficient() * departConst;
		aircrafts.remove(aircraft);
		return departureFee;
	}

	@Override
	public double landAircraft(Aircraft aircraft) {
		double landingFee = operationFee * getAircraftWeightRatio(aircraft) * getFullnessCoefficient() * landConst;
		aircrafts.add(aircraft);
		return landingFee;
	}

}
