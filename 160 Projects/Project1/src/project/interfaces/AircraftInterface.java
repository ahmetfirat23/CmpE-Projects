package project.interfaces;

import project.airport.Airport;

public interface AircraftInterface {

	double fly(Airport toAirport);

	// Refuels the aircraft by given amount of fuel
	boolean addFuel(double fuel);

	// Refuels the aircraft to its full capacity
	double fillUp();

	// Checks if the aircraft has the specified amount of fuel
	boolean hasFuel(double fuel);

	// Returns the ratio of weight to maximum weight
	double getWeightRatio();

}
