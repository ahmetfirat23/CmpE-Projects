package project.interfaces;

import project.airline.aircraft.PassengerAircraft;
import project.passenger.Passenger;

public interface PassengerInterface {

	double transferPassenger(Passenger passenger, PassengerAircraft toAircraft);

	double loadPassenger(Passenger passenger);

	double unloadPassenger(Passenger passenger);

	// Checks whether the aircraft is full
	boolean isFull();

	// Checks whether a certain seat type is full
	boolean isFull(int seatType);

	// Checks whether the aircraft is empty
	boolean isEmpty();

	// Returns the leftover weight capacity of the aircraft
	public double getAvailableWeight();

	// Resets all seats
	public void resetSeats();

	// Sets seats by given numbers
	public boolean setSeats(int economy, int business, int firstClass);

	// Sets every seat to economy
	public boolean setAllEconomy();

	// Sets every seat to business
	public boolean setAllBusiness();

	// Sets every seat to first class
	public boolean setAllFirstClass();

	// Sets the remaining to economy
	public boolean setRemainingEconomy();

	// Sets the remaining to business
	public boolean setRemainingBusiness();

	// Sets the remaining to first class.
	public boolean setRemainingFirstClass();

	// Returns the ratio of occupied seats to all seats
	public double getFullness();

}
