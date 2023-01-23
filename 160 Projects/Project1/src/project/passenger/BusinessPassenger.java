package project.passenger;

import java.util.ArrayList;

import project.airport.Airport;

public class BusinessPassenger extends Passenger {

	private final double passengerMultiplier = 1.2;
	private double airportMultiplier;

	public BusinessPassenger(long ID, double weight, int baggageCount, ArrayList<Airport> destinationsAirport,
			Airport currentAirport) {
		super(ID, weight, baggageCount, destinationsAirport, currentAirport);
	}

	@Override
	protected double calculateTicketPrice(Airport toAirport, double aircraftTypeMultiplier) {
		double distance = currentAirport.getDistance(toAirport);
		airportMultiplier = calculateAirportMultiplier(toAirport);
		double ticketPrice = distance * aircraftTypeMultiplier * connectionMultiplier * airportMultiplier
				* passengerMultiplier * seatMultiplier;
		return ticketPrice + ticketPrice * (5f / 100) * getBaggageCount();
	}

	public int getPassengerType() {
		return 1;
	}
}
