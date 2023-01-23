package project.passenger;

import java.util.ArrayList;


import project.airport.Airport;
import project.airport.HubAirport;
import project.airport.MajorAirport;
import project.airport.RegionalAirport;

public abstract class Passenger {

	private final long ID;
	private final double weight; 
	private final int baggageCount; 

	private ArrayList<Airport> destinations = new ArrayList<Airport>();
	protected Airport currentAirport;

	protected double connectionMultiplier = 1;
	protected double seatMultiplier = 1;
	
	private double seatConstant = 1;
	private int seatType = 0;
	
	private boolean transferred;

	public Passenger(long ID, double weight, int baggageCount, ArrayList<Airport> destinationsAirport,
			Airport currentAirport) {
		this.ID = ID;
		this.weight = weight;
		this.baggageCount = baggageCount;
		this.destinations = destinationsAirport;
		this.currentAirport = currentAirport;
	}

	public long getID() {
		return ID;
	}

	public double getWeight() {
		return weight;
	}

	public int getBaggageCount() {
		return baggageCount;
	}

	public ArrayList<Airport> getDestinationsAirports() {
		return destinations;
	}

	public Airport getCurrentAirport() {
		return currentAirport;
	}

	public int getSeatType() {
		return seatType;
	}

	public void setSeatType(int seatType) {
		this.seatType = seatType;
	}

	public boolean hasTransferred() {
		return transferred;
	}

	public void setTransferred(boolean transferred) {
		this.transferred = transferred;
	}

	public double getSeatConstant() {
		return seatConstant;
	}
	
	//Sets seat constants while loading passenger to aircraft
	public boolean board(int seatType) {
		switch (seatType) {
		case 0: // Economy
			seatConstant = 1.0;
			seatMultiplier = 0.6;
			break;
		case 1: // Business
			seatConstant = 2.8;
			seatMultiplier = 1.2;
			break;
		case 2: // First-class
			seatConstant = 7.5;
			seatMultiplier = 3.2;
			break;
		}
		return true;
	}

	//Returns ticket price and resets flight related constants
	public double disembark(Airport airport, double aircraftTypeMultiplier) {
		double ticketPrice = calculateTicketPrice(airport, aircraftTypeMultiplier);
		currentAirport = airport;
		seatMultiplier = 1;
		seatConstant = 1;
		connectionMultiplier = 1;
		return ticketPrice;
	}

	public boolean canDisembark(Airport airport) {
		if(destinations.contains(airport)) {
			return destinations.indexOf(currentAirport) < destinations.indexOf(airport);
		};
		return false;
	}

	//Adjusts seat constants on connection
	public boolean connection(int seatType) {
		switch (seatType) {
		case 0: // Economy
			seatConstant = 1.0;
			seatMultiplier *= 0.6;
			break;
		case 1: // Business
			seatConstant = 2.8;
			seatMultiplier *= 1.2;
			break;
		case 2: // First-class
			seatConstant = 7.5;
			seatMultiplier *= 3.2;
			break;
		}
		connectionMultiplier *= 0.8;
		return true;
	}

	protected double calculateAirportMultiplier(Airport toAirport) {
		if(currentAirport instanceof HubAirport) {
			if(toAirport instanceof HubAirport) 
				return 0.5;
			else if(toAirport instanceof MajorAirport)
				return 0.7;
			else if(toAirport instanceof RegionalAirport)
				return 1;
		}
		else if(currentAirport instanceof MajorAirport) {
			if(toAirport instanceof HubAirport) 
				return 0.6;
			else if(toAirport instanceof MajorAirport)
				return 0.8;
			else if(toAirport instanceof RegionalAirport)
				return 1.8;
		}
		else if(currentAirport instanceof RegionalAirport) {
			if(toAirport instanceof HubAirport) 
				return 0.9;
			else if(toAirport instanceof MajorAirport)
				return 1.6;
			else if(toAirport instanceof RegionalAirport)
				return 3.0;
		}
		return 1;
	}

	protected abstract double calculateTicketPrice(Airport airport, double aircraftTypeMultiplier);

	public abstract int getPassengerType();
}
