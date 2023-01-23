package project.airline.aircraft;

import java.util.ArrayList;

import project.airport.Airport;
import project.interfaces.PassengerInterface;
import project.passenger.Passenger;

public abstract class PassengerAircraft extends Aircraft implements PassengerInterface {

	public PassengerAircraft(double operationFee, Airport currentAirport, int ID) {
		super(operationFee, currentAirport, ID);
	}

	protected double floorArea;
	private int economySeats, businessSeats, firstClassSeats;
	private int occupiedEconomySeats, occupiedBusinessSeats, occupiedFirstClassSeats;

	protected ArrayList<Passenger> passengers = new ArrayList<Passenger>();

	final int economySeatArea = 1;
	final int businessSeatArea = 3;
	final int firstclassSeatArea = 8;

	// Loads passenger to appropriate seat
	public double loadPassenger(Passenger passenger) {
		passengers.add(passenger);
		int selectedSeat = 0;
		double constant = 1.2;
		if (passenger.getPassengerType() == 3 || passenger.getPassengerType() == 2) {
			if (isFull(2)) {
				if (!isFull(1)) {
					selectedSeat = 1;
					constant = 1.5;
				}
			} else {
				selectedSeat = 2;
				constant = 2.5;
			}
		} else if (passenger.getPassengerType() == 1) {
			if (!isFull(1)) {
				selectedSeat = 1;
				constant = 1.5;
			}
		}
		passenger.setSeatType(selectedSeat);
		occupySeat(selectedSeat);
		passenger.board(selectedSeat);
		weight += passenger.getWeight();
		return operationFee * aircraftTypeMultiplier * constant;
	}

	public boolean checkLoad(Passenger passenger) {
		return passenger.getCurrentAirport() == currentAirport && passenger.getWeight() + weight <= maxWeight
				&& !isFull();
	}

	// Calculates disembarkation revenue
	public double unloadPassenger(Passenger passenger) {
		double seatConst = passenger.getSeatConstant();
		double disembarkationRevenue = passenger.disembark(currentAirport, aircraftTypeMultiplier);
		passengers.remove(passenger);
		unoccupySeat(passenger.getSeatType());
		weight -= passenger.getWeight();
		return disembarkationRevenue * seatConst;
	}

	public boolean checkUnload(Passenger passenger) {
		return passengers.contains(passenger) && passenger.canDisembark(currentAirport);
	}

	// Transfers passenger to appropriate seat in another aircraft
	public double transferPassenger(Passenger passenger, PassengerAircraft toAircraft) {
		passengers.remove(passenger);
		toAircraft.addPassenger(passenger);
		int selectedSeat = 0;
		double constant = 1.2;
		if (passenger.getPassengerType() == 3 || passenger.getPassengerType() == 2) {
			if (isFull(2)) {
				if (!isFull(1)) {
					selectedSeat = 1;
					constant = 1.5;
				}
			} else {
				selectedSeat = 2;
				constant = 2.5;
			}
		} else if (passenger.getPassengerType() == 1) {
			if (!isFull(1)) {
				selectedSeat = 1;
				constant = 1.5;
			}
		}
		passenger.connection(selectedSeat);
		passenger.setTransferred(true);
		return operationFee * aircraftTypeMultiplier * constant;
	}

	public boolean checkTransfer(Passenger passenger, PassengerAircraft toAircraft) {
		return toAircraft.checkLoad(passenger) && !passenger.hasTransferred();
	}

	@Override
	public boolean isFull() {
		return occupiedEconomySeats >= economySeats && occupiedBusinessSeats >= businessSeats
				&& occupiedFirstClassSeats >= firstClassSeats;
	}

	@Override
	public boolean isFull(int seatType) {
		switch (seatType) {
		case 0:
			return occupiedEconomySeats >= economySeats;
		case 1:
			return occupiedBusinessSeats >= businessSeats;
		case 2:
			return occupiedFirstClassSeats >= firstClassSeats;
		}
		return false;
	}

	@Override
	public boolean isEmpty() {
		return occupiedEconomySeats == 0 && occupiedBusinessSeats == 0 && occupiedFirstClassSeats == 0;
	}

	@Override
	public double getAvailableWeight() {
		return maxWeight - weight;
	}

	public void occupySeat(int seatType) {
		switch (seatType) {
		case 0:
			occupiedEconomySeats++;
			break;
		case 1:
			occupiedBusinessSeats++;
			break;
		default:
			occupiedFirstClassSeats++;
			break;
		}
	}

	public void unoccupySeat(int seatType) {
		switch (seatType) {
		case 0:
			occupiedEconomySeats--;
			break;
		case 1:
			occupiedBusinessSeats--;
			break;
		default:
			occupiedFirstClassSeats--;
			break;
		}
	}

	@Override
	public void resetSeats() {
		economySeats = 0;
		businessSeats = 0;
		firstClassSeats = 0;
		System.out.println("2 " + ID + " " + "0 " + "0 " + "0");
	}

	@Override
	public boolean setSeats(int economy, int business, int firstClass) {
		if (economySeatArea * economy + businessSeatArea * business
				+ firstclassSeatArea * firstClass <= getRemainingArea()) {
			economySeats += economy;
			businessSeats += business;
			firstClassSeats += firstClass;
			System.out.println("2 " + ID + " " + economySeats + " " + businessSeats + " " + firstClassSeats);
			return true;
		}
		return false;
	}

	@Override
	public boolean setAllEconomy() {
		economySeats = (int) floorArea / economySeatArea;
		System.out.println("2 " + ID + " " + economySeats + " " + "0 " + "0");
		return true;
	}

	@Override
	public boolean setAllBusiness() {
		businessSeats = (int) floorArea / businessSeatArea;
		System.out.println("2 " + ID + " " + "0 " + businessSeats + "0");
		return true;
	}

	@Override
	public boolean setAllFirstClass() {
		firstClassSeats = (int) floorArea / firstclassSeatArea;
		System.out.println("2 " + ID + " " + "0 " + "0 " + firstClassSeats);
		return true;
	}

	@Override
	public boolean setRemainingEconomy() {
		double tmpArea = getRemainingArea();
		economySeats += (int) tmpArea / economySeatArea;
		System.out.println("2 " + ID + " " + economySeats + " " + businessSeats + " " + firstClassSeats);
		return true;
	}

	@Override
	public boolean setRemainingBusiness() {
		double tmpArea = getRemainingArea();
		businessSeats += (int) tmpArea / businessSeatArea;
		System.out.println("2 " + ID + " " + economySeats + " " + businessSeats + " " + firstClassSeats);
		return true;
	}

	@Override
	public boolean setRemainingFirstClass() {
		double tmpArea = getRemainingArea();
		firstClassSeats += (int) tmpArea / firstclassSeatArea;
		System.out.println("2 " + ID + " " + economySeats + " " + businessSeats + " " + firstClassSeats);
		return true;
	}

	private double getRemainingArea() {
		return floorArea - economySeatArea * economySeats - businessSeatArea * businessSeats
				- firstclassSeatArea * firstClassSeats;
	}

	@Override
	public double getFullness() {
		return (occupiedEconomySeats + occupiedBusinessSeats + occupiedFirstClassSeats) * 1.0
				/ (economySeats + businessSeats + firstClassSeats);
	}

	public void addPassenger(Passenger passenger) {
		passengers.add(passenger);
	}

	public void removePassenger(Passenger passenger) {
		passengers.remove(passenger);
	}

	protected void resetPassengersTransfer() {
		for (Passenger passenger : passengers) {
			passenger.setTransferred(false);
		}
	}
}
