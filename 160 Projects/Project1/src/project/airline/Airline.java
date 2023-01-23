package project.airline;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import project.airline.aircraft.Aircraft;
import project.airline.aircraft.PassengerAircraft;
import project.airline.aircraft.concrete.JetPassengerAircraft;
import project.airline.aircraft.concrete.PropPassengerAircraft;
import project.airline.aircraft.concrete.RapidPassengerAircraft;
import project.airline.aircraft.concrete.WidebodyPassengerAircraft;
import project.airport.Airport;
import project.airport.HubAirport;
import project.airport.MajorAirport;
import project.airport.RegionalAirport;
import project.passenger.BusinessPassenger;
import project.passenger.EconomyPassenger;
import project.passenger.FirstClassPassenger;
import project.passenger.LuxuryPassenger;
import project.passenger.Passenger;

/**
 * Airline is the conductor class of the whole project. Main class import only
 * this class so the airline company is managed in Main through this class.
 * Every airport, passenger and aircraft is hold here and their functions are
 * also called from here either by wrapper functions or directly. This airline
 * aims to maximize its profit by using some predetermined flight plans. These
 * flight plans are only executed when there are more passengers than estimated
 * efficient limits. Also it is important that flight distances must be in
 * between calculated efficient distances with an offset.
 * 
 * @author Ahmet Firat Gamsiz
 *
 */
public class Airline {
	/**
	 * Initializes the airline with given parameters
	 * 
	 * @param maxAircraftCount max aircraft number the airline can hold
	 * @param operationalCost  cost of the operations for the airline
	 */
	public Airline(int maxAircraftCount, double operationalCost) {
		super();
		this.maxAircraftCount = maxAircraftCount;
		this.operationalCost = operationalCost;
	}

	/**
	 * Holds the aircrafts in this airline
	 */
	private ArrayList<PassengerAircraft> aircrafts = new ArrayList<PassengerAircraft>();
	/**
	 * Holds the passengers and their IDs in this airline
	 */
	private HashMap<Long, Passenger> passengers = new HashMap<Long, Passenger>();
	/**
	 * Holds the airports and their IDs in this airline
	 */
	private HashMap<Integer, Airport> airports = new HashMap<Integer, Airport>();

	/**
	 * Maximum number of aircrafts this airline can hold.
	 */
	private int maxAircraftCount;
	/**
	 * Operational cost value for this airline.
	 */
	private double operationalCost;
	/**
	 * Total expenses of this airline
	 */
	private double expenses = 0;
	/**
	 * Total revenue of this airline
	 */
	private double revenue = 0;

	/**
	 * Flies an aircraft from its current airport to a destination airport and adds
	 * the flight cost to expenses. Operation's success depends on whether the
	 * destination airport is empty and the aircraft has enough fuel.
	 * <p>
	 * Operational cost is added to expenses regardless of the flight's success
	 * 
	 * @param toAirport     destination airport
	 * @param aircraftIndex index of the chosen aircraft
	 * @return success of the flight operation
	 */
	private boolean fly(Airport toAirport, int aircraftIndex) {
		expenses += operationalCost * aircrafts.size();
		PassengerAircraft aircraft = aircrafts.get(aircraftIndex);
		if (aircraft.checkFly(toAirport)) {
			double flightCost = aircraft.fly(toAirport);

			expenses += flightCost;
			System.out.println("1 " + toAirport.getID() + " " + aircraftIndex);
			return true;
		}
		return false;
	}

	/**
	 * Loads a passenger from an airport to an aircraft and adds the loading cost to
	 * expenses. Operation's success depends on whether the passenger and the
	 * aircraft are in the same airport, aircraft's weight limit is not exceeded and
	 * there is appropriate seat for the passenger.
	 * 
	 * @param passenger     chosen passenger to load
	 * @param airport       the airport that the passenger is in
	 * @param aircraftIndex index of the chosen aircraft
	 * @return success of the load operation
	 */
	private boolean loadPassenger(Passenger passenger, Airport airport, int aircraftIndex) {
		PassengerAircraft aircraft = aircrafts.get(aircraftIndex);
		if (aircraft.checkLoad(passenger)) {
			airport.removePassenger(passenger);
			double loadingCost = aircraft.loadPassenger(passenger);
			expenses += loadingCost;
			System.out.println("4 " + passenger.getID() + " " + aircraftIndex + " " + airport.getID());

			return true;
		}
		return false;
	}

	/**
	 * Unloads a passenger from an aircraft and adds the ticket price to revenue.
	 * Operation's success depends on whether the disembarkation airport is in the
	 * passenger's further destinations.
	 * 
	 * @param passenger     chosen passenger to unload
	 * @param aircraftIndex index of the chosen aircraft
	 * @return success of the unload operation
	 */
	private boolean unloadPassenger(Passenger passenger, int aircraftIndex) {
		PassengerAircraft aircraft = aircrafts.get(aircraftIndex);
		Airport disembarkAirport = aircraft.getCurrentAirport();
		if (aircraft.checkUnload(passenger)) {
			double ticketPrice = aircraft.unloadPassenger(passenger);
			revenue += ticketPrice;
			System.out.println("5 " + passenger.getID() + " " + aircraftIndex + " " + disembarkAirport.getID());

			disembarkAirport.addPassenger(passenger);
			return true;
		}
		return false;
	}

	/**
	 * Transfers a passenger from an aircraft to another aircraft and adds the
	 * transfer cost to expenses. Operation's success depends on whether the
	 * passenger can load to the aircraft and the passenger has transfered at the
	 * current airport previously.
	 * 
	 * @param passenger       chosen passenger to transfer
	 * @param aircraftIndex   index of the chosen aircraft
	 * @param toAircraftIndex index of the chosen aircraft to transfer in
	 * @return success of the transfer operation
	 */
	private boolean transferPassenger(Passenger passenger, int aircraftIndex, int toAircraftIndex) {
		PassengerAircraft aircraft = aircrafts.get(aircraftIndex);
		PassengerAircraft toAircraft = aircrafts.get(toAircraftIndex);
		if (aircraft.checkTransfer(passenger, toAircraft)) {
			double transferCost = aircraft.transferPassenger(passenger, toAircraft);
			System.out.println("6 " + passenger.getID() + " " + aircraftIndex + " " + toAircraftIndex);
			expenses += transferCost;
			return true;
		}
		return false;
	}

	/**
	 * Refuels an aircraft with given amount of fuel and adds the refuel expense to
	 * expenses.
	 * 
	 * @param aircraftIndex index of the chosen aircraft
	 * @param fuel          fuel amount to be loaded
	 */
	private void refuel(int aircraftIndex, double fuel) {
		Aircraft aircraft = aircrafts.get(aircraftIndex);
		if (aircraft.addFuel(fuel)) {
			double refuelExpense = aircraft.refuelExpense(fuel);
			expenses += refuelExpense;
			System.out.println("3 " + aircraftIndex + " " + fuel);

		}
	}

	/**
	 * Fills up an aircraft's fuel and adds the refuel expense to expenses.
	 * 
	 * @param aircraftIndex index of the chosen aircraft
	 */
	private void fillUp(int aircraftIndex) {
		Aircraft aircraft = aircrafts.get(aircraftIndex);
		double fuel = aircraft.fillUp();
		double refuelExpense = aircraft.refuelExpense(fuel);
		System.out.println("3 " + aircraftIndex + " " + fuel);
		expenses += refuelExpense;
	}

	/**
	 * Creates an airport of given type and variables
	 * 
	 * @param type             type of the airport
	 * @param ID               id of the airport
	 * @param x                x-coordinate of the airport position
	 * @param y                y-coordinate of the airport position
	 * @param fuelCost         unit fuel cost in the airport
	 * @param operationFee     fee for the operations in the aircraft
	 * @param aircraftCapacity max aircraft number the airport can hold
	 */
	public void createAirport(String type, int ID, double x, double y, double fuelCost, double operationFee,
			int aircraftCapacity) {
		Airport airport;
		if (type.equals("hub")) {
			airport = new HubAirport(ID, x, y, fuelCost, operationFee, aircraftCapacity);
		} else if (type.equals("major")) {
			airport = new MajorAirport(ID, x, y, fuelCost, operationFee, aircraftCapacity);
		} else {
			airport = new RegionalAirport(ID, x, y, fuelCost, operationFee, aircraftCapacity);
		}
		airports.put(ID, airport);
	}

	/**
	 * Creates a passenger of given type and variables
	 * 
	 * @param type            type of the passenger
	 * @param ID              id of the passenger
	 * @param weight          total weight of the passenger
	 * @param baggageCount    amount of baggage passenger has
	 * @param destinationsIDs IDs of the airports passenger wants to travel
	 */
	public void createPassenger(String type, long ID, double weight, int baggageCount,
			ArrayList<Integer> destinationsIDs) {
		ArrayList<Airport> destinationsAirport = new ArrayList<Airport>();
		for (int dest : destinationsIDs) {
			destinationsAirport.add(airports.get(dest));
		}
		Airport currentAirport = destinationsAirport.get(0);
		Passenger passenger;
		if (type.equals("business")) {
			passenger = new BusinessPassenger(ID, weight, baggageCount, destinationsAirport, currentAirport);
		} else if (type.equals("economy")) {
			passenger = new EconomyPassenger(ID, weight, baggageCount, destinationsAirport, currentAirport);
		} else if (type.equals("first")) {
			passenger = new FirstClassPassenger(ID, weight, baggageCount, destinationsAirport, currentAirport);
		} else {
			passenger = new LuxuryPassenger(ID, weight, baggageCount, destinationsAirport, currentAirport);
		}
		passengers.put(ID, passenger);
	}

	/**
	 * Creates a jet aircraft with given variables.
	 * 
	 * @param operationalCost cost of operations for this aircraft
	 * @param airportID       id of the airport aircraft created in
	 * @return success of operation
	 */
	public boolean createJetAircraft(double operationalCost, int airportID) {
		Airport airport = airports.get(airportID);
		if (aircrafts.size() < maxAircraftCount && !airport.isFull()) {
			PassengerAircraft aircraft = new JetPassengerAircraft(operationalCost, airport, aircrafts.size());
			aircrafts.add(aircraft);
			airport.addAircraft(aircraft);
			System.out.println("0 " + airportID + " " + 3);
			return true;
		}
		return false;
	}

	/**
	 * Creates a prop aircraft with given variables.
	 * 
	 * @param operationalCost cost of operations for this aircraft
	 * @param airportID       id of the airport aircraft created in
	 * @return success of operation
	 */
	public boolean createPropAircraft(double operationalCost, int airportID) {
		Airport airport = airports.get(airportID);
		if (aircrafts.size() < maxAircraftCount && !airport.isFull()) {
			PassengerAircraft aircraft = new PropPassengerAircraft(operationalCost, airport, aircrafts.size());
			aircrafts.add(aircraft);
			airport.addAircraft(aircraft);
			System.out.println("0 " + airportID + " " + 0);
			return true;
		}
		return false;
	}

	/**
	 * Creates a rapid aircraft with given variables.
	 * 
	 * @param operationalCost cost of operations for this aircraft
	 * @param airportID       id of the airport aircraft created in
	 * @return success of operation
	 */
	public boolean createRapidAircraft(double operationalCost, int airportID) {
		Airport airport = airports.get(airportID);
		if (aircrafts.size() < maxAircraftCount && !airport.isFull()) {
			PassengerAircraft aircraft = new RapidPassengerAircraft(operationalCost, airport, aircrafts.size());
			aircrafts.add(aircraft);
			airport.addAircraft(aircraft);
			System.out.println("0 " + airportID + " " + 2);
			return true;
		}
		return false;
	}

	/**
	 * Creates a widebody aircraft with given variables.
	 * 
	 * @param operationalCost cost of operations for this aircraft
	 * @param airportID       id of the airport aircraft created in
	 * @return success of operation
	 */
	public boolean createWidebodyAircraft(double operationalCost, int airportID) {
		Airport airport = airports.get(airportID);
		if (aircrafts.size() < maxAircraftCount && !airport.isFull()) {
			PassengerAircraft aircraft = new WidebodyPassengerAircraft(operationalCost, airport, aircrafts.size());
			aircrafts.add(aircraft);
			airport.addAircraft(aircraft);
			System.out.println("0 " + airportID + " " + 1);
			return true;
		}
		return false;
	}

	/**
	 * Sorts passengers according to their priority and adds seats with this order.
	 * Highest appropriate seats are created firstly for passengers with highest
	 * priority. This is a utility function for flight plans.
	 * 
	 * @param passengers list of passengers to be seated
	 * @param aircraft   chosen aircraft object
	 */
	private void setSeating(ArrayList<Passenger> passengers, PassengerAircraft aircraft) {
		Collections.sort(passengers, (Passenger p1, Passenger p2) -> {
			return -1 * (p1.getPassengerType() - p2.getPassengerType());
		});
		aircraft.resetSeats();
		for (Passenger passenger : passengers) {
			switch (passenger.getPassengerType()) {
			case 3:
			case 2:
				aircraft.setSeats(0, 0, 1);
				continue;
			case 1:
				aircraft.setSeats(0, 1, 0);
				continue;
			case 0:
				aircraft.setRemainingEconomy();
				break;
			}
			return;
		}
	}

	/**
	 * Gets an airport's ID from a passenger's destinations. This is a utility
	 * function for flight plans.
	 * 
	 * @param passenger the chosen passenger
	 * @param index     index of the airport desired
	 * @return id of the desired airport
	 */
	private String getIDFromDests(Passenger passenger, int index) {
		return Integer.toString(passenger.getDestinationsAirports().get(index).getID());
	}

	/**
	 * Returns the last aircraft from aircrafts list. This is a utility function for
	 * flight plans.
	 * 
	 * @return last aircraft in the aircrafts list
	 */
	private PassengerAircraft getLastAircraft() {
		return aircrafts.get(aircrafts.size() - 1);
	}

	/**
	 * Executes flights for luxury passengers with jet aircrafts. This method
	 * categorizes passengers with the same starting airport and same first or
	 * second destinations. Then, it scans through this categories and select the
	 * ones has at least three passengers and the distance is less than second
	 * efficient distance for the aircraft plus an offset. Efficient distances are
	 * calculated by using critical points of bathtub curve and given as constants.
	 * Passengers are seated according to aircraft's capacity and required fuel is
	 * loaded. Finally flight and unload operations are executed.
	 * 
	 * @param operationalCost operation cost for the jet aircraft
	 */
	public void jetFlightPlan(double operationalCost) {
		if (aircrafts.size() == maxAircraftCount) {
			return;
		}
		HashMap<String, ArrayList<Passenger>> destCount = new HashMap<String, ArrayList<Passenger>>();
		for (Passenger passenger : passengers.values()) {
			if (passenger.getPassengerType() == 3) {
				if (destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1)) != null) {
					destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1)).add(passenger);
				} else {
					destCount.put(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1),
							new ArrayList<>(Arrays.asList(passenger)));
				}
				if (passenger.getDestinationsAirports().size() > 2
						&& destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2)) != null) {
					destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2)).add(passenger);
				} else if (passenger.getDestinationsAirports().size() > 2) {
					destCount.put(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2),
							new ArrayList<>(Arrays.asList(passenger)));
				}
			}
		}

		for (String dest : destCount.keySet()) {
			if (aircrafts.size() == maxAircraftCount) {
				break;
			}
			ArrayList<Passenger> passengers = destCount.get(dest);
			Airport airport = airports.get(Integer.parseInt(dest.split(" ")[0]));
			Airport toAirport = airports.get(Integer.parseInt(dest.split(" ")[1]));
			if (passengers.size() >= 2 && airport.getDistance(toAirport) < 3154 + 700) {
				if (createJetAircraft(operationalCost, airport.getID())) {
					getLastAircraft().setAllFirstClass();
					for (Passenger passenger : passengers) {
						loadPassenger(passenger, airport, aircrafts.size() - 1);
					}
					refuel(aircrafts.size() - 1, getLastAircraft().getRequiredFuel(toAirport));
					fly(toAirport, aircrafts.size() - 1);
					for (Passenger passenger : passengers) {
						unloadPassenger(passenger, aircrafts.size() - 1);
					}
				}
			}
		}
	}

	/**
	 * Executes flights for all passengers except luxury with prop aircrafts. This
	 * method categorizes passengers with the same starting airport and same first
	 * or second destinations. Then, it scans through this categories and select the
	 * ones has at least thirty passengers and the distance is in between efficient
	 * distances for the aircraft plus an offset. Efficient distances are calculated
	 * by using critical points of bathtub curve and given as constants. Passengers
	 * are seated according to aircraft's capacity and required fuel is loaded.
	 * Finally flight and unload operations are executed.
	 * 
	 * @param operationalCost operation cost for the prop aircraft
	 */
	public void propFlightPlan(double operationalCost) {
		if (revenue > 0 || aircrafts.size() == maxAircraftCount) {
			return;
		}
		HashMap<String, ArrayList<Passenger>> destCount = new HashMap<String, ArrayList<Passenger>>();
		for (Passenger passenger : passengers.values()) {
			if (passenger.getPassengerType() != 3) {
				if (destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1)) != null) {
					destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1)).add(passenger);
				} else {
					destCount.put(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1),
							new ArrayList<>(Arrays.asList(passenger)));
				}
				if (passenger.getDestinationsAirports().size() > 2
						&& destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2)) != null) {
					destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2)).add(passenger);
				} else if (passenger.getDestinationsAirports().size() > 2) {
					destCount.put(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2),
							new ArrayList<>(Arrays.asList(passenger)));
				}
			}
		}
		for (String dest : destCount.keySet()) {
			int aircraftIndex = aircrafts.size();
			ArrayList<Passenger> passengers = destCount.get(dest);
			Airport airport = airports.get(Integer.parseInt(dest.split(" ")[0]));
			Airport toAirport = airports.get(Integer.parseInt(dest.split(" ")[1]));
			if (passengers.size() >= 30 && airport.getDistance(toAirport) < 1261 + 200
					&& airport.getDistance(toAirport) > 500) {
				if (createPropAircraft(operationalCost, airport.getID())) {
					setSeating(passengers, getLastAircraft());
					for (Passenger passenger : passengers) {
						loadPassenger(passenger, airport, aircraftIndex);
					}
					fillUp(aircraftIndex);
					fly(toAirport, aircraftIndex);
					for (Passenger passenger : passengers) {
						unloadPassenger(passenger, aircraftIndex);
					}
				}
			}
		}
	}

	/**
	 * Executes flights for all passengers with wide aircrafts. This method
	 * categorizes passengers with the same starting airport and same first or
	 * second destinations. Passengers are seated according to aircraft's capacity
	 * and fuel is filled up. Finally flight and unload operations are executed. If
	 * destination airport is out of range flights are executed through path finder.
	 * 
	 * @param operationalCost operation cost for the wide aircraft
	 */
	public void wideFlightPlan(double operationalCost) {
		if (revenue > 0 || aircrafts.size() == maxAircraftCount) {
			return;
		}
		HashMap<String, ArrayList<Passenger>> destCount = new HashMap<String, ArrayList<Passenger>>();
		for (Passenger passenger : passengers.values()) {
			if (destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1)) != null) {
				destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1)).add(passenger);
			} else {
				destCount.put(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 1),
						new ArrayList<>(Arrays.asList(passenger)));
			}
			if (passenger.getDestinationsAirports().size() > 2
					&& destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2)) != null) {
				destCount.get(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2)).add(passenger);
			} else if (passenger.getDestinationsAirports().size() > 2) {
				destCount.put(getIDFromDests(passenger, 0) + " " + getIDFromDests(passenger, 2),
						new ArrayList<>(Arrays.asList(passenger)));
			}
		}
		for (String dest : destCount.keySet()) {
			int aircraftIndex = aircrafts.size();
			ArrayList<Passenger> passengers = destCount.get(dest);
			Airport airport = airports.get(Integer.parseInt(dest.split(" ")[0]));
			Airport toAirport = airports.get(Integer.parseInt(dest.split(" ")[1]));
			if (passengers.size() >= 1 && airport.getDistance(toAirport) < 13000) {
				if (createWidebodyAircraft(operationalCost, airport.getID())) {
					ArrayList<Passenger> pass = new ArrayList<Passenger>();
					pass.add(passengers.get(0));
					setSeating(pass, getLastAircraft());
					loadPassenger(passengers.get(0), airport, aircraftIndex);
					fillUp(aircraftIndex);
					fly(toAirport, aircraftIndex);
					boolean hasRev = unloadPassenger(passengers.get(0), aircraftIndex);
					if (hasRev) {
						break;
					}
				}
			}
		}
		for (String dest : destCount.keySet()) {
			if (revenue > 0) {
				break;
			}
			int aircraftIndex = aircrafts.size();
			ArrayList<Passenger> passengers = destCount.get(dest);
			Airport airport = airports.get(Integer.parseInt(dest.split(" ")[0]));
			Airport toAirport = airports.get(Integer.parseInt(dest.split(" ")[1]));
			if (passengers.size() >= 1 && createWidebodyAircraft(operationalCost, airport.getID())) {
				ArrayList<Passenger> pass = new ArrayList<Passenger>();
				pass.add(passengers.get(0));
				setSeating(pass, getLastAircraft());
				loadPassenger(passengers.get(0), airport, aircraftIndex);
				pathFinder(airport, toAirport, aircraftIndex);
				unloadPassenger(passengers.get(0), aircraftIndex);
			}
		}
	}

	/**
	 * Executes multiple flights through a distant airport. Scans through airports
	 * in flight range and flies to the one closest to the target airport until
	 * target airport is in the range.
	 * 
	 * @param airport       starting airport
	 * @param toAirport     target airport
	 * @param aircraftIndex index of the chosen aircraft
	 */
	private void pathFinder(Airport airport, Airport toAirport, int aircraftIndex) {
		ArrayList<Airport> visitedAirports = new ArrayList<Airport>();
		visitedAirports.add(airport);
		fillUp(aircraftIndex);
		while (airport.getDistance(toAirport) > 14000 && !airport.equals(toAirport)) {
			double totalDistance = 99999999;
			Airport nextAirport = null;
			fillUp(aircraftIndex);
			for (Airport interAirport : airports.values()) {
				if (!visitedAirports.contains(interAirport) && airport.getDistance(interAirport) < 14000
						&& totalDistance > airport.getDistance(interAirport) + interAirport.getDistance(toAirport)) {
					totalDistance = airport.getDistance(interAirport) + interAirport.getDistance(toAirport);
					nextAirport = interAirport;
				}
			}

			fly(nextAirport, aircraftIndex);
			airport = nextAirport;
			visitedAirports.add(airport);

		}
		fillUp(aircraftIndex);
		fly(toAirport, aircraftIndex);
	}

	/**
	 * Prints profit. Profit is revenue minus expenses.
	 */
	public void printProfit() {
		System.out.println(revenue - expenses);
	}
}
