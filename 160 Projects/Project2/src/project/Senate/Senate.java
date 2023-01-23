package project.Senate;

import java.util.ArrayList;
import java.util.Arrays;

import project.Affiliation;
import project.Intrinsic;
import project.Crewman.Crewman;
import project.Crewman.General;
import project.Crewman.Concrete.Jedi;
import project.Crewman.Concrete.Officer;
import project.Crewman.Concrete.Sith;
import project.Sector.Sector;
import project.Warship.Warship;
import project.Warship.RepublicWarship.RepublicCruiser;
import project.Warship.SeparatistWarship.SeparatistBattleship;
import project.Warship.SeparatistWarship.SeparatistDestroyer;
import project.Warship.SeparatistWarship.SeparatistFrigate;

/**
 * Senate is the conductor class of the project. This class connects other class
 * with Main class. All orders are given here from wrapper functions.This is just like
 * Darth Sidious has given all the orders as Chancellor Palpatine (a.k.a. The Senate).
 * 
 * @author Firat
 *
 */
public class Senate {

	ArrayList<Sector> sectors = new ArrayList<>();
	ArrayList<Crewman> crewmans = new ArrayList<>();
	ArrayList<Officer> officers = new ArrayList<>(); // Officers and generals are hold separately
	ArrayList<General> generals = new ArrayList<>(); // to log them in correct order at the end.
	ArrayList<Warship> ships = new ArrayList<>();

	/**
	 * Adds null decoys (dummies) to lists to keep given IDs and indexes of objects
	 * equal.
	 */
	public void adjustLists() {
		Sector decoy = new Sector(0, null, null);
		Officer decoy1 = new Officer(0, null, null, 0);
		RepublicCruiser decoy2 = new RepublicCruiser(0, null, null, 0, null);
		sectors.add(decoy);
		crewmans.add(decoy1);
		ships.add(decoy2);
	}

	/**
	 * Generates sector with given input and ID.
	 * 
	 * @param input Command line input entered
	 * @param id    Creation order
	 */
	public void createSector(String input, int id) {
		String[] inputs = input.split(" ");
		Sector sector = new Sector(id, inputs[0], convertStrtoAffl(inputs[1]));
		sectors.add(sector);
	}

	/**
	 * Generates officer with given input and ID.
	 * 
	 * @param input Command line input entered
	 * @param id    Creation order
	 */
	public void createOfficer(String input, int id) {
		String[] inputs = input.split(" ");
		Officer officer = new Officer(id, inputs[0], convertStrtoIntrinsic(inputs[1]), Integer.parseInt(inputs[2]));
		crewmans.add(officer);
		officers.add(officer);
	}

	/**
	 * Generates Jedi with given input and ID.
	 * 
	 * @param input Command line input entered
	 * @param id    Creation order
	 */
	public void createJedi(String input, int id) {
		String[] inputs = input.split(" ");
		Jedi jedi = new Jedi(id, inputs[0], Integer.parseInt(inputs[1]), Integer.parseInt(inputs[2]),
				Integer.parseInt(inputs[3]));
		crewmans.add(jedi);
		generals.add(jedi);
	}

	/**
	 * Generates Sith with given input and ID.
	 * 
	 * @param input Command line input entered
	 * @param id    Creation order
	 */
	public void createSith(String input, int id) {
		String[] inputs = input.split(" ");
		Sith sith = new Sith(id, inputs[0], Integer.parseInt(inputs[1]), Integer.parseInt(inputs[2]),
				Integer.parseInt(inputs[3]));
		crewmans.add(sith);
		generals.add(sith);
	}

	/**
	 * Generates warship with given input and ID and adds crew to the ship.
	 * 
	 * @param input     Command line input entered
	 * @param crewInput Second command line input entered
	 * @param id        Creation order
	 */
	public void createShip(String input, String crewInput, int id) {
		String[] inputs = input.split(" ");
		String[] crewInputs = crewInput.split(" ");
		crewInputs = Arrays.copyOfRange(crewInputs, 1, crewInputs.length);
		ArrayList<Crewman> shipCrew = new ArrayList<Crewman>();
		for (String str : crewInputs) {
			shipCrew.add(crewmans.get(Integer.parseInt(str)));
		}
		Warship ship = null;
		if (inputs[0].equals("RepublicCruiser")) {
			ship = new RepublicCruiser(id, inputs[1], sectors.get(Integer.parseInt(inputs[2])),
					Integer.parseInt(inputs[3]), shipCrew);
		} else if (inputs[0].equals("SeparatistBattleship")) {
			ship = new SeparatistBattleship(id, inputs[1], sectors.get(Integer.parseInt(inputs[2])),
					Integer.parseInt(inputs[3]), shipCrew);
		} else if (inputs[0].equals("SeparatistFrigate")) {
			ship = new SeparatistFrigate(id, inputs[1], sectors.get(Integer.parseInt(inputs[2])),
					Integer.parseInt(inputs[3]), shipCrew);
		} else {
			ship = new SeparatistDestroyer(id, inputs[1], sectors.get(Integer.parseInt(inputs[2])),
					Integer.parseInt(inputs[3]), shipCrew);
		}
		ships.add(ship);
	}

	/**
	 * Simulates the attack if both warships aren't destroyed.
	 * 
	 * @param input Command line input entered
	 */
	public void simulateAttack(String input) {
		String[] inputs = input.split(" ");
		int attackerID = Integer.parseInt(inputs[0]);
		int defenderID = Integer.parseInt(inputs[1]);
		Warship attacker = ships.get(attackerID);
		Warship defender = ships.get(defenderID);
		if (attacker.isDestroyed() || defender.isDestroyed()) {
			return;
		} else if (attacker instanceof RepublicCruiser) {
			((RepublicCruiser) attacker).attack((SeparatistDestroyer) defender);
		} else {
			((SeparatistDestroyer) attacker).attack((RepublicCruiser) defender);
		}
	}

	/**
	 * Simulates the assault in a sector.
	 * 
	 * @param input Command line input entered
	 */
	public void simulateAssault(String input) {
		Sector sector = sectors.get(Integer.parseInt(input));
		sector.assault();
	}

	/**
	 * Relocates warship to given position. Operation may not be successful if
	 * coordinate is occupied or ship is destroyed.
	 * 
	 * @param input Command line input entered
	 */
	public void relocate(String input) {
		String[] inputs = input.split(" ");
		Warship ship = ships.get(Integer.parseInt(inputs[0]));
		Sector sector = sectors.get(Integer.parseInt(inputs[1]));
		int coordinate = Integer.parseInt(inputs[2]);
		if (ship.isDestroyed() || sector.isOccupied(coordinate)) {
			return;
		}
		ship.jumpToSector(sector, coordinate);
	}

	/**
	 * Makes given non-destroyed republic cruiser visit Corousant.
	 * 
	 * @param input Command line input entered
	 */
	public void visitCorousant(String input) {
		RepublicCruiser ship = ((RepublicCruiser) ships.get(Integer.parseInt(input)));
		if (ship.isDestroyed()) {
			return;
		}
		ship.visitCoruscant();
	}

	/**
	 * Adds given crewman to given warship. Operation success depends on the ship
	 * type.
	 * 
	 * @param input Command line input entered
	 */
	public void addCrewman(String input) {
		String[] inputs = input.split(" ");
		Crewman crewman = crewmans.get(Integer.parseInt(inputs[0]));
		Warship ship = ships.get(Integer.parseInt(inputs[1]));
		ship.addCrewman(crewman);
	}

	/**
	 * Removes given crewman from given warship. Operation may not be successful if
	 * ship is destroyed or crewman is not available.
	 * 
	 * @param input Command line input entered
	 */
	public void removeCrewman(String input) {
		String[] inputs = input.split(" ");
		Crewman crewman = crewmans.get(Integer.parseInt(inputs[0]));
		Warship ship = ships.get(Integer.parseInt(inputs[1]));
		if (ship.isDestroyed() || !crewman.isAlive() || crewman.isImprisoned() || crewman.isInCorousant()) {
			return;
		}
		ship.removeCrewman(crewman);
	}

	/**
	 * Trains an alive officer by 1 point.
	 * 
	 * @param input Command line input entered
	 */
	public void trainOfficer(String input) {
		int id = Integer.parseInt(input);
		if (crewmans.get(id).isAlive()) {
			((Officer) crewmans.get(id)).train();
		}
	}

	/**
	 * Upgrades a non-destroyed ship's given part by given amount.
	 * 
	 * @param input Command line input entered
	 */
	public void upgradeShip(String input) {
		String[] inputs = input.split(" ");
		Warship ship = ships.get(Integer.parseInt(inputs[0]));
		if (ship.isDestroyed()) {
			return;
		}
		switch (inputs[1]) {
		case "Armament":
			ship.upgradeArmament(Integer.parseInt(inputs[2]));
			break;
		default:
			ship.upgradeShield(Integer.parseInt(inputs[2]));
			break;
		}
	}

	/**
	 * Logs warships' present condition.
	 */
	public void logWarships() {
		ships.remove(0);
		ships.sort((ship1, ship2) -> ship1.compareTo(ship2));
		for (Warship ship : ships) {
			ship.logOutput();
		}
	}

	/**
	 * Logs crewmans' present condition.
	 */
	public void logCrewmans() {
		generals.sort((general1, general2) -> general1.compareTo(general2));
		officers.sort((officer1, officer2) -> officer1.compareTo(officer2));
		for (General general : generals) {
			general.logOutput();
		}
		for (Officer officer : officers) {
			officer.logOutput();
		}
	}

	/**
	 * Converts given string to an affiliation. Exact match with affiliation name is
	 * required.
	 * 
	 * @param str affiliation name string
	 * @return affiliation with given name
	 */
	private Affiliation convertStrtoAffl(String str) {
		if (str.equals("REPUBLIC")) {
			return Affiliation.REPUBLIC;
		}
		return Affiliation.SEPARATISTS;
	}

	/**
	 * Converts given string to an intrinsic. Exact match with intrinsic name is
	 * required.
	 * 
	 * @param str intrinsic name string
	 * @return intrinsic with given name
	 */
	private Intrinsic convertStrtoIntrinsic(String str) {
		if (str.equals("PILOTING")) {
			return Intrinsic.PILOTING;
		} else if (str.equals("TACTICAL")) {
			return Intrinsic.TACTICAL;
		} else if (str.equals("GUNNERY")) {
			return Intrinsic.GUNNERY;
		} else if (str.equals("ENGINEERING")) {
			return Intrinsic.ENGINEERING;
		} else {
			return Intrinsic.COMMAND;
		}
	}
}
