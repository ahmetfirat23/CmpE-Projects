package project.Sector;

import java.util.ArrayList;
import java.util.HashMap;

import project.Affiliation;
import project.Warship.Warship;
import project.Warship.RepublicWarship.RepublicCruiser;
import project.Warship.SeparatistWarship.SeparatistDestroyer;

public class Sector {
	private int id;
	private String name;
	private Affiliation affiliation;
	private ArrayList<Warship> ships = new ArrayList<>();

	public Sector(int id, String name, Affiliation affiliation) {
		this.id = id;
		this.name = name;
		this.affiliation = affiliation;
	}

	public Affiliation getAffiliation() {
		return affiliation;
	}

	public void setAffiliation(Affiliation affiliation) {
		this.affiliation = affiliation;
	}

	public int getId() {
		return id;
	}

	/**
	 * Checks whether given coordinate is occupied
	 * 
	 * @param coordinate
	 * @return status of occupation
	 */
	public boolean isOccupied(int coordinate) {
		for (Warship ship : ships) {
			if (ship.getCoordinate() == coordinate)
				return true;
		}
		return false;
	}

	public String getName() {
		return name;
	}

	public void addToSector(Warship ship) {
		ships.add(ship);
	}

	public void removeFromSector(Warship ship) {
		ships.remove(ship);
	}

	/**
	 * Conducts republic assault on separatist ships. This function starts with
	 * sorting ships in the sector by ascending coordinates. Then it starts to scan
	 * through ship list to find an republic ship. When republic ship is found
	 * another loop is started from the index of the ship found. This loop searches
	 * for the first separatist ship with strictly lower power output. When ship is
	 * found republic ship is added to separatist ship's attackers list. This list
	 * separatist ship and attackers list is hold in a hashmap. When all the list is
	 * scanned in outer loop, keys in the hashmap are looped over. In this loop the
	 * last ship in the attackers list is ordered to attack the separatist ship. The
	 * last ship is always the closest ship to the separatist ship because ships
	 * were ordered in the beginning.
	 * <p>
	 * There are three snippets have complexities other than O(1). Our analysis
	 * should focus on them.
	 * <p>
	 * orderShips function uses Timsort algorithm which has O(nlogn) complexity in
	 * worst case and O(n) complexity in best case. (Source and further info:
	 * https://en.wikipedia.org/wiki/Timsort)
	 * <p>
	 * Snippet with outer and inner loop has O(n^2) complexity in worst case with
	 * some optimization. Both loops run until the end of the list from an entrance
	 * point. But sometimes inner loop is not reached or broken early. This is the
	 * optimization. In best case complexity is O(n) which happens when there is no
	 * republic ship, so inner loop is never reached.
	 * <p>
	 * Finally the last snippet with one enhanced for loop has O(n) complexity. Best
	 * and worst case's are same because given key set is always looped until the
	 * end.
	 * <p>
	 * Our analysis shows that in the worst case we have O(nlogn)+O(n^2)+O(n)
	 * complexity which equals to O(n^2) complexity. In best case which both ships
	 * are ordered correctly and there are no republic ships we have O(n)+O(n)+O(n)
	 * complexity which equals to O(n) complexity.
	 */
	public void assault() {
		orderShips(ships);
		HashMap<Warship, ArrayList<Warship>> attackMap = new HashMap<>();
		for (int i = 0; i < ships.size(); i++) {
			Warship ship = ships.get(i);
			if (ship instanceof SeparatistDestroyer)
				continue;
			int power = ship.getPowerOutput();
			for (int j = i; j < ships.size(); j++) {
				Warship targetShip = ships.get(j);
				if (targetShip instanceof SeparatistDestroyer && targetShip.getPowerOutput() < power) {
					attackMap.putIfAbsent(targetShip, new ArrayList<Warship>());
					attackMap.get(targetShip).add(ship);
					break;
				}
			}
		}
		for (Warship ship : attackMap.keySet()) {
			ArrayList<Warship> attackers = attackMap.get(ship);
			orderShips(attackers);
			((RepublicCruiser) attackers.get(attackers.size() - 1)).attack((SeparatistDestroyer) ship);
		}
	}

	/**
	 * Orders ship with ascend coordinate
	 * 
	 * @param ships ships in the sector
	 */
	private void orderShips(ArrayList<Warship> ships) {
		ships.sort((ship1, ship2) -> {
			return ship1.getCoordinate() - ship2.getCoordinate();
		});
	}
}
