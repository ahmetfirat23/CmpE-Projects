package project.Warship;

import java.util.ArrayList;

import project.Affiliation;
import project.IWarship;
import project.Crewman.Crewman;
import project.Crewman.General;
import project.Crewman.Concrete.Jedi;
import project.Crewman.Concrete.Officer;
import project.Crewman.Concrete.Sith;
import project.Sector.Sector;
import project.Warship.RepublicWarship.RepublicCruiser;
import project.Warship.SeparatistWarship.SeparatistDestroyer;

public abstract class Warship implements IWarship, Comparable<Warship> {
	protected int id;
	private String name;
	protected Sector currentSector;
	protected int coordinate;
	protected Affiliation affiliation;
	protected int armamentPower;
	protected int shieldPower;
	protected int crewCapacity;
	protected ArrayList<Crewman> crew;
	private boolean isDestroyed;
	private String destroyedBy;

	public Warship(int id, String name, Sector currentSector, int coordinate, ArrayList<Crewman> crew) {
		this.id = id;
		this.name = name;
		this.currentSector = currentSector;
		this.coordinate = coordinate;
		this.crew = crew;
		isDestroyed = false;
		if (currentSector != null) {
			currentSector.addToSector(this);
			for (Crewman crewman : crew) {
				crewman.setFree(false);
				crewman.setCurrentShip(this);
			}
		}
	}

	public int getId() {
		return id;
	}

	public String getName() {
		return name;
	}

	public Sector getCurrentSector() {
		return currentSector;
	}

	public int getCoordinate() {
		return coordinate;
	}

	public ArrayList<Crewman> getCrew() {
		return crew;
	}

	public boolean isDestroyed() {
		return isDestroyed;
	}

	public void setDestroyed(String name) {
		if (getName() == "Destroyer-56") {
			System.out.print("hello");
		}
		currentSector.removeFromSector(this);
		isDestroyed = true;
		destroyedBy = name;
	}

	protected boolean isFull() {
		return crewCapacity <= crew.size();
	}

	@Override
	public void removeCrewman(Crewman crewman) {
		if (crew.contains(crewman)) {
			crew.remove(crewman);
			crewman.setCurrentShip(null);
			crewman.setFree(true);
		}
	}

	@Override
	public void jumpToSector(Sector sector, int coordinate) {
		this.coordinate = coordinate;
		if (currentSector.equals(sector)) {
			return;
		}
		currentSector.removeFromSector(this);
		currentSector = sector;
		currentSector.addToSector(this);

	}

	@Override
	public int getPowerOutput() {
		int generalCont = calcGeneralCont();
		int officerCont = calcOfficerCont();
		int sectorBuff = affiliation == currentSector.getAffiliation() ? 3 : 2;
		return sectorBuff * (armamentPower + shieldPower + generalCont + officerCont);
	}

	private int calcGeneralCont() {
		int generalCont = 0;
		for (Crewman crewman : crew) {
			if (crewman instanceof General) {
				generalCont += ((General) crewman).getCombatPower();
			}
		}
		return generalCont;
	}

	private int calcOfficerCont() {
		int pilotMax = 0;
		int gunneryMax = 0;
		int engMax = 0;
		int tacticMax = 0;
		int commandMax = 0;
		for (Crewman crewman : crew) {
			if (crewman instanceof Officer) {
				Officer officer = (Officer) crewman;
				switch (officer.getIntrinsic()) {
				case PILOTING:
					pilotMax = officer.getIntrinsicLevel() > pilotMax ? officer.getIntrinsicLevel() : pilotMax;
					continue;
				case GUNNERY:
					gunneryMax = officer.getIntrinsicLevel() > gunneryMax ? officer.getIntrinsicLevel() : gunneryMax;
					continue;
				case ENGINEERING:
					engMax = officer.getIntrinsicLevel() > engMax ? officer.getIntrinsicLevel() : engMax;
					continue;
				case TACTICAL:
					tacticMax = officer.getIntrinsicLevel() > tacticMax ? officer.getIntrinsicLevel() : tacticMax;
					continue;
				case COMMAND:
					commandMax = officer.getIntrinsicLevel() > commandMax ? officer.getIntrinsicLevel() : commandMax;
					continue;
				}
			}
		}
		return (pilotMax + 1) * (gunneryMax + 1) * (engMax + 1) * (tacticMax + 1) * (commandMax + 1);
	}

	@Override
	public void upgradeArmament(int amount) {
		armamentPower += amount;
	}

	@Override
	public void upgradeShield(int amount) {
		shieldPower += amount;
	}

	/**
	 * Returns negative if ship has higher power output. In equality or when both
	 * ships are destroyed returns negative when ship has lower ID. If this ship is
	 * destroyed returns positive. If compared ship is destroyed returns negative.
	 */
	@Override
	public int compareTo(Warship other) {
		if (isDestroyed() && other.isDestroyed()) {
			return getId() - other.getId();
		} else if (isDestroyed()) {
			return 1;
		} else if (other.isDestroyed()) {
			return -1;
		} else if (other.getPowerOutput() - getPowerOutput() == 0) {
			return getId() - other.getId();
		}
		return other.getPowerOutput() - getPowerOutput();
	}

	public void logOutput() {
		if (isDestroyed) {
			System.out.println("Warship " + name + " is destroyed by " + destroyedBy + " in (" + currentSector.getName()
					+ "," + coordinate + ")");
			return;
		}
		System.out.println("Warship " + name + " in (" + currentSector.getName() + ", " + coordinate + ")");
		System.out.println(getCommander().getName() + " " + getPowerOutput());
	}

	/**
	 * Commits war crimes, I guess... Jedi turns to the dark side and kills everyone
	 * in the ship. Jedi kills them all, not just the officers but the Jedis and
	 * captives too. Jedi collects their experience and intrinsic level. Later Jedi
	 * is killed by the Sith and their experience is collected by the Sith. Finally
	 * republic ship is destroyed.
	 * 
	 * @param jedi      Jedi that turns to the dark side
	 * @param sith      Sith that convinces Jedi
	 * @param ship      Republic ship Jedi is in
	 * @param destroyer Separatist ship Sith is in
	 */
	protected void yesMaster(Jedi jedi, Sith sith, RepublicCruiser ship, SeparatistDestroyer destroyer) {
		for (Crewman crewman : ship.getCrew()) {
			if (!crewman.equals(jedi)) {
				crewman.setDead(jedi.getName());
				if (crewman instanceof Officer) {
					jedi.setExperience(((Officer) crewman).getIntrinsicLevel());
				} else {
					jedi.setExperience(((Jedi) crewman).getExperience());
				}
			}
		}
		for (Crewman crewman : ship.getCaptives()) {
			crewman.setDead(jedi.getName());
			jedi.setExperience(((Officer) crewman).getIntrinsicLevel());
		}
		ship.setDestroyed(destroyer.getName());
		jedi.setDead(sith.getName());
		sith.setExperience(jedi.getExperience());
		ship.getCrew().clear();
		ship.getCaptives().clear();
	}
}
