package project.Warship.RepublicWarship;

import java.util.ArrayList;

import project.Affiliation;
import project.Crewman.Crewman;
import project.Crewman.General;
import project.Crewman.Concrete.Jedi;
import project.Crewman.Concrete.Officer;
import project.Crewman.Concrete.Sith;
import project.Sector.Sector;
import project.Warship.Warship;
import project.Warship.SeparatistWarship.SeparatistDestroyer;

public class RepublicCruiser extends Warship {
	private ArrayList<Crewman> captives = new ArrayList<>();

	public RepublicCruiser(int id, String name, Sector currentSector, int coordinate, ArrayList<Crewman> crew) {
		super(id, name, currentSector, coordinate, crew);
		affiliation = Affiliation.REPUBLIC;
		armamentPower = 100;
		shieldPower = 100;
		crewCapacity = 10;
	}

	public ArrayList<Crewman> getCaptives() {
		return captives;
	}

	/**
	 * Visits coruscant, replenishes all Jedis sanity in the ship and sets captives
	 * imprisoned and removes them from ship.
	 */
	public void visitCoruscant() {
		for (Crewman crewman : crew) {
			if (crewman instanceof Jedi) {
				((Jedi) crewman).replenishSanity();
			}
		}

		for (Crewman captive : captives) {
			captive.setInCorousant(true);
		}
		captives.clear();
	}

	@Override
	public void addCrewman(Crewman crewman) {
		if (!isDestroyed() && crewman.isAlive() && crewman.isFree() && !isFull() && !(crewman instanceof Sith)) {
			crew.add(crewman);
			crewman.setFree(false);
			crewman.setCurrentShip(this);
		}
	}

	@Override
	public General getCommander() {
		int highestExp = Integer.MIN_VALUE;
		General commander = null;
		for (Crewman crewman : crew) {
			if (crewman instanceof Jedi && (((General) crewman).getExperience() > highestExp
					|| ((General) crewman).getExperience() == highestExp && crewman.getId() < commander.getId())) {
				commander = ((General) crewman);
				highestExp = commander.getExperience();
			}
		}
		return commander;
	}

	/**
	 * Conducts attack on the enemy ship. First this ship jumps to the enemy ship's
	 * sector and coordinate. Then, Jedi and Sith have talk and Jedi's sanity is
	 * adjusted. If Jedi losses all sanity, they turn to the dark side.
	 * <p>
	 * If not, power outputs are compared. If republic ship has higher power output,
	 * separatist ship is destroyed. All the officers are captured. Siths with
	 * higher combat power than Jedi general uses escape pods to escape if there is
	 * any. Remaining sith are killed by Jedi and their experience are collected.
	 * <p>
	 * Otherwise, republic ship is destroyed. Everyone on the republic ship
	 * (including captives) are killed by Sith and their experience and intrinsic
	 * level are collected.
	 * <p>
	 * This function is not inherited from warship class due to design choice.
	 * Because in power equality, republic attack and separatist attack behave
	 * differently; it would be necessary to write different if else cases for
	 * attacker types. That would be unnecessarily long (twice of the length of this
	 * function) and hard to read. So they are divided in child classes.
	 * 
	 * @param enemyShip Target ship of the attack
	 * @see Warship#yesMaster(Jedi, Sith, RepublicCruiser, SeparatistDestroyer)
	 */
	public void attack(SeparatistDestroyer enemyShip) {
		jumpToSector(enemyShip.getCurrentSector(), enemyShip.getCoordinate());
		Jedi jedi = ((Jedi) getCommander());
		Sith sith = ((Sith) enemyShip.getCommander());
		jedi.setSanity(sith.getPersuasion());
		if (jedi.getSanity() <= 0) {
			yesMaster(jedi, sith, this, enemyShip);
		}

		else if (getPowerOutput() > enemyShip.getPowerOutput()) {
			enemyShip.setDestroyed(getName());
			ArrayList<Sith> siths = new ArrayList<>();
			for (Crewman captive : enemyShip.getCrew()) {
				if (captive instanceof Sith) {
					siths.add((Sith) captive);
				} else {
					captive.setImprisoned(true);
					addCaptive(captive);
					captive.setCurrentShip(this);
				}
			}
			enemyShip.getCrew().clear();

			siths.sort((sith1, sith2) -> {
				return sith1.compareTo(sith2);
			});
			int jediPower = jedi.getCombatPower();
			for (Sith defeatedSith : siths) {
				if (jediPower >= defeatedSith.getCombatPower()) {
					defeatedSith.setDead(jedi.getName());
					jedi.setExperience(defeatedSith.getExperience());
				} else if (enemyShip.hasEscapePod()) {
					defeatedSith.setFree(true);
					enemyShip.useEscapePod();
				} else {
					defeatedSith.setDead(jedi.getName());
					jedi.setExperience(defeatedSith.getExperience());
				}
				defeatedSith.setCurrentShip(null);
			}
		} else {
			setDestroyed(enemyShip.getName());
			for (Crewman crewman : getCrew()) {
				crewman.setDead(sith.getName());
				if (crewman instanceof Jedi) {
					sith.setExperience(((Jedi) crewman).getExperience());
				} else {
					sith.setExperience(((Officer) crewman).getIntrinsicLevel());
				}
			}
			for (Crewman crewman : getCaptives()) {
				crewman.setDead(sith.getName());
				sith.setExperience(((Officer) crewman).getIntrinsicLevel());
			}
			getCaptives().clear();
			crew.clear();
		}
	}

	public void addCaptive(Crewman captive) {
		captives.add(captive);
	}
}
