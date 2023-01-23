package project.Warship.SeparatistWarship;

import java.util.ArrayList;

import project.Affiliation;
import project.Crewman.Crewman;
import project.Crewman.General;
import project.Crewman.Concrete.Jedi;
import project.Crewman.Concrete.Officer;
import project.Crewman.Concrete.Sith;
import project.Sector.Sector;
import project.Warship.Warship;
import project.Warship.RepublicWarship.RepublicCruiser;

public class SeparatistDestroyer extends Warship {
	protected int escapePods = 1;

	public SeparatistDestroyer(int id, String name, Sector currentSector, int coordinate, ArrayList<Crewman> crew) {
		super(id, name, currentSector, coordinate, crew);
		affiliation = Affiliation.SEPARATISTS;
		armamentPower = 80;
		shieldPower = 60;
		crewCapacity = 7;
	}

	public boolean hasEscapePod() {
		return escapePods > 0;
	}

	public void useEscapePod() {
		escapePods--;
	}

	@Override
	public void addCrewman(Crewman crewman) {
		if (!isDestroyed() && crewman.isAlive() && crewman.isFree() && !isFull() && !crewman.isImprisoned()
				&& !crewman.isInCorousant() && !(crewman instanceof Jedi)) {
			int prevPower = getPowerOutput();
			crew.add(crewman);
			int power = getPowerOutput();
			if (power > prevPower) {
				crewman.setFree(false);
				crewman.setCurrentShip(this);
			} else {
				crew.remove(crewman);
			}
		}
	}

	@Override
	public General getCommander() {
		int highestPow = Integer.MIN_VALUE;
		General commander = null;
		for (Crewman crewman : crew) {
			if (crewman instanceof Sith && (((Sith) crewman).getCombatPower() > highestPow
					|| ((Sith) crewman).getCombatPower() == highestPow && crewman.getId() < commander.getId())) {
				commander = ((General) crewman);
				highestPow = commander.getCombatPower();
			}
		}
		return commander;
	}

	/**
	 * Conducts attack on the enemy ship. First this ship jumps to the enemy ship's
	 * sector and coordinate. Then, Jedi and Sith have talk and Jedi's sanity is
	 * adjusted. If Jedi losses all sanity, they turn to the dark side.
	 * <p>
	 * If not, power outputs are compared. If separatist ship has higher power
	 * output, republic ship is destroyed. Everyone on the republic ship (including
	 * captives) are killed by Sith and their experience and intrinsic level are
	 * collected.
	 * <p>
	 * Otherwise, separatist ship is destroyed. All the officers are captured. Siths
	 * with higher combat power than Jedi general uses escape pods to escape if
	 * there is any. Remaining sith are killed by Jedi and their experience are
	 * collected.
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
	public void attack(RepublicCruiser enemyShip) {
		jumpToSector(enemyShip.getCurrentSector(), enemyShip.getCoordinate());
		Sith sith = ((Sith) getCommander());
		Jedi jedi = ((Jedi) enemyShip.getCommander());
		jedi.setSanity(sith.getPersuasion());
		if (jedi.getSanity() <= 0) {
			yesMaster(jedi, sith, enemyShip, this);
		} else if (getPowerOutput() > enemyShip.getPowerOutput()) {
			enemyShip.setDestroyed(getName());
			for (Crewman crewman : enemyShip.getCrew()) {
				crewman.setDead(sith.getName());
				if (crewman instanceof Jedi) {
					sith.setExperience(((Jedi) crewman).getExperience());
				} else {
					sith.setExperience(((Officer) crewman).getIntrinsicLevel());
				}
			}
			for (Crewman crewman : enemyShip.getCaptives()) {
				crewman.setDead(sith.getName());
				sith.setExperience(((Officer) crewman).getIntrinsicLevel());
			}
			enemyShip.getCaptives().clear();
			enemyShip.getCrew().clear();

		} else {
			setDestroyed(enemyShip.getName());
			ArrayList<Sith> siths = new ArrayList<>();
			for (Crewman captive : getCrew()) {
				if (captive instanceof Sith) {
					siths.add((Sith) captive);
				} else {
					captive.setImprisoned(true);
					enemyShip.addCaptive(captive);
					captive.setCurrentShip(enemyShip);
				}
			}
			getCrew().clear();

			siths.sort((sith1, sith2) -> {
				return sith1.compareTo(sith2);
			});
			int jediPower = jedi.getCombatPower();
			for (Sith defeatedSith : siths) {
				if (jediPower >= defeatedSith.getCombatPower()) {
					defeatedSith.setDead(jedi.getName());
					jedi.setExperience(defeatedSith.getExperience());
				} else if (hasEscapePod()) {
					defeatedSith.setFree(true);
					useEscapePod();

				} else {
					defeatedSith.setDead(jedi.getName());
					jedi.setExperience(defeatedSith.getExperience());
				}
				defeatedSith.setCurrentShip(null);
			}
		}
	}
}
