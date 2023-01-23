package project.Crewman;

import project.IForceUser;

public abstract class General extends Crewman implements IForceUser, Comparable<General> {
	protected int experience;
	protected int midichlorian;


	public General(int id, String name, int experience, int midichloarian) {
		super(id, name);
		this.experience = experience;
		this.midichlorian = midichloarian;
	}

	public int getExperience() {
		return experience;
	}

	public void setExperience(int experience) {
		this.experience += experience;
	}

	public int getMidichlorian() {
		return midichlorian;
	}

	/**
	 * Returns negative if general has higher combat power. In equality returns negative when general has lower ID.
	 */
	@Override
	public int compareTo(General other) {
		if (other.getCombatPower() - getCombatPower() == 0) {
			return getId() - other.getId();
		}
		return other.getCombatPower() - getCombatPower();
	}
	
	@Override
	protected void logSecondLine() {
		System.out.println(getCombatPower());
	}
}
