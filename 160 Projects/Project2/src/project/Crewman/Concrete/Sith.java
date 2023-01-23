package project.Crewman.Concrete;

import project.Crewman.General;

public class Sith extends General{
	int persuasion;
	
	public Sith(int id, String name, int experience, int midichloarian, int persuasion) {
		super(id, name, experience, midichloarian);
		this.persuasion = persuasion;
	}
	
	public int getPersuasion() {
		return persuasion;
	}

	@Override
	public int getForcePower() {
		return 4 * midichlorian;
	}

	@Override
	public int getCombatPower() {
		return getForcePower() + experience + persuasion;
	}
	
	protected String getType() {
		return "Sith";
	}
}
