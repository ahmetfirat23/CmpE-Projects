package project.Crewman.Concrete;

import project.Crewman.General;

public class Jedi extends General {
	int sanity;
	int intelligence;
	
	public Jedi(int id, String name, int experience, int midichloarian, int intelligence) {
		super(id, name, experience, midichloarian);
		sanity = 100;
		this.intelligence = intelligence;
	}
	
	public int getSanity() {
		return sanity;
	}
	
	public int getIntelligence() {
		return intelligence;
	}

	@Override
	public int getForcePower() {
		return 3 * midichlorian;
	}

	@Override
	public int getCombatPower() {
		return getForcePower() + experience + sanity - 100 + intelligence;
	}
	
	public void replenishSanity() {
		sanity = 100;
	}
	
	public void setSanity(int persuasion) {
		sanity -= persuasion - intelligence >  0 ? persuasion - intelligence : 0 ;
		if(sanity < 0)
			sanity = 0;
	}
	
	protected String getType() {
		return "Jedi";
	}
}
