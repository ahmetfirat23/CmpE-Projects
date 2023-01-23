package project.Crewman;

import project.Warship.Warship;

public abstract class Crewman {
	protected int id;
	protected String name;
	protected Warship currentShip;
	protected boolean isFree;
	protected boolean isImprisoned;
	protected boolean isAlive;
	protected boolean isInCorousant;
	private String killedBy;
	
	public Crewman(int id, String name) {
		this.id = id;
		this.name = name;
		isFree = true;
		isImprisoned = false;
		isAlive = true;
	}

	public int getId() {
		return id;
	}

	public String getName() {
		return name;
	}

	public Warship getCurrentShip() {
		return currentShip;
	}

	public void setCurrentShip(Warship currentShip) {
		this.currentShip = currentShip;
	}

	public boolean isFree() {
		return isFree;
	}

	public void setFree(boolean isFree) {
		this.isFree = isFree;
	}
	
	public boolean isImprisoned() {
		return isImprisoned;
	}

	public void setImprisoned(boolean isImprisoned) {
		this.isImprisoned = isImprisoned;
	}
	
	public boolean isInCorousant() {
		return isInCorousant;
	}

	public void setInCorousant(boolean isInCorousant) {
		this.isInCorousant = isInCorousant;
	}
	
	public boolean isAlive() {
		return isAlive;
	}

	public void setDead(String killedBy) {
		isAlive = false;
		this.killedBy = killedBy;
	}

	protected abstract String getType();
	protected abstract void logSecondLine();
	
	public void logOutput() {
		if(isFree) {
			System.out.println(getType() + " " + name + " is free");
		} else if(!isAlive) {
			System.out.println(getType() + " " + name + " is killed by " + killedBy);
		} else if(isInCorousant()) {
			System.out.println(getType() + " " + name + " is imprisoned");
		} else {
			System.out.println(getType() + " " + name + " is in " + currentShip.getName());
		}
		logSecondLine();
	}
}
