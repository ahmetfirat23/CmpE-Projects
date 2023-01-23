package project.Warship.SeparatistWarship;

import java.util.ArrayList;

import project.Crewman.Crewman;
import project.Sector.Sector;

public class SeparatistFrigate extends SeparatistDestroyer{

	public SeparatistFrigate(int id, String name, Sector currentSector, int coordinate, ArrayList<Crewman> crew) {
		super(id, name, currentSector, coordinate, crew);
		escapePods = 2;
		armamentPower = 120;
		shieldPower = 100;
		crewCapacity = 12;
	}
	
}
