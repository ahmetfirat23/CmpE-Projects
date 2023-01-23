package project.Warship.SeparatistWarship;

import java.util.ArrayList;

import project.Crewman.Crewman;
import project.Sector.Sector;

public class SeparatistBattleship extends SeparatistDestroyer {

	public SeparatistBattleship(int id, String name, Sector currentSector, int coordinate, ArrayList<Crewman> crew) {
		super(id, name, currentSector, coordinate, crew);
		escapePods = 3;
		armamentPower = 400;
		shieldPower = 200;
		crewCapacity = 20;
	}
}
