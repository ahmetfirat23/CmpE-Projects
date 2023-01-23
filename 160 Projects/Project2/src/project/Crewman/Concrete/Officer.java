package project.Crewman.Concrete;

import project.Intrinsic;
import project.Crewman.Crewman;

public class Officer extends Crewman implements Comparable<Officer>{
	private int intrinsicLevel;
	private Intrinsic intrinsic;
	
	public Officer(int id, String name, Intrinsic intrinsic, int intrinsicLevel) {
		super(id, name);
		this.intrinsic = intrinsic;
		this.intrinsicLevel = intrinsicLevel;
	}
	
	public void train() {
		intrinsicLevel += intrinsicLevel == 10 ? 0 : 1;
	}
	
	public int getIntrinsicLevel() {
		return intrinsicLevel;
	}

	public Intrinsic getIntrinsic() {
		return intrinsic;
	}

	protected String getType() {
		return "Officer";
	}
	
	protected void logSecondLine() {
		System.out.println(intrinsic.toString() + " " + intrinsicLevel);
	}
	
	/**
	 * Returns negative if officer has higher intrinsic level. In equality returns negative when officer has lower ID.
	 */
	@Override
	public int compareTo(Officer other) {
		if (other.getIntrinsicLevel() - getIntrinsicLevel() == 0) {
			return getId() - other.getId();
		}
		return other.getIntrinsicLevel() - getIntrinsicLevel();
	}
	
}
