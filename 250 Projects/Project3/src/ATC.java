import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;

public class ATC {
    public String name;
    public Queue<Flight> readyQueue;
    public PriorityQueue<Flight> waitingQueue;


    public ATC(String name) {
        this.name = name;
        readyQueue = new LinkedList<>(); //addfirst getfirst polllast -> returns null if the list is empty
        waitingQueue = new PriorityQueue<>(new Flight.WaitingComparator());
    }
}
