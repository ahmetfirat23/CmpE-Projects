import java.util.LinkedList;
import java.util.PriorityQueue;

public class ACC {
    public String name;
    public LinkedList<Flight> readyQueue;
    public PriorityQueue<Flight> waitingQueue;

    public ACC(String name){
        this.name = name;
        readyQueue = new LinkedList<>(); //add peek poll
        waitingQueue = new PriorityQueue<>(new Flight.WaitingComparator());
    }
}
