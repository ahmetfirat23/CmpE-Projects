import java.util.*;

public class Simulator {
    public ACC acc;
    public ArrayList<ATC> atcs;
    public AirportHashMap atcMap;
    public int time;
    public PriorityQueue<Flight> enteringQueue;
    public PriorityQueue<Flight> processingQueue;
    public HashMap<Flight, Queue<Flight>> processingMap;


    public Simulator(ACC acc) {
        this.acc = acc;

        time = 0;

        atcs = new ArrayList<>();
        atcMap = new AirportHashMap();

        enteringQueue = new PriorityQueue<>(new Flight.EnteringComparator());
        processingQueue = new PriorityQueue<>(new Flight.ProcessingComparator());
        processingMap = new HashMap<>();
    }

    public void start(){
        while(true){
            prepareProcessingQueue();
            if(processingQueue.isEmpty()){
                break;
            }

            Flight decoy = processingQueue.poll();
            Queue<Flight> queue =  processingMap.get(decoy);

            processingQueue.clear();
            processingMap.clear();

            Flight target = queue.poll();
            assert target != null;
            int duration = target.getCurrentEventDuration();
            process(target ,queue);
            update(duration, queue);
            time += duration; //update time before placement because comparisons must be made acc. to new time
            placeFlight(target);
        }

    }

    public void prepareProcessingQueue(){
        Flight acc1 = acc.readyQueue.peek();
        if(acc1!=null) {
            processingQueue.add(acc1);
            processingMap.put(acc1, acc.readyQueue);
        }
        Flight acc2 = acc.waitingQueue.peek();
        if(acc2!=null) {
            processingQueue.add(acc2);
            processingMap.put(acc2, acc.waitingQueue);
        }
        Flight entering = enteringQueue.peek();
        if(entering!=null){
            processingQueue.add(entering);
            processingMap.put(entering, enteringQueue);
        }
        for (ATC atc : atcs){
            Flight atc1 = atc.readyQueue.peek();
            if(atc1!=null){
                processingQueue.add(atc1);
                processingMap.put(atc1, atc.readyQueue);
            }
            Flight atc2 = atc.waitingQueue.peek();
            if(atc2!=null){
                processingQueue.add(atc2);
                processingMap.put(atc2, atc.waitingQueue);
            }
        }

    }

    public void process(Flight f, Queue<Flight> queue){
        int idx = f.currentEventIdx;
        int duration = f.getCurrentEventDuration();
        if(f.isNew){ //entering
            f.doEntering(duration, true, queue);
        }
        else if( idx == 1 || idx == 11 || idx == 4 || idx == 6 ||
                idx == 8 || idx ==14 || idx ==16|| idx == 18){ //waiting
            f.doWaiting(duration, true, queue);
        }
        else if(idx == 0 || idx == 2 || idx == 10 || idx == 12 || idx == 20) { //acc running
            f.doACCRunning(duration, true);
        }
        else if(idx == 3 || idx == 5 || idx == 7 || idx == 9 ||
                idx ==13 || idx == 15 || idx == 17 || idx == 19){ //atc running
            f.doATCRunning(duration, true);
        }
    }

    public void update(int duration, Queue<Flight> queue){   //update everything as duration
        //  (if not empty it will be currently active (not scheduled) and will not be finished after addition.
        //In entering queue instead of decreasing duration decrease entering time
        if(!acc.readyQueue.isEmpty() && queue != acc.readyQueue){
            Flight f = acc.readyQueue.peek();
            f.doACCRunning(duration, false);
        }
        if(!acc.waitingQueue.isEmpty()){
            for (Flight f : acc.waitingQueue){
                f.doWaiting(duration, false, null);
            }
        }
        if(!enteringQueue.isEmpty()){
            for(Flight f : enteringQueue){
                f.doEntering(duration, false, null);
            }
        }
        for (ATC atc: atcs){
            if(!atc.readyQueue.isEmpty() && queue != atc.readyQueue){
                Flight f = atc.readyQueue.peek();
                f.doATCRunning(duration, false);
            }
            if(!atc.waitingQueue.isEmpty()){
                for (Flight f: atc.waitingQueue){
                    f.doWaiting(duration, false, null);
                }
            }
        }
    }

    public void placeFlight(Flight flight){
        int idx = flight.currentEventIdx; //next event
        if (idx == 1 || idx ==11){ //acc waiting
            acc.waitingQueue.add(flight);
        }
        else if(idx == 0 || idx == 2 || idx == 10 || idx == 12 || idx == 20){//acc ready
            acc.readyQueue.add(flight);
            if (flight.eventDurations[flight.currentEventIdx] > flight.remainingACCtime){
                flight.interrupted = true;
            }
        }
        else if(idx == 4 || idx == 6 || idx == 8){// start atc waiting
            flight.startAtc.waitingQueue.add(flight);
        }
        else if(idx == 14 || idx == 16 || idx == 18){//end atc waiting
            flight.endAtc.waitingQueue.add(flight);
        }
        else if(idx == 3 || idx == 5 || idx == 7 || idx == 9){//start atc ready
            flight.startAtc.readyQueue.add(flight);
        }
        else if(idx == 13 || idx == 15 || idx == 17 || idx == 19){//end atc ready
            flight.endAtc.readyQueue.add(flight);
        }
    }
}
