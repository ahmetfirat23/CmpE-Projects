import java.util.Comparator;
import java.util.Queue;

public class Flight {
    public int entryTime;
    public String flightCode;
    public Simulator simulator;
    public ACC acc;
    public ATC startAtc;
    public ATC endAtc;
    public int[] eventDurations;
    public int currentEventIdx; //index of the current operation
    public boolean interrupted; //Only true in acc longer then 30
    public int remainingACCtime;
    public boolean isNew;

    public Flight(int entryTime, String flightCode, Simulator simulator, ACC acc, ATC startAtc, ATC endAtc, int[] durations){
        this.entryTime = entryTime;
        this.flightCode = flightCode;
        this.simulator = simulator;
        this.acc = acc;
        this.startAtc = startAtc;
        this.endAtc = endAtc;
        eventDurations = durations;
        currentEventIdx = 0;
        interrupted = false;
        remainingACCtime = 30;
        isNew = true;
    }

    public int getCurrentEventDuration(){
        if(isNew){
            return entryTime;
        }
        if (currentEventIdx==0 || currentEventIdx == 2 || currentEventIdx == 10 || currentEventIdx == 12 || currentEventIdx == 20) {
            return Math.min(remainingACCtime, eventDurations[currentEventIdx]);
        }
        return eventDurations[currentEventIdx];
    }

    public void doWaiting(int duration, boolean isProcessed, Queue<Flight> queue){ // There are two possibilities either it is processed or updated
        if (isProcessed){
            eventDurations[currentEventIdx] = 0;
            currentEventIdx++;
        }
        else{
            eventDurations[currentEventIdx] -= duration;
        }
    }

    public void doACCRunning(int duration, boolean isProcessed){ // There are two possibilities either it is processed or updated
        if(isProcessed) {
            eventDurations[currentEventIdx] -= getCurrentEventDuration();
            remainingACCtime=30;
            if (getCurrentEventDuration() <= remainingACCtime) {
                interrupted = false;
            }
            if(getCurrentEventDuration() == 0){
                currentEventIdx++;
            }
        }
        else{
            eventDurations[currentEventIdx] -= duration;
            remainingACCtime -= duration;
        }
    }

    public void doATCRunning(int duration, boolean isProcessed){ // There are two possibilities either it is processed or updated
        if(isProcessed){
            eventDurations[currentEventIdx] = 0;
            currentEventIdx++;
        }
        else{
            eventDurations[currentEventIdx] -= duration;
        }
    }

    public void doEntering(int duration, boolean isProcessed, Queue<Flight> queue){
        if(isProcessed){
            entryTime = 0;
            isNew = false;
        }
        else{
            entryTime -= duration;
        }
    }

    public static class WaitingComparator implements Comparator<Flight> {
        public int compare(Flight f1, Flight f2) {
            if(f1.getCurrentEventDuration() != f2.getCurrentEventDuration()){
                return f1.getCurrentEventDuration() - f2.getCurrentEventDuration();
            }
            return f1.flightCode.compareTo(f2.flightCode);
        }
    }

    public static class EnteringComparator implements Comparator<Flight>{
        public int compare(Flight f1, Flight f2){
            if(f1.entryTime != f2.entryTime){
                return f1.entryTime - f2.entryTime;
            }
            return f1.flightCode.compareTo(f2.flightCode);
        }
    }

    public static class ProcessingComparator implements Comparator<Flight>{
        public int compare(Flight f1, Flight f2){
            if(f1.getCurrentEventDuration() != f2.getCurrentEventDuration()){
                return f1.getCurrentEventDuration() - f2.getCurrentEventDuration();
            }
            else if(f1.interrupted && !f2.interrupted){
                return 1;
            }
            else if(f2.interrupted && !f1.interrupted){
                return -1;
            }
            return f1.flightCode.compareTo(f2.flightCode);
        }
    }
}
