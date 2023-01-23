import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

public class Project3 {
    public static void main(String[] args) throws FileNotFoundException {
        File outFile = new File(args[1]);
        PrintStream printStream = new PrintStream(outFile);
        System.setOut(printStream);

        File inFile = new File(args[0]);
        Scanner scanner = new Scanner(inFile);

        String[] firstLine = scanner.nextLine().split(" ");
        int A = Integer.parseInt(firstLine[0]);
        int F = Integer.parseInt(firstLine[1]);

        ACC[] accs = new ACC[A]; //will be printed at the end
        Simulator[] sims = new Simulator[A];
        HashMap<String, Simulator> simulatorMap = new HashMap<>();

        for (int i = 0; i < A; i++){
            String[] line = scanner.nextLine().split(" ");

            ACC acc = new ACC(line[0]);
            accs[i] = acc;

            Simulator simulator = new Simulator(acc);
            sims[i] = simulator;
            simulatorMap.put(line[0], simulator);

            for (int j = 1; j < line.length; j++){
                ATC atc = new ATC(line[j]);
                simulator.atcs.add(atc);
                simulator.atcMap.add(atc);
            }
        }
        for(int i = 0; i < F; i++){
            String[] line = scanner.nextLine().split(" ");

            int entryTime = Integer.parseInt(line[0]);
            String flightCode = line[1];
            Simulator simulator = simulatorMap.get(line[2]);
            ACC acc = simulator.acc;
            String strStartAtc = line[3];
            ATC startAtc = simulator.atcMap.find(strStartAtc);
            String strEndAtc = line[4];
            ATC endAtc = simulator.atcMap.find(strEndAtc);
            int[] durations = new int[21];
            for (int j = 5; j<line.length; j++){
                durations[j-5] = Integer.parseInt(line[j]);
            }

            Flight flight = new Flight(entryTime, flightCode, simulator, acc, startAtc, endAtc, durations);
            simulator.enteringQueue.add(flight);
        }

        for(Simulator sim : sims){
            sim.start();
            System.out.println(sim.acc.name + " " + sim.time + sim.atcMap.listNames());
        }
    }
}