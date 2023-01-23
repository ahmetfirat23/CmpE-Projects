import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class project4 {
    public static void main(String[] args) throws IOException {
        File outFile = new File(args[1]);
        PrintStream printStream = new PrintStream(outFile);
        System.setOut(printStream);

        FileReader inFile = new FileReader(args[0]);
        BufferedReader scanner = new BufferedReader(inFile);

        String firstLine = scanner.readLine();
        int V = Integer.parseInt(firstLine);
        String secondLine = scanner.readLine();
        int M = Integer.parseInt(secondLine);
        String[] stPoints = scanner.readLine().split(" ");
        HashSet<String> flagPoints = new HashSet<>(List.of(scanner.readLine().split(" ")));
        HashMap<String, Dijkstra.Node> lookupTable = new HashMap<>();
        Dijkstra.Node flagSource = null;
        Dijkstra.Node sourceNode = null;
        for (int i = 0 ; i<V; i++){
            String[] line = scanner.readLine().split(" ");
            String point = line[0];

            if (!lookupTable.containsKey(point)){
                lookupTable.put(point, new Dijkstra.Node(point));
            }
            Dijkstra.Node node = lookupTable.get(point);

            if(node.name.equals(stPoints[0])){
              sourceNode = node;
            }
            if(flagPoints.contains(node.name)){
                node.isFlag = true;
                flagSource = node;
            }

            for (int j = 1; j<line.length; j+=2 ){
                String toPoint = line[j];
                int distance = Integer.parseInt(line[j+1]);

                if(!lookupTable.containsKey(toPoint)){
                    lookupTable.put(toPoint, new Dijkstra.Node(toPoint));
                }
                Dijkstra.Node adj = lookupTable.get(toPoint);
                node.addAdjacent(adj, distance);
                adj.addAdjacent(node, distance);
            }
        }

        Dijkstra dijkstra = new Dijkstra(sourceNode, stPoints[1], flagSource, V, M);
        dijkstra.solveGraph();

        lookupTable.forEach((key, value)->{
            value.setMinDistance(Integer.MAX_VALUE);
        });

        dijkstra.findMinFlags();
    }
}
