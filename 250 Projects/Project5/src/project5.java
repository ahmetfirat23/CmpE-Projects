import java.io.*;
import java.util.ArrayList;

public class project5 {
    public static void main(String[] args) throws IOException {
        FordFulkerson ff = new FordFulkerson();

        File outFile = new File(args[1]);
        PrintStream printStream = new PrintStream(outFile);
        System.setOut(printStream);

        FileReader inFile = new FileReader(args[0]);
        BufferedReader reader = new BufferedReader(inFile);

        String firstLine = reader.readLine();
        int V = Integer.parseInt(firstLine);
        String[] line = reader.readLine().split(" ");
        ArrayList<ArrayList<FordFulkerson.Edge>> graph = ff.createGraph(V+8);
        for (int i = 0; i<6; i++){
            ff.addEdge(graph,0 ,i+1, Integer.parseInt(line[i]));
            String inp = reader.readLine();
            String[] inps = inp.split(" ");
            int sinp = Integer.parseInt(inps[0].substring(1)) + 1;
            addEdge(ff, V, graph, inps, sinp);
        }
        for(int i = 0; i < V; i++) {
            String inp = reader.readLine();
            String[] inps = inp.split(" ");
            int sinp = Integer.parseInt(inps[0].substring(1)) + 7;
            addEdge(ff, V, graph, inps, sinp);
        }
        ff.maxFlow(graph,V+7);
    }

    private static void addEdge(FordFulkerson ff, int v, ArrayList<ArrayList<FordFulkerson.Edge>> graph, String[] inps, int sinp) {
        for (int j = 1; j < inps.length; j += 2) {
            int ind;
            if (!inps[j].equals("KL")) {
                ind = Integer.parseInt(inps[j].substring(1)) + 7;
            } else {
                ind = v + 7;
            }
            int c = Integer.parseInt(inps[j + 1]);
            ff.addEdge(graph, sinp, ind, c);
        }
    }
}
