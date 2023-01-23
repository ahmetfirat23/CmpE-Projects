
import java.util.*;

public class FordFulkerson {

    public static class Edge {
        int to;
        int capacity;
        int reverse;

        public Edge(int to, int capacity, int reverse) {
            this.to = to;
            this.capacity = capacity;
            this.reverse = reverse;
        }
    }

    public ArrayList<ArrayList<Edge>> createGraph(int V) {
        ArrayList<ArrayList<Edge>> graph = new ArrayList<>(V);
        for (int i = 0; i < V; i++)
            graph.add(new ArrayList<>());
        return graph;
    }

    public void addEdge(ArrayList<ArrayList<Edge>> graph, int s, int to, int capacity) {
        if (s < to){
            graph.get(s).add(new Edge(to, capacity, graph.get(to).size()));
            graph.get(to).add(new Edge(s, 0, graph.get(s).size() - 1));
        }
        else{
            ArrayList<Edge> arr = graph.get(s);
            boolean modified = false;
            for (Edge e : arr){
                if(e.to == to){
                    e.capacity = capacity;
                    modified = true;
                    break;
                }
            }
            if(!modified){
                graph.get(s).add(new Edge(to, capacity, graph.get(to).size()));
                graph.get(to).add(new Edge(s, 0, graph.get(s).size() - 1));
            }
        }
    }

    private boolean bfs(ArrayList<ArrayList<Edge>> graph, int sink, ArrayList<ArrayList<Integer>> parents) {
        LinkedList<Edge> queue = new LinkedList<>();
        queue.addFirst(new Edge(0, -1, -1));

        boolean[] visited = new boolean[graph.size()];
        visited[0] = true;

        for(int i = 0; i<graph.size(); i++){
            parents.set(i, new ArrayList<>(2));
        }
        ArrayList<Integer> sMap = parents.get(0);
        sMap.add(-1); //To node first
        sMap.add(-1); //Edge index second

        while (!queue.isEmpty()) {
            int to = queue.poll().to;
            ArrayList<Edge> toEdges = graph.get(to);
            for (int i = 0; i<toEdges.size(); i++){
                Edge edge = toEdges.get(i);
                if (edge.capacity > 0 && !visited[edge.to]) {
                    queue.add(edge);
                    visited[edge.to] = true;
                    ArrayList<Integer> toMap = parents.get(edge.to);
                    toMap.add(to);
                    toMap.add(i);
                    if (edge.to == sink) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private void dfs(ArrayList<ArrayList<Edge>> graph, boolean[] visited){
        LinkedList<Edge> stack = new LinkedList<>();
        stack.addLast(new Edge(0, -1, -1));
        visited[0] = true;
        while(!stack.isEmpty()){
            int to = stack.poll().to;
            visited[to] = true;
            ArrayList<Edge> toEdges = graph.get(to);
            for (Edge edge : toEdges) {
                if (edge.capacity > 0 && !visited[edge.to]) {
                    stack.addLast(edge);
                }
            }
        }
    }

    public void maxFlow(ArrayList<ArrayList<Edge>> graph, int sink) {
        ArrayList<ArrayList<Integer>> parents = new ArrayList<>();
        for(int i = 0; i<graph.size(); i++){
            parents.add(new ArrayList<>(2));
        }

        ArrayList<ArrayList<Edge>> copy = new ArrayList<>();
        for(int i = 0; i< graph.size(); i++){
            ArrayList<Edge> arr = graph.get(i);
            copy.add(new ArrayList<>(arr.size()));
            for(Edge e: arr){
                Edge e_c = new Edge(e.to, e.capacity, e.reverse);
                copy.get(i).add(e_c);
            }
        }

        int maxFlow = 0;
        while (bfs(graph, sink, parents)) {
            int flow = Integer.MAX_VALUE;
            for (int node = sink; node != 0; node = parents.get(node).get(0)) {
                int reverse = parents.get(node).get(0);
                int edgeIdx = parents.get(node).get(1);
                int capacity = graph.get(reverse).get(edgeIdx).capacity;
                if (capacity < flow) {
                    flow = capacity;
                }
            }
            for (int node = sink; node != 0; node = parents.get(node).get(0)) {
                int reverse = parents.get(node).get(0);
                int edgeIdx = parents.get(node).get(1);
                Edge edge = graph.get(reverse).get(edgeIdx);
                Edge reverseEdge = graph.get(node).get(edge.reverse);
                edge.capacity -= flow;
                reverseEdge.capacity += flow;
            }
            maxFlow += flow;
        }
        System.out.println(maxFlow);

        boolean[] visited = new boolean[graph.size()];
        dfs(graph, visited);

        for(int i = 0; i < graph.size(); i++){
            ArrayList<Edge> arr = copy.get(i);
            for (Edge e : arr) {
                if (e.capacity > 0 && visited[i] && !visited[e.to]) {
                    if (i == 0) {
                        System.out.println("r" + (e.to - 1));
                    }
                    else if (i < 7)
                        if (e.to != graph.size() - 1) {
                            System.out.println("r" + (i - 1) + " " + "c" + (e.to - 7));
                        }
                        else {
                            System.out.println("r" + (i - 1) + " " + "KL");
                        }
                    else {
                        if (e.to != graph.size() - 1) {
                            System.out.println("c" + (i - 7) + " " + "c" + (e.to - 7));
                        }
                        else {
                            System.out.println("c" + (i - 7) + " " + "KL");
                        }
                    }
                }
            }
        }
    }
}