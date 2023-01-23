import java.util.*;

public class Dijkstra {
    Node sourceNode;
    String terminal;
    HashSet<String> visited;
    PriorityQueue<Node> distanceQueue;
    Node flagSource;
    int gSize;
    int fSize;
    public Dijkstra(Node sourceNode, String terminal, Node flagSource, int gSize, int fSize ){
        this.terminal = terminal;
        this.flagSource = flagSource;
        visited = new HashSet<>();
        distanceQueue = new PriorityQueue<>();
        this.sourceNode = sourceNode;
        this.gSize = gSize;
        this.fSize= fSize;
    }

    public static class Node implements Comparable<Node>{
        String name;
        int minDistance;
        boolean inQueue=false;
        boolean foundFlag = false;
        boolean isFlag = false;
        ArrayList<Node> adjacents;
        ArrayList<Integer> distances;


        public Node(String name){
            this.name = name;
            this.minDistance = Integer.MAX_VALUE;
            adjacents = new ArrayList<>();
            distances = new ArrayList<>();
        }

        public void addAdjacent(Node node, int distance){
            adjacents.add(node);
            distances.add(distance);
        }

        public void setMinDistance(int minDistance) {
            this.minDistance = minDistance;
        }

        @Override
        public int compareTo(Node n2){
            return Integer.compare(this.minDistance, n2.minDistance);
        }
    }

    public void solveGraph(){
        sourceNode.setMinDistance(0);
        distanceQueue.add(sourceNode);

        while(visited.size() != gSize){
            if(distanceQueue.isEmpty()){
                System.out.println(-1);
                return;
            }

            Node node = distanceQueue.poll();

            if (visited.contains(node.name)){
                continue;
            }

            visited.add(node.name);

            if(node.name.equals(terminal)){
                System.out.println(node.minDistance);
                return;
            }

            for(int i = 0; i<node.adjacents.size(); i++){
                Node adjNode = node.adjacents.get(i);
                int value = node.distances.get(i);
                if (adjNode.minDistance >= value + node.minDistance){
                    adjNode.setMinDistance(value + node.minDistance);
                    distanceQueue.add(adjNode);
                }
            }
        }
    }

    public void findMinFlags(){
        visited = new HashSet<>();
        distanceQueue = new PriorityQueue<>();
        HashMap<Node, Integer> foundFlags = new HashMap<>();
        flagSource.setMinDistance(0);
        distanceQueue.add(flagSource);

        while(foundFlags.size()!= fSize){
            if(distanceQueue.isEmpty()){
                System.out.println(-1);
                return;
            }

            Node node = distanceQueue.poll();
            node.inQueue=false;
            if (visited.contains(node.name)){
                continue;
            }

            visited.add(node.name);

            if(node.isFlag && !node.foundFlag){
                foundFlags.put(node, node.minDistance);
                node.foundFlag=true;
                visited = new HashSet<>();
                distanceQueue = new PriorityQueue<>();
                foundFlags.forEach((key,value)->{
                    key.setMinDistance(0);
                    distanceQueue.add(key);
                });
                continue;
            }

            for(int i = 0; i<node.adjacents.size(); i++){
                Node adjNode = node.adjacents.get(i);
                int value = node.distances.get(i);
                if (adjNode.minDistance >= value + node.minDistance){
                    adjNode.setMinDistance(value + node.minDistance);
                    if(adjNode.inQueue){
                        distanceQueue.remove(adjNode);
                    }
                    distanceQueue.add(adjNode);
                    adjNode.inQueue = true;
                }
            }
        }
        int total = 0;
        for (int i : foundFlags.values()){
            total += i;
        }
        System.out.println(total);
    }
}
