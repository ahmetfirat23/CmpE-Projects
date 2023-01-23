public class Node {
    public String IP;
    public Node left;
    public Node right;
    public Node parent;
    public int height;

    public Node(String IP, Node parent){
        this.IP = IP;
        this.parent = parent;
        height = 0;
    }

    public int compareTo(String IP) {
        return this.IP.compareTo(IP);
    }
}
