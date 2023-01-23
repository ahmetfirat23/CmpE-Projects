import java.io.FileWriter;
import java.io.IOException;

public class BST implements Tree {

    private Node root;
    private final FileWriter writer;

    public BST(String IP, FileWriter writer) {
        root = new Node(IP, null);
        this.writer = writer;
    }

    /**
     * Instruction: ADDNODE <IP_ADDRESS>
     * During addition of a new node, its place is found using binary search. All the nodes on the path from
     * root node to parent node of the newly added node shall add the event to their log messages, and others
     * should not change their log files. Log message structure for adding a new node is:
     * <Logging_node_IP>: New Node being added: <IP_Address>
     *
     * @param IP IP of new node
     */
    @Override
    public void addNode(String IP) throws IOException {
        root = recAdd(root, IP, null);
    }

    Node recAdd(Node node, String IP, Node parent) throws IOException {
        if (node==null){
            return new Node(IP, parent);
        }
        writer.write(node.IP + ": New node being added with IP:"+IP+'\n');
        if(node.compareTo(IP)<0){
            node.right = recAdd(node.right,IP, node);
        }
        else if(node.compareTo(IP)>0){
            node.left = recAdd(node.left, IP, node);
        }
        return node;
    }

    /**
     * Instruction: DELETE <IP_ADDRESS>
     * When a node is deleted, there are 2 cases: The node is a leaf node and deleted immediately or the node
     * is not a leaf node and upon deletion, a leaf node takes its place. For this project, you are expected to use
     * the smallest node from right subtree to replace the deleted node.
     * In terms of logging, logging of the delete operation is performed by the parent node of the deleted node.
     * For the 2 cases, there are 2 possible log lines:
     * • When a leaf node is deleted, its parent logs:
     * <Logging_node_IP>: Leaf Node Deleted: <Deleted Node IP>
     * • When a non-leaf node is deleted:
     * <Logging_node_IP>: Non-Leaf Node deleted; removed: <Deleted Node IP> replaced: <Replacing Node IP>
     * convention for deleting a node with single child
     * <Parent_IP>+ ": Node with single child Deleted: " + <Deleted_node_IP>
     * Notice that only the parent of actually deleted node’s parent performs a logging operation, so when you
     * swap the intermediate node with a leaf node to remove the leaf node, the parent of the leaf node should not
     * log anything.
     *
     * @param IP IP of the node to be deleted
     */
    @Override
    public void deleteNode(String IP) throws IOException {
        if (root.IP.equals(IP)){
            return;
        }
        root = recDelete(root, IP, null);
    }

    Node recDelete(Node node, String IP, Node parent) throws IOException{
        if (node==null){
            System.out.println("I was going to delete " + IP + " but i couldn't find anything after checking node " + parent.IP);
            return null;
        }
        else if(node.compareTo(IP)<0){
            node.right = recDelete(node.right, IP, node);
        }
        else if(node.compareTo(IP)>0){
            node.left = recDelete(node.left, IP, node);
        }
        else{ //Target found
            if(node.left==null && node.right==null){
                writer.write(parent.IP + ": Leaf Node Deleted: " + IP+'\n');
                return null;
            }
            else if(node.left==null){
                writer.write(parent.IP+ ": Node with single child Deleted: " + IP+'\n');
                Node child = node.right;
                node.IP = child.IP;
                node.right = child.right;
                node.left = child.left;
            }
            else if(node.right==null){
                writer.write(parent.IP+ ": Node with single child Deleted: " + IP+'\n');
                Node child = node.left;
                node.IP = child.IP;
                node.left = child.left;
                node.right = child.right;
            }
            else{
                Node min = node.right;
                Node pom = node;
                while(min.left!=null){
                    if(min.left.left==null){
                        pom = min;
                    }
                    min = min.left;
                }
                writer.write(parent.IP + ": Non Leaf Node Deleted; removed: "+IP+" replaced: "+ min.IP+'\n');
                if(!pom.IP.equals(node.IP)){
                    pom.left = min.right;
                }
                else{
                    node.right = min.right;
                }
                node.IP = min.IP;
                if (min.right!=null){
                    min.right.parent = pom;
                }
            }
        }
        return node;
    }

    /**
     * Instruction: SEND <SENDER_IP_ADDRESS> <RECEIVER_IP_ADDRESS>
     * Tree is a connected graph, meaning there is a path between each node in the topology, so it is possible to
     * send a message between each node. However, a node can transmit a message only to its child nodes or the
     * parent node. Thus, to arrive to the receiver, the message must jump between nodes. These message hops
     * are logged in the nodes as:
     * <Logging_node_IP>: Transmission from: <NODE_IP_THE_MESSAGE_COMES_FROM> receiver: <RECEIVER_IP>
     * sender: <SENDER_IP>
     * For example, consider the tree structure from the introduction. A message sent from 172.46.7.10 to
     * 172.200.56.78 follows the following path:
     * 172.46.7.10 −→ 172.50.4.70 −→ 172.100.0.1 −→ 172.200.56.78
     * This transmission is logged in 3 different ways in different type of nodes:
     * • Sender(172.46.7.10):
     * <Logging_node_IP>: Sending message to: 172.200.56.78
     * • Receiver(172.200.56.78):
     * <Logging_node_IP>: Received message from: 172.46.7.10
     * • Intermediate node(172.100.0.1):
     * <Logging_node_IP>: Transmission from:172.50.4.70 receiver: 172.200.56.78 sender: 172.46.7.10
     * One another thing to keep in mind is that, we want you to commute the message in the shortest path,
     * so a message should not go over the same node 2 times. (Pro tip: throw an exception when this happens for
     * easy debugging)
     *
     * @param senderIP IP of the node sends transmission
     * @param receiverIP IP of the node target of transmission
     */
    @Override
    public void sendMessage(String senderIP, String receiverIP) throws IOException {
        writer.write(senderIP + ": Sending message to: "+ receiverIP+'\n');
        Node common = findCommonAncestor(senderIP, receiverIP);
        Node sender = root;
        while(sender!=null){
            if(sender.compareTo(senderIP) < 0 ){
                sender = sender.right;
            }
            else if(sender.compareTo(senderIP) > 0){
                sender = sender.left;
            }
            else{
                break;
            }
        }

        while (sender!=common){
            if(!sender.parent.IP.equals(receiverIP)){
                writer.write(sender.parent.IP+": Transmission from: "+sender.IP+" receiver: "+receiverIP+" sender:"+senderIP+'\n');
            }
            sender = sender.parent;
        }
        while (!sender.IP.equals(receiverIP)){
            if (sender==null){
                break;
            }
            else if(sender.compareTo(receiverIP)<0){
                sender = sender.right;
            }
            else{
                sender = sender.left;
            }
            if(!sender.IP.equals(receiverIP)){
                writer.write(sender.IP+": Transmission from: "+sender.parent.IP+" receiver: "+receiverIP+" sender:"+senderIP+'\n');
            }
        }
        writer.write(receiverIP+": Received message from: "+senderIP+'\n');
    }

    // Finds lowest common ancestor to find the root of smallest subtree contains both elements
    Node findCommonAncestor(String IP1, String IP2){
        Node iter = root;
        while(iter!=null){
            if(iter.compareTo(IP1) < 0 && iter.compareTo(IP2) < 0){
                iter = iter.right;
            }
            else if(iter.compareTo(IP1) > 0 && iter.compareTo(IP2) > 0){
                iter = iter.left;
            }
            else{
                break;
            }
        }
        return iter;
    }
}

