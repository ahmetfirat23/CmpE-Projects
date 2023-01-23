import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {
        File inFile = new File(args[0]);
        Scanner scanner = new Scanner(inFile);

        FileWriter bstWriter = new FileWriter(args[1]+"_BST.txt", false);
        FileWriter avlWriter = new FileWriter(args[1]+"_AVL.txt", false);

        String IP = scanner.nextLine();
        BST bst = new BST(IP, bstWriter);
        AVL avl = new AVL(IP, avlWriter);
        while (scanner.hasNext()){
            String[] line = scanner.nextLine().split(" ");
            String instruc = line[0];
            if (instruc.equals("ADDNODE")){
                bst.addNode(line[1]);
                avl.addNode(line[1]);
            }
            else if(instruc.equals("DELETE")){
                bst.deleteNode(line[1]);
                avl.deleteNode(line[1]);
            }
            else if(instruc.equals("SEND")){
                bst.sendMessage(line[1],line[2]);
                avl.sendMessage(line[1], line[2]);
            }
        }
        bstWriter.close();
        avlWriter.close();
    }
}
