import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class Project1 {
    public static void main(String[] args) throws FileNotFoundException {

        File outFile = new File(args[1]);
        PrintStream printStream = new PrintStream(outFile);
        System.setOut(printStream);

        File file = new File(args[0]);
        Scanner scanner = new Scanner(file);

        FactoryImpl factory = new FactoryImpl();

        while (scanner.hasNext()){
            String[] line = scanner.nextLine().split(" ");
            //System.out.println(Arrays.toString(line));
            String command = line[0];
            switch (command) {
                case "AF": {
                    int id = Integer.parseInt(line[1]);
                    int value = Integer.parseInt(line[2]);
                    factory.addFirst(new Product(id, value));
                    break;
                }
                case "AL": {
                    int id = Integer.parseInt(line[1]);
                    int value = Integer.parseInt(line[2]);
                    factory.addLast(new Product(id, value));
                    break;
                }
                case "A": {
                    int idx = Integer.parseInt(line[1]);
                    int id = Integer.parseInt(line[2]);
                    int value = Integer.parseInt(line[3]);
                    try {
                        factory.add(idx, new Product(id, value));
                    } catch (IndexOutOfBoundsException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                }
                case ("RF"):
                    try {
                        Product removedProduct = factory.removeFirst();
                        System.out.println(removedProduct.toString());
                    } catch (NoSuchElementException e) {
                        System.out.println(e.getMessage());
                    }

                    break;
                case "RL":
                    try {
                        Product removedProduct = factory.removeLast();
                        System.out.println(removedProduct.toString());
                    } catch (NoSuchElementException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case "RI": {
                    int idx = Integer.parseInt(line[1]);
                    try {
                        Product removedProduct = factory.removeIndex(idx);
                        System.out.println(removedProduct.toString());
                    } catch (IndexOutOfBoundsException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                }
                case "RP": {
                    int value = Integer.parseInt(line[1]);
                    try {
                        Product removedProduct = factory.removeProduct(value);
                        System.out.println(removedProduct.toString());
                    } catch (NoSuchElementException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                }
                case "F": {
                    int id = Integer.parseInt(line[1]);
                    try {
                        Product found = factory.find(id);
                        System.out.println(found.toString());
                    } catch (NoSuchElementException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                }
                case "G": {
                    int idx = Integer.parseInt(line[1]);
                    try {
                        Product product = factory.get(idx);
                        System.out.println(product.toString());
                    } catch (IndexOutOfBoundsException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                }
                case "U": {
                    int idx = Integer.parseInt(line[1]);
                    int value = Integer.parseInt(line[2]);
                    try {
                        Product product = factory.update(idx, value);
                        System.out.println(product.toString());
                    } catch (NoSuchElementException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                }
                case "FD":
                    int count = factory.filterDuplicates();
                    System.out.println(count);
                    break;
                case "R":
                    factory.reverse();
                    factory.print();
                    break;
                case "P":
                    factory.print();
                    break;
            }
        }
    }
}