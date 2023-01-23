public class AirportHashMap {
    ATC[] map;

    public AirportHashMap(){
        map = new ATC[1000];
    }

    private int hash(String str){
        char[] str_arr = str.toCharArray();
        int hashValue = 0;
        for (int i = str_arr.length - 1; i >= 0; i--){
            hashValue = (int) str_arr[i] + 31 * hashValue;
        }
        return hashValue % 1000;
    }

    public void add(ATC atc){
        String str = atc.name;
        int hashValue = hash(str);
        while(map[hashValue]!=null){
            hashValue = (hashValue+1)%1000;
        }
        map[hashValue] = atc;
    }

    public ATC find(String str){
        int hashValue = hash(str);
        while(map[hashValue]!=null && !map[hashValue].name.equals(str)){
            hashValue = (hashValue+1)%1000;
        }
        if (map[hashValue]==null){
            System.out.println("Can't be found" + str + " " + hashValue);
            return null;
        }
        return map[hashValue];
    }

    public String listNames(){
        String output = "";
        for (int i = 0; i<1000; i++){
            if (map[i]==null){
                continue;
            }
            output += " " + map[i].name + String.format("%03d", i);;
        }
        return output;
    }
}
