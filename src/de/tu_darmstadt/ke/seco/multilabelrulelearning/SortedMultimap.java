package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import java.util.*;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.MulticlassCovering.Closure;

public class SortedMultimap {

    private class KeyValuePair {

        public double key;
        public Closure value;

        public KeyValuePair(double key, Closure value) {
            this.key = key;
            this.value = value;
        }

    }

    private ArrayList<KeyValuePair> map = new ArrayList<>();

    public SortedMultimap() {

    }

    public void put(double key, Closure value) {
        KeyValuePair keyValuePair = new KeyValuePair(key, value);
        insertSorted(keyValuePair);
    }

    private void insertSorted(KeyValuePair keyValuePair) {
        if (map.isEmpty())
            map.add(keyValuePair);
        for (int i = 0; i < map.size(); i++) {
            if (i == map.size() - 1) {
                map.add(i, keyValuePair);
                break;
            }
            KeyValuePair listElement = map.get(i);
            if (keyValuePair.key > listElement.key) {
                map.add(i, keyValuePair);
                break;
            } else if (keyValuePair.key == listElement.key) {
                int compare = keyValuePair.value.compareTo(listElement.value); // -1 if listElement better
                if (compare == 1) { // i.e. keyValuePair is better
                    map.add(i, keyValuePair);
                    break;
                }
            }
        }
    }

    /**
     * Removes a specific key value pair from the map.
     * @param key
     * @param value
     */
    public void remove(double key, Closure value) {
        for (int i = 0; i < map.size(); i++) {
            KeyValuePair keyValuePair = map.get(i);
            if (keyValuePair.key == key && keyValuePair.value.equals(value)) {
                map.remove(i);
                break;
            }
        }
    }

    /**
     * Returns the value of the first matching key.
     * @param key
     * @return
     */
    public Closure get(double key) {
        for (int i = 0; i < map.size(); i++) {
            KeyValuePair keyValuePair = map.get(i);
            if (keyValuePair.key == key)
                return keyValuePair.value;
        }
        return null;
    }

    /**
     * Returns the key value of the highest ranking key value pair.
     * @return
     */
    public double firstKey() {
        if (isEmpty())
            try {
                throw new Exception("There are no elements in SortedMultimap.");
            } catch (Exception e) {
                e.printStackTrace();
            }
        return map.get(0).key;
    }

    public boolean isEmpty() {
        return (map.size() == 0);
    }

    public List<java.lang.Double> keys() {
        ArrayList<java.lang.Double> keys = new ArrayList<>();
        for (KeyValuePair keyValuePair : map)
            keys.add(keyValuePair.key);
        return keys;
    }

    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("[");
        for (KeyValuePair keyValuePair : map)
            stringBuilder.append("(" + keyValuePair.key + "," + keyValuePair.value + ")");
        stringBuilder.append("]");
        return stringBuilder.toString();
    }

}
