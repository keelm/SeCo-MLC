/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * Instances.java Copyright (C) 1999 Eibe Frank Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Adopted to the SeCo-Project by Viktor Seifert November 2006
 */

package de.tu_darmstadt.ke.seco.models;

import de.tu_darmstadt.ke.seco.utils.Logger;
import weka.core.Instance;
import weka.core.UnassignedClassException;
import weka.core.UnassignedDatasetException;

import java.io.IOException;
import java.io.Reader;
import java.io.Serializable;
import java.io.StringReader;
import java.lang.reflect.Field;
import java.util.*;

public class Instances extends weka.core.Instances implements Serializable {

    public Instances(Instances dataset) {
        super(dataset);
    }

    public Instances(Instances dataset, int capacity) {
        super(dataset, capacity);
    }

    public Instances(Reader reader) throws IOException {
        super(reader);
    }

    public Instances(final Instances source, final int first, final int toCopy) {
        super(source, first, toCopy);
    }

    public Instances(String name, ArrayList<weka.core.Attribute> attInfo, final int capacity) {
        super(name, attInfo, capacity);
    }

    @Override
    public Instance instance(int index) {
        return m_Instances.get(index);
    }

    /**
     * Checks, whether the given examples contain positive ones according to classVal. If you want to use weighted covering, you had to change your RuleStop to check for your preferred stop of the weighted covering
     *
     * @param classValue The class value that is to be considered positive.
     * @return true, if there are positive examples, false otherwise.
     */
    public boolean containsPositive(final double classValue) {
        // final java.util.Enumeration<Instance> en = examples.enumerateInstances();

        for (final Instance instance : m_Instances)
            try {
                if ((instance.classValue() == classValue))
                    return true;
            } catch (final UnassignedClassException ex) {
                System.err.println(ex.getMessage());
                ex.printStackTrace();
                System.exit(-1);
            } catch (final UnassignedDatasetException ex) {
                System.err.println(ex.getMessage());
                ex.printStackTrace();
                System.exit(-1);
            }
        Logger.debug("containsPositive() returns false");
        return false;
    }

    /**
     * Returns an enumeration of all the attributes without the class attribute.
     *
     * @return enumeration of all the attributes.
     */
    public Enumeration<de.tu_darmstadt.ke.seco.models.Attribute> enumerateAttributesWithoutClass() {
        final Instances innerTemp = this;

        return new Enumeration<de.tu_darmstadt.ke.seco.models.Attribute>() {

            private int currentPosition = 0;

            @Override
            public boolean hasMoreElements() {
                if (currentPosition != innerTemp.classIndex())
                    return currentPosition < innerTemp.getNumberOfAttributes();
                else
                    // skip the current attribute if it is the class attribute
                    return currentPosition + 1 < innerTemp.getNumberOfAttributes();
            }

            @Override
            public de.tu_darmstadt.ke.seco.models.Attribute nextElement() {
                // skip the class attribute
                if (currentPosition == innerTemp.classIndex())
                    currentPosition++;

                de.tu_darmstadt.ke.seco.models.Attribute att = null;
                if (currentPosition < innerTemp.getNumberOfAttributes())
                    att = Attribute.toSeCoAttribute(innerTemp.attribute(currentPosition));

                currentPosition++;

                return att;
            }
        };

        // old implementation of this method
        // return m_Attributes.elements(m_ClassIndex);
    }

    public int getNumberOfAttributes() { // TODO by m.zopf: use everywhere numAttributes()
        return numAttributes();
    }

    public int countInstances(final double classValue) throws UnassignedClassException, UnassignedDatasetException {
        int count = 0;

        final Enumeration<Instance> en = enumerateInstances();
        while (en.hasMoreElements()) {
            final Instance inst = en.nextElement();

            if (inst.classValue() == classValue)
                count++;
        }

        return count;
    }

    /**
     * This will order the given examples in ascending frequency of their classes.
     *
     * @return The sorted examples.
     */
    public Instances orderClasses() throws Exception {

        Instances newResult = null;

        newResult = new Instances(new StringReader(this.toString()));
        newResult.setClassIndex(this.classIndex());
        final Attribute classAttribute = newResult.classAttribute();

        // analyze class frequency
        final TreeMap<Integer, Set<Double>> mapping = new TreeMap<Integer, Set<Double>>();

        // TODO by m.zopf: doesn't make sense with floating point class values, should be something with classIndex?
        for (double classValue = 0; classValue < classAttribute.numValues(); classValue++) {
            final int classCount = newResult.countInstances(classValue);
            Set<Double> classCountSet = mapping.get(classCount);
            if (classCountSet == null) {
                classCountSet = new TreeSet<Double>();
                classCountSet.add(classValue);
                mapping.put(classCount, classCountSet);
            } else
                mapping.get(classCount).add(classValue);
        }

        final Collection<Set<Double>> orderedClassSet = mapping.values();
        final ArrayList<String> orderedClassList = new ArrayList<String>();
        final ArrayList<Double> orderedClassValueList = new ArrayList<Double>();

        for (final Set<Double> classValSet : orderedClassSet)
            for (final Double classVal : classValSet) {
                orderedClassList.add(classAttribute.value(classVal.intValue()));
                orderedClassValueList.add(classVal);
            }

        for (int i = 0; i < classAttribute.numValues(); i++)
            classAttribute.setValue(i, orderedClassList.get(i));

        final double[] classValMap = new double[classAttribute.numValues()];
        for (int i = 0; i < classAttribute.numValues(); i++)
            classValMap[orderedClassValueList.get(i).intValue()] = i;

        for (final Instance instance : newResult.m_Instances)
            instance.setClassValue(classValMap[(int) instance.classValue()]);

        return newResult;
    }

    public ArrayList<Instance> toWekaInstances() {
        ArrayList<Instance> arrayList = new ArrayList<>();

        for (Instance instance : this)
            arrayList.add(instance);

        return arrayList;
    }

    public void setWeightsTo0() {
        for (final Instance instance : this)
            instance.setWeight(0);
    }

    public static Instances toSeCoInstances(weka.core.Instances wekaInstances) {
        Instances seCoInstances = new Instances("", new ArrayList<weka.core.Attribute>(), 0);

        try {
            Field m_RelationNameField = weka.core.Instances.class.getDeclaredField("m_RelationName");
            Field m_AttributesField = weka.core.Instances.class.getDeclaredField("m_Attributes");
            Field m_InstancesField = weka.core.Instances.class.getDeclaredField("m_Instances");
            Field m_ClassIndexField = weka.core.Instances.class.getDeclaredField("m_ClassIndex");
            Field m_LinesField = weka.core.Instances.class.getDeclaredField("m_Lines");

            m_RelationNameField.setAccessible(true);
            m_AttributesField.setAccessible(true);
            m_InstancesField.setAccessible(true);
            m_ClassIndexField.setAccessible(true);
            m_LinesField.setAccessible(true);

            m_RelationNameField.set(seCoInstances, m_RelationNameField.get(wekaInstances));
            m_AttributesField.set(seCoInstances, m_AttributesField.get(wekaInstances));
            m_InstancesField.set(seCoInstances, m_InstancesField.get(wekaInstances));
            m_ClassIndexField.set(seCoInstances, m_ClassIndexField.get(wekaInstances));
            m_LinesField.set(seCoInstances, m_LinesField.get(wekaInstances));
        } catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException e) {
            e.printStackTrace();
        }

        return seCoInstances;
    }

    @Override
    public Attribute classAttribute() throws UnassignedClassException {
        if (m_ClassIndex < 0)
            throw new UnassignedClassException("Class index is negative (not set)!");

        return Attribute.toSeCoAttribute(attribute(m_ClassIndex));
    }

    public Instances[] splitInstances(final double growingSetSize, final Random random) throws UnassignedClassException, UnassignedDatasetException {

        // stratify:
        final int folds = (int) Math.round(1 / (1 - growingSetSize));
        if (!classAttribute().isNominal()) // TODO by m.zopf: why this distinction?
            return new Instances[]{this};

        final Instances result = new Instances(this, 0);
        final Instances[] bagsByClasses = new Instances[numClasses()];
        for (int i = 0; i < bagsByClasses.length; i++)
            bagsByClasses[i] = new Instances(this, 0);

        // Sort by class
        for (int j = 0; j < numInstances(); j++) {
            final Instance datum = instance(j);
            bagsByClasses[(int) datum.classValue()].add(datum);
        }

        // bagsByClasses is sorted by the number of instances, very important for RIPPER

        final int n = bagsByClasses.length;
        final Instances[] temp = new Instances[1];
        for (int pass = 1; pass < n; pass++)
            // This next loop becomes shorter and shorter
            for (int i = 0; i < n - pass; i++)
                if (bagsByClasses[i].numInstances() > bagsByClasses[i + 1].numInstances()) {
                    // exchange elements
                    temp[0] = bagsByClasses[i];
                    bagsByClasses[i] = bagsByClasses[i + 1];
                    bagsByClasses[i + 1] = temp[0];
                }

        // Randomize each class
        for (final Instances bagsByClasse : bagsByClasses)
            bagsByClasse.randomize(random);

        for (int k = 0; k < folds; k++) {
            int offset = k, bag = 0;
            oneFold:
            while (true) {
                while (offset >= bagsByClasses[bag].numInstances()) {
                    offset -= bagsByClasses[bag].numInstances();
                    if (++bag >= bagsByClasses.length)// Next bag
                        break oneFold;
                }

                result.add(bagsByClasses[bag].instance(offset));
                offset += folds;
            }
        }

        // split:
        final Instances[] rt = new Instances[2];
        final int splits = (int) (result.numInstances() * growingSetSize);

        rt[0] = new Instances(result, 0, splits);
        rt[1] = new Instances(result, splits, result.numInstances() - splits);

        return rt;
    }

    /**
     * Returns all the distinct class values.
     *
     * @return a collection with all the distinct class values.
     */
    public Collection<Double> getDistinctClassValues(final boolean ascending, final boolean skipLast) throws UnassignedClassException, UnassignedDatasetException {

        final Set<Double> valuesSet = new HashSet<Double>();

        final Enumeration<Instance> en = this.enumerateInstances();
        while (en.hasMoreElements()) {
            final Instance inst = en.nextElement();
            valuesSet.add(inst.classValue());
        }
        TreeSet<Double> orderedSet;

        orderedSet = new TreeSet<Double>();
        orderedSet.addAll(valuesSet);

        // remove the highest value
        if (skipLast)
            orderedSet.pollLast();

        Logger.debug("Created distinct class values:");
        if (ascending)
            Logger.debug(orderedSet.toString());
        else
            Logger.debug(orderedSet.descendingSet().toString());

        if (ascending)
            return orderedSet;
        else
            return orderedSet.descendingSet();
    }

    /**
     * Returns the arrayList of instances
     *
     * @return
     */
    public ArrayList<Instance> getInstances() {
        return m_Instances;
    }

    public void addDirectly(Instance instance) {
        instance.setDataset(this);
        m_Instances.add(instance);
    }
}
