/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * SplittedInstances.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by David Schuld
 */
package de.tu_darmstadt.ke.seco.models;

import weka.core.Instance;
import weka.core.UnassignedClassException;
import weka.core.UnassignedDatasetException;

/**
 * Class for representing a data set that is divided into a growing set and a pruning set, for learning algorithms that recognize this.
 *
 * @author Knowledge Engineering Group
 */
public class SplittedInstances extends Instances {

    /**
     *
     */
    private static final long serialVersionUID = 1L;

    // The growing set
    private Instances growingSet;

    // The pruning set
    private Instances pruningSet;

    public SplittedInstances(final Instances dataset) {
        super(dataset);
    }

    /**
     * Splits the dataset into a growing and a pruning set.
     *
     * @param growingSetSize The percentage of the instances that will be contained in the growing set. The rest of the dataset is in the pruning set.
     */
    public void splitInstances(final double growingSetSize) {

        // final double growPercent = (growingSetSize - 1) / (growingSetSize);
        final Instances[] rt = new Instances[2];
        final int splits = (int) (this.numInstances() * growingSetSize);

        rt[0] = new Instances(this, 0, splits);
        rt[1] = new Instances(this, splits, this.numInstances() - splits);

        growingSet = rt[0];
        pruningSet = rt[1];

        // return rt;
    }

    /**
     * Stratify the given data into the given number of bags based on the class values. It differs from the <code>Instances.stratify(int fold)</code> that before stratification it sorts the instances according to the class order in the header file. It assumes no missing values in the class.
     *
     * @param data  the given data
     * @param folds the given number of folds
     * @param rand  the random object used to randomize the instances
     * @return the stratified instances
     * @throws UnassignedClassException
     * @throws UnassignedDatasetException
     */
    public static final Instances stratify(final Instances data, final int folds, final Random rand) throws UnassignedClassException, UnassignedDatasetException {
        if (!data.classAttribute().isNominal())
            return data;

        final Instances result = new Instances(data, 0);
        final Instances[] bagsByClasses = new Instances[data.numClasses()];

        for (int i = 0; i < bagsByClasses.length; i++)
            bagsByClasses[i] = new Instances(data, 0);

        // Sort by class
        for (int j = 0; j < data.numInstances(); j++) {
            final Instance datum = data.instance(j);
            bagsByClasses[(int) datum.classValue()].add(datum);
        }

        // Randomize each class
        for (final Instances bagsByClasse : bagsByClasses)
            bagsByClasse.randomize(rand);

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

        return result;
    }

    /**
     * Returns the growing set.
     *
     * @return The instances contained in the growing set.
     */
    public Instances getGrowingSet() {
        return growingSet;
    }

    /**
     * Returns the pruning set.
     *
     * @return The instances contained in the pruning set.
     */
    public Instances getPruningSet() {
        return pruningSet;
    }

}
