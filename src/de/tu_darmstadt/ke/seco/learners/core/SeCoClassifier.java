/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * AbstractSeco.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Matthias Thiel Created on 29.10.2004, 17:30 Modified by Frederik Janssen, Jiawei Du, David Schuld, Raad Bahmani
 */

package de.tu_darmstadt.ke.seco.learners.core;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Attribute;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.RuleSet;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;
import de.tu_darmstadt.ke.seco.stats.ConfusionMatrix;
import de.tu_darmstadt.ke.seco.utils.Logger;
import weka.core.Instance;
import weka.core.UnassignedClassException;
import weka.core.Utils;

import java.io.*;
import java.util.Enumeration;
import java.util.HashMap;

/**
 * The AbstractSeco class is the abstract base class for classifiers in the SeCo framework.
 *
 * @author Knowledge Engineering Group
 */
public class SeCoClassifier implements Serializable {

    /**
     * default serial UID
     */
    private static final long serialVersionUID = 1L;

    /**
     * Needed for stopping recursion in setClassVal.
     */
    protected boolean m_settingUpComponents = false;

    /**
     * The final theory that is built with this algorithm.
     */
    protected RuleSet<?> m_theory;

    /**
     * used for classifying purposes
     */
    public final HashMap<Double, Double> orderedClassMapping = new HashMap<Double, Double>();

    protected String m_id = "";

    private SeCoAlgorithm seCoAlgorithm;

    /**
     * Creates a new instance of AbstractSeCo
     */
    public SeCoClassifier() {
        m_theory = new SingleHeadRuleSet();
    }

    /**
     * Getter for learned theory.
     */
    public RuleSet<?> getTheory() {
        return m_theory;
    }

    public void setTheory(final RuleSet<?> theory) {
        this.m_theory = theory;
    }

    /**
     * Getter for the used heuristic
     */
    public Heuristic getHeuristic() {
        return seCoAlgorithm.getHeuristic();
    }

    /**
     * Classifies the passed instance, i.e. return the class value if the instance is covered by the rule, or the missing value if it is not covered by the rule. Currently we assume classification as a decision list, i.e., the prediction of the first rule that doesn't predict the missing value is returned. Eventually, this should probably be a parameter or maybe even a separate subclass.
     *
     * @param instance The instance.
     * @return The class of the instance or a missing value.
     */
    public double classifyInstance(final Instance instance) throws Exception {
        if (!(m_theory instanceof SingleHeadRuleSet))
            throw new IllegalStateException("Only single head rules supported");
        Logger.debug("classifyInstance(" + instance.toString() + ")");
        double tmp1 = ((SingleHeadRuleSet) m_theory).classifyInstance(instance);
        if (Double.isNaN(tmp1))
//			Missing class output hack
            return tmp1;
//	Missing class output hack
//			if (seCoAlgorithm.getRandom().nextBoolean())
//				tmp1 = 0.0;
//			else
//				tmp1 = 1.0;
        final double tmp2 = orderedClassMapping.get(tmp1);
        return tmp2;
    }

    @Override
    public String toString() {
        final StringBuilder stringBuilder = new StringBuilder();

        stringBuilder.append(getSeCoAlgorithm().toString()).append("\n");

        if (m_theory != null)
            stringBuilder.append(m_theory);

        stringBuilder.append("\nbuilding time........: " + learningTime + " ms.");

        return stringBuilder.toString();
    }

    /**
     * Returns the time required for learning a model in seconds
     *
     * @return the time interval in seconds
     */
    public double getLearningTime() {
        return learningTime / 1000f;
    }

    public void setLearningTime(final long learningTime) {
        this.learningTime = learningTime;
    }

    /**
     * the time taken for learning
     */
    private double learningTime;

    /**
     * This method calculates the confusion-matrix for a data-set
     *
     * @param data The dataset.
     * @return the confusion matrix
     */
    public ConfusionMatrix evaluateClassifier(final Instances data) {

        ConfusionMatrix confMatrix = null;
        int count = 0;

        try {
            count = data.numClasses();

        } catch (final UnassignedClassException e) {
            Logger.error("Undefined index", e);
            return confMatrix;
        }

        confMatrix = new ConfusionMatrix(count);

        for (int actualClassValue = 0; actualClassValue < count; actualClassValue++)
            for (int predictedClassValue = 0; predictedClassValue < count; predictedClassValue++)
                confMatrix.addToElement(actualClassValue, predictedClassValue, getNumberOfClassifiedAs(actualClassValue, predictedClassValue, data));

        return confMatrix;
    }

    /**
     * This method calculates the number of classes, identified as a particular class
     *
     * @param actualClass    The actual class.
     * @param predictedClass The class predicted by the classifier.
     * @param data           The dataset.
     * @return the count of classes
     */
    private int getNumberOfClassifiedAs(final double actualClass, final double predictedClass, final Instances data) {

        int count = 0;

        final Enumeration<Instance> en = data.enumerateInstances();
        while (en.hasMoreElements()) {
            final Instance i = en.nextElement();

            try {
                if (i.classValue() == actualClass && this.classifyInstanceFromClassifier(i) == predictedClass)
                    count++;
            } catch (final Exception ex) {
                Logger.error("Could not classify instance:");
                Logger.error(i.toString());
                Logger.error("Reason:");
                Logger.error(ex.toString());
                Logger.error("Stack trace:", ex);

            }
        }

        return count;
    }

    /**
     * Classifies the given test instance. The instance has to belong to a dataset when it's being classified. Note that a classifier MUST implement either this or distributionForInstance().
     *
     * @param instance The instance to be classified.
     * @return the predicted most likely class for the instance or Utils.missingValue() if no prediction is made
     * @throws Exception if an error occurred during the prediction
     */
    private double classifyInstanceFromClassifier(final Instance instance) throws Exception {

        final double[] dist = distributionForInstance(instance);
        if (dist == null)
            throw new Exception("Null distribution predicted");
        switch (instance.classAttribute().type()) {
            case Attribute.NOMINAL:
                double max = 0;
                int maxIndex = 0;

                for (int i = 0; i < dist.length; i++)
                    if (dist[i] > max) {
                        maxIndex = i;
                        max = dist[i];
                    }
                if (max > 0)
                    return maxIndex;
                else
                    return Utils.missingValue();
            case Attribute.NUMERIC:
                return dist[0];
            default:
                return Utils.missingValue();
        }
    }

    /**
     * Predicts the class memberships for a given instance. If an instance is unclassified, the returned array elements must be all zero. If the class is numeric, the array must consist of only one element, which contains the predicted value. Note that a classifier MUST implement either this or classifyInstance().
     *
     * @param instance The instance to be classified.
     * @return an array containing the estimated membership probabilities of the test instance in each class or the numeric prediction
     * @throws Exception if distribution could not be computed successfully
     */
    public double[] distributionForInstance(final Instance instance) throws Exception {

        final double[] dist = new double[instance.numClasses()];
        switch (instance.classAttribute().type()) {
            case Attribute.NOMINAL:
                final double classification = classifyInstance(instance);
                if (Utils.isMissingValue(classification))
                    return dist;
                else
                    dist[(int) classification] = 1.0;
                return dist;
            case Attribute.NUMERIC:
                dist[0] = classifyInstance(instance);
                return dist;
            default:
                return dist;
        }
    }

    /**
     * Creates a new instance of ConfigurableSeco
     *
     * @param id Optional identifier for this Classifier.
     */
    public SeCoClassifier(final String id) {
        m_id = id;
        if (m_id == null)
            m_id = "";
    }

    @Override
    public SeCoClassifier clone() {

        SeCoClassifier retVal = null;
        final String separator = System.getProperty("file.separator");
        final String filename = System.getProperty("user.dir") + separator + "tmp.obj";

        try {
            final ObjectOutputStream o = new ObjectOutputStream(new FileOutputStream(filename));
            o.writeObject(this);
            o.close();

            final ObjectInputStream st = new ObjectInputStream(new FileInputStream(filename));
            retVal = (SeCoClassifier) st.readObject();
            st.close();
        } catch (final Exception ex) {
            ex.printStackTrace();
        }

        return retVal;
    }

    public SeCoAlgorithm getSeCoAlgorithm() {
        return seCoAlgorithm;
    }

    public void setSeCoAlgorithm(final SeCoAlgorithm seCoAlgorithm) {
        this.seCoAlgorithm = seCoAlgorithm;
    }

}
