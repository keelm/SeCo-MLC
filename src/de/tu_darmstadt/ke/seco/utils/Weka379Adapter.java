package de.tu_darmstadt.ke.seco.utils;

import java.io.Serializable;

import de.tu_darmstadt.ke.seco.learners.core.SeCoClassifier;
import de.tu_darmstadt.ke.seco.learners.core.SeCoClassifierFactory;
import de.tu_darmstadt.ke.seco.models.*;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;

/**
 * Implements the adapter pattern for {@link SeCoClassifier} to be compatible with Weka 3.7.4.
 *
 * @author Markus Zopf
 * @version 1.0
 */
public class Weka379Adapter implements Classifier, Serializable {

    /**
     * The adaptee in the adapter pattern.
     */
    private SeCoClassifier seCoClassifier;

    private final SeCoAlgorithm seCoAlgorithm;

    boolean predictMissing = false;

    /**
     * Creates an instance of {@link Weka379Adapter}.
     *
     * @param seCoClassifier the {@link SeCoClassifier} that should be adapted
     */
    public Weka379Adapter(SeCoAlgorithm seCoAlgorithm) {
        if (seCoAlgorithm == null)
            throw new IllegalArgumentException("seCoAlgorithm must not be null.");

        this.seCoAlgorithm = seCoAlgorithm;
    }

    @Override
    public void buildClassifier(final weka.core.Instances data) throws Exception {
        Instances secoData = Instances.toSeCoInstances(data);
        seCoClassifier = SeCoClassifierFactory.buildSeCoClassifier(seCoAlgorithm, secoData);
        if (predictMissing) {
            final NominalCondition defHead = new NominalCondition(secoData.classAttribute(), Double.NaN);
            final SingleHeadRule def = new SingleHeadRule(seCoClassifier.getHeuristic(), defHead);
            SingleHeadRuleSet ruleSet = (SingleHeadRuleSet) seCoClassifier.getTheory();
            ruleSet.setDefaultRule(def);
//			System.out.println(seCoClassifier.getTheory().getDefaultPrediction());
        }

    }

    @Override
    public double classifyInstance(final Instance instance) throws Exception {
        return seCoClassifier.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        return seCoClassifier.distributionForInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO: implement
        System.err.println("getCapabilities");
        throw new NotImplementedException();
    }

    public SeCoClassifier getSeCoClassifier() {
        return seCoClassifier;
    }

    @Override
    public String toString() {
        return seCoClassifier.toString();
    }

    public void setPredictMissing(boolean predictMissing) {
        this.predictMissing = predictMissing;
    }

    public boolean getPredictMissing() {
        return predictMissing;
    }


}
