package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.learners.core.SeCoClassifier;
import de.tu_darmstadt.ke.seco.learners.core.SeCoClassifierFactory;
import de.tu_darmstadt.ke.seco.models.Instances;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.TechnicalInformation;

import java.io.Serializable;

/**
 * Implements the adapter pattern for {@link SeCoClassifier} to be compatible with Weka 3.7.4.
 *
 * @author Markus Zopf
 * @version 1.0
 */
public class Weka379AdapterPrepending extends MultiLabelLearnerBase implements Serializable {

    /**
     * The adaptee in the adapter pattern.
     */
    private PrependingSecoClassifier seCoClassifier;

    private final SeCoAlgorithm seCoAlgorithm;

    /**
     * Creates an instance of {@link Weka379AdapterPrepending}.
     *
     * @param seCoAlgorithm the {@link SeCoAlgorithm} that should be adapted
     */
    public Weka379AdapterPrepending(SeCoAlgorithm seCoAlgorithm) {
        if (seCoAlgorithm == null)
            throw new IllegalArgumentException("seCoAlgorithm must not be null.");
        this.seCoAlgorithm = seCoAlgorithm;
    }

    /**
     * Creates an instance of {@link Weka379AdapterPrepending}.
     *
     * @param seCoAlgorithm the {@link SeCoAlgorithm} that should be adapted
     */
    public Weka379AdapterPrepending(final SeCoAlgorithm seCoAlgorithm, final double remainingInstancesPercentage,
                                    final boolean readdAllCovered, final double skipThresholdPercentage,
                                    final boolean predictZeroRules, final boolean useMultilabelHeads,
                                    final String evaluationStrategy,
                                    final String averagingStrategy,
                                    final boolean useRelaxedPruning,
                                    final boolean useBoostedHeuristicForRules,
                                    final String boostFunction,
                                    final double label,
                                    final double boostAtLabel,
                                    final double curvature,
                                    final int pruningDepth) {
        if (seCoAlgorithm == null)
            throw new IllegalArgumentException("seCoAlgorithm must not be null.");

        this.seCoAlgorithm = seCoAlgorithm;
        seCoAlgorithm.setUncoveredInstancesPercentage(remainingInstancesPercentage);
        seCoAlgorithm.setPredictZero(predictZeroRules);
        seCoAlgorithm.setSkipThresholdPercentage(skipThresholdPercentage);
        seCoAlgorithm.setReaddAllCovered(readdAllCovered);
        seCoAlgorithm.setUseMultilabelHeads(useMultilabelHeads);
        seCoAlgorithm.setEvaluationStrategy(evaluationStrategy);
        seCoAlgorithm.setAveragingStrategy(averagingStrategy);
        seCoAlgorithm.setRelaxedPruning(useRelaxedPruning);
        seCoAlgorithm.setBoostedHeuristicForRules(useBoostedHeuristicForRules);
        seCoAlgorithm.setBoostFunction(boostFunction);
        seCoAlgorithm.setLabel(label);
        seCoAlgorithm.setBoostAtLabel(boostAtLabel);
        seCoAlgorithm.setCurvature(curvature);
        seCoAlgorithm.setPruningDepth(pruningDepth);

    }

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

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        seCoClassifier = SeCoClassifierFactory.buildSeCoClassifierPrepending(seCoAlgorithm,
                Instances.toSeCoInstances(trainingSet.getDataSet()), trainingSet.getLabelIndices());
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        return seCoClassifier.makePrediction(instance);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        // TODO Auto-generated method stub
        return null;
    }

}