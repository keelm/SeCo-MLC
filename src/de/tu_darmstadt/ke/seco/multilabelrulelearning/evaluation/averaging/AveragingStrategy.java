package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Condition;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.DenseInstanceWrapper;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.SparseInstanceWrapper;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import weka.core.Instance;
import weka.core.Utils;

import java.util.Collection;
import java.util.Iterator;

public abstract class AveragingStrategy {

    public static final String MICRO_AVERAGING = "micro-averaging";

    public static final String LABEL_BASED_AVERAGING = "label-based-averaging";

    public static final String EXAMPLE_BASED_AVERAGING = "example-based-averaging";

    public static final String MACRO_AVERAGING = "macro-averaging";

    private double getLabelValue(final Instance instance, final int labelIndex) {
        Instance wrappedInstance =
                instance instanceof DenseInstanceWrapper ? ((DenseInstanceWrapper) instance).getWrappedInstance() :
                        ((SparseInstanceWrapper) instance).getWrappedInstance();
        return wrappedInstance.value(labelIndex);
    }

    protected final Iterable<Integer> instancesIterable(final Instances instances) {
        return () -> new Iterator<Integer>() {

            private int i = 0;

            @Override
            public boolean hasNext() {
                return i < instances.numInstances();
            }

            @Override
            public Integer next() {
                return i++;
            }

        };
    }

    protected final boolean areAllLabelsAlreadyPredicted(final Instance instance, final Head head) {
        for (Condition labelAttribute : head) {
            if (Utils.isMissingValue(instance.value(labelAttribute.getAttr().index()))) {
                return false;
            }
        }

        return true;
    }

    /**
     * Aggregations for Prepending. Changed semantics of TP, FP, FN and TN.
     * Only works for rule-dependent evaluation.
     */
    protected final void aggregatePrepending(final boolean covers, final Head head, final Instance instance, final int labelIndex,
                                final TwoClassConfusionMatrix confusionMatrix, final TwoClassConfusionMatrix stats) {
        double trueValue = getLabelValue(instance, labelIndex); // true label value (WrappedInstance)
        boolean isMissing = instance.isMissing(labelIndex);
        double setValue = isMissing ? 0 : instance.value(labelIndex);

        Condition labelAttribute = head.getCondition(labelIndex); // get the attribute of the head
        double predictedValue = labelAttribute.getValue(); // value of the attribute in the head

        if (covers) { // the rule covers the instance

            // rule does not correct the value as it is already set correctly
            if (setValue == trueValue && trueValue == predictedValue) {
                return;
            }

            if (trueValue == 0.0) {
                if (predictedValue == 0.0 && setValue == 1.0) {
                    confusionMatrix.addTruePositives(instance.weight());
                    return;
                } else if (predictedValue == 1.0 && setValue == 0.0) {
                    confusionMatrix.addFalsePositives(instance.weight());
                    return;
                } else if (predictedValue == 1.0 && setValue == 1.0) {
                    confusionMatrix.addFalseNegatives(instance.weight());
                    return;
                }
            }

            if (trueValue == 1.0) {
                if (predictedValue == 0.0 && setValue == 0.0) {
                    confusionMatrix.addFalseNegatives(instance.weight());
                    return;
                } else if (predictedValue == 0.0 && setValue == 1.0) {
                    confusionMatrix.addFalsePositives(instance.weight());
                    return;
                } else if (predictedValue == 1.0 && setValue == 0.0) {
                    confusionMatrix.addTruePositives(instance.weight());
                    return;
                }
            }

        } else {
            if (trueValue != setValue) // && predictedValue == trueValue
                confusionMatrix.addFalseNegatives(instance.weight());
        }

    }

    /**
     * Prepending with only positive heads.
     */
    final void aggregatePrependingPositiveHeads(final boolean covers, final Head head, final Instance instance, final int labelIndex,
                         final TwoClassConfusionMatrix confusionMatrix, final TwoClassConfusionMatrix stats) {
        double trueLabelValue = getLabelValue(instance, labelIndex); // true label value (WrappedInstance)

        boolean isMissing = instance.isMissing(labelIndex);

        // nur wenn korrekt
        if (!isMissing)
            return;

        if (covers) { // the rule covers the instance
            Condition labelAttribute = head.getCondition(labelIndex); // get the attribute of the head
            double labelValue = labelAttribute.getValue(); // value of the attribute in the head

            if (labelValue == trueLabelValue) { // then we correct the value
                confusionMatrix.addTruePositives(instance.weight());
            } else { // we don't...
                confusionMatrix.addFalsePositives(instance.weight());
            }

        } else {
            if (trueLabelValue == 1.0) {
                confusionMatrix.addFalseNegatives(instance.weight());
            }
        }

    }

    protected final void aggregate(final boolean covers, final Head head, final Instance instance, final int labelIndex,
                                   final TwoClassConfusionMatrix confusionMatrix, final TwoClassConfusionMatrix stats) {
        double labelValue = getLabelValue(instance, labelIndex);

        if (covers) {
            Condition labelAttribute = head.getCondition(labelIndex);

            if (labelAttribute != null) {
                if (labelAttribute.getValue() != labelValue) {
                    confusionMatrix.addFalsePositives(instance.weight());

                    if (stats != null) {
                        stats.addFalsePositives(instance.weight());
                    }
                } else {
                    confusionMatrix.addTruePositives(instance.weight());

                    if (stats != null) {
                        stats.addTruePositives(instance.weight());
                    }
                }
            } else {
                if (labelValue == 1.0) {
                    confusionMatrix.addFalsePositives(instance.weight());

                    if (stats != null) {
                        stats.addFalsePositives(instance.weight());
                    }
                } else {
                    confusionMatrix.addTruePositives(instance.weight());

                    if (stats != null) {
                        stats.addTruePositives(instance.weight());
                    }
                }
            }
        } else {

            if (labelValue == 1.0) {
                confusionMatrix.addFalseNegatives(instance.weight());

                if (stats != null) {
                    stats.addFalseNegatives(instance.weight());
                }
            } else {
                confusionMatrix.addTrueNegatives(instance.weight());

                if (stats != null) {
                    stats.addTrueNegatives(instance.weight());
                }
            }
        }
    }

    public final MetaData evaluate(final Instances instances, final MultiHeadRule rule,
                                   final Heuristic heuristic,
                                   final Collection<Integer> relevantLabels, final MetaData metaData) {
        TwoClassConfusionMatrix stats = new TwoClassConfusionMatrix();

        if (metaData != null) {
            stats.addTrueNegatives(metaData.stats.getNumberOfTrueNegatives());
            stats.addFalseNegatives(metaData.stats.getNumberOfFalseNegatives());
        }

        MetaData result = evaluate(instances, rule, heuristic, relevantLabels, metaData, stats);
        rule.setStats(stats);
        return result;
    }

    protected abstract MetaData evaluate(final Instances instances, final MultiHeadRule rule, final Heuristic heuristic,
                                         final Collection<Integer> relevantLabels, final MetaData metaData,
                                         final TwoClassConfusionMatrix stats);

    public static AveragingStrategy create(final String strategy) {
        if (strategy.equalsIgnoreCase(MICRO_AVERAGING)) {
            return new MicroAveraging();
        } else if (strategy.equalsIgnoreCase(LABEL_BASED_AVERAGING)) {
            return new LabelBasedAveraging();
        } else if (strategy.equalsIgnoreCase(EXAMPLE_BASED_AVERAGING)) {
            return new ExampleBasedAveraging();
        } else if (strategy.equalsIgnoreCase(MACRO_AVERAGING)) {
            return new MacroAveraging();
        }

        throw new IllegalArgumentException("Invalid averaging strategy: " + strategy);
    }

    public static AveragingStrategy createPrepending(final String strategy) {
        if (strategy.equalsIgnoreCase(MICRO_AVERAGING)) {
            return new de.tu_darmstadt.ke.seco.multilabelrulelearning.prepending.evaluation.averaging.MicroAveraging();
        } else if (strategy.equalsIgnoreCase(LABEL_BASED_AVERAGING)) {
            return new de.tu_darmstadt.ke.seco.multilabelrulelearning.prepending.evaluation.averaging.LabelBasedAveraging();
        } else if (strategy.equalsIgnoreCase(EXAMPLE_BASED_AVERAGING)) {
            return new de.tu_darmstadt.ke.seco.multilabelrulelearning.prepending.evaluation.averaging.ExampleBasedAveraging();
        } else if (strategy.equalsIgnoreCase(MACRO_AVERAGING)) {
            return new de.tu_darmstadt.ke.seco.multilabelrulelearning.prepending.evaluation.averaging.MacroAveraging();
        }

        throw new IllegalArgumentException("Invalid averaging strategy: " + strategy);
    }

}