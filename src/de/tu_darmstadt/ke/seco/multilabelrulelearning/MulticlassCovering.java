package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic.Characteristic;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.RuleIndependentEvaluation;
import weka.core.Attribute;
import weka.core.Instance;

import javax.annotation.Nonnull;
import java.util.*;

import static de.tu_darmstadt.ke.seco.models.Attribute.toSeCoAttribute;

public class MulticlassCovering {

    /**
     * An increasingly sorted queue with a fixed size. If adding a new element extends the maximum
     * size, the smallest element is thrown away.
     */
    private class FixedPriorityQueue<T extends Comparable<T>> extends PriorityQueue<T> {

        private static final long serialVersionUID = -5614452897067090251L;

        private final int maxSize;

        FixedPriorityQueue(final int maxSize) {
            super(maxSize);
            this.maxSize = maxSize;
        }

        @Override
        public final boolean offer(final T element) {
            if (size() >= maxSize) {
                T smallestElement = peek();

                if (element.compareTo(smallestElement) > 0) {
                    poll();
                    return super.offer(element);
                }

                return false;
            }

            return super.offer(element);
        }

    }

    /**
     * Encloses a rule and additional metadata.
     */
    private class Closure implements Comparable<Closure> {

        /**
         * The rule.
         **/
        private final MultiHeadRule rule;

        /**
         * Meta data, which is associated with the rule.
         **/
        private MetaData metaData;

        /**
         * The indices of the attributes, which are used as the rule's conditions, mapped to the
         * conditions.
         */
        private Map<Integer, Collection<Condition>> conditions;

        /**
         * True, if the rule should be further refined in the next step, false otherwise.
         */
        private boolean refineFurther;

        Closure(final MultiHeadRule rule, final MetaData metaData) {
            if (rule == null)
                throw new IllegalArgumentException("The rule may not be null");
            this.rule = rule;
            this.metaData = metaData;
            this.conditions = new HashMap<>();
            this.refineFurther = true;
        }

        void addCondition(final int index, final Condition condition) {
            Collection<Condition> conditions = this.conditions.get(index);

            if (conditions == null) {
                conditions = new LinkedList<>();
                this.conditions.put(index, conditions);
            }

            conditions.add(condition);
        }

        boolean containsCondition(final Condition condition) {
            Collection<Condition> conditions = this.conditions.get(condition.getAttr().index());
            return conditions != null &&
                    (condition instanceof NominalCondition || conditions.contains(condition));
        }

        @Override
        public int compareTo(@Nonnull final Closure closure) {
            return rule.compareTo(closure.rule);
        }

        @Override
        public String toString() {
            return rule.toString();
        }

    }

    private static final boolean DEBUG_STEP_BY_STEP = true;
    private static final boolean DEBUG_STEP_BY_STEP_V = true;

    private final MultiLabelEvaluation multiLabelEvaluation;

    private final boolean predictZero;

    public MulticlassCovering(final MultiLabelEvaluation multiLabelEvaluation,
                              final boolean predictZero) {
        this.multiLabelEvaluation = multiLabelEvaluation;
        this.predictZero = predictZero;
    }

    /**
     * @param beamWidthPercentage The beam width as a percentage of the number of attributes
     */
    public final MultiHeadRule findBestGlobalRule(final Instances instances,
                                                  final LinkedHashSet<Integer> labelIndices,
                                                  final Set<Integer> predictedLabels,
                                                  final float beamWidthPercentage) throws
            Exception {
        if (beamWidthPercentage < 0)
            throw new IllegalArgumentException("Beam width must be at least 0.0");
        else if (beamWidthPercentage > 1)
            throw new IllegalArgumentException("Beam width must be at maximum 1.0");
        int numAttributes = instances.numAttributes();
        int beamWidth = Math
                .max(1, Math.min(numAttributes, Math.round(numAttributes * beamWidthPercentage)));
        return findBestGlobalRule(instances, labelIndices, predictedLabels, beamWidth);
    }

    public final MultiHeadRule findBestGlobalRule(final Instances instances,
                                                  final LinkedHashSet<Integer> labelIndices,
                                                  final Set<Integer> predictedLabels,
                                                  final int beamWidth) throws
            Exception {
        if (beamWidth < 1)
            throw new IllegalArgumentException("Beam width must be at least 1");
        else if (beamWidth > instances.numAttributes())
            throw new IllegalArgumentException(
                    "Beam width must be at maximum " + instances.numAttributes());
        if (DEBUG_STEP_BY_STEP)
            System.out.println(instances.size() + " instances remaining");
        Queue<Closure> bestClosures = new FixedPriorityQueue<>(beamWidth);
        boolean improved = true;

        while (improved) { // Until no improvement possible
            improved = refineRule(instances, labelIndices, predictedLabels, bestClosures);

            if (improved && DEBUG_STEP_BY_STEP_V)
                System.out.println(
                        "Specialized rule conditions (beam width = " + beamWidth + "): " +
                                Arrays.toString(bestClosures.toArray()));
        }

        MultiHeadRule bestRule = getBestRule(bestClosures);
        if (DEBUG_STEP_BY_STEP)
            System.out.println("Found best rule: " + bestRule + "\n");
        return bestRule;
    }

    private boolean refineRule(final Instances instances, final LinkedHashSet<Integer> labelIndices,
                               final Set<Integer> predictedLabels,
                               final Queue<Closure> closures) throws
            Exception {
        boolean improved = false;

        for (Closure closure : beamWidthIterable(closures)) {
            if (closure == null || closure.refineFurther) {
                if (closure != null) {
                    closure.refineFurther = false;
                }

                // When starting off with a new rule, try empty body first
                if (closure == null) {
                    MultiHeadRule refinedRule = new MultiHeadRule(
                            multiLabelEvaluation.getHeuristic());
                    Closure refinedClosure = new Closure(refinedRule, null);
                    refinedClosure = findBestHead(instances, labelIndices, refinedClosure);

                    if (refinedClosure != null) {
                        improved |= closures.offer(refinedClosure);
                    }
                }

                for (int i : attributeIterable(instances, labelIndices,
                        predictedLabels)) { // For all attributes
                    Attribute attribute = instances.attribute(i);

                    for (Condition condition : attribute.isNumeric() ?
                            numericConditionsIterable(instances, labelIndices, attribute) :
                            nominalConditionsIterable(attribute)) {

                        // If condition is not part of the rule
                        if (closure == null || !closure.containsCondition(condition)) {
                            MultiHeadRule refinedRule =
                                    closure != null ? (MultiHeadRule) closure.rule.copy() :
                                            new MultiHeadRule(multiLabelEvaluation.getHeuristic());
                            refinedRule.addCondition(condition);
                            Closure refinedClosure = new Closure(refinedRule,
                                    closure != null ? closure.metaData : null);
                            refinedClosure.addCondition(i, condition);
                            refinedClosure = findBestHead(instances, labelIndices, refinedClosure);

                            if (refinedClosure != null) {
                                improved |= closures.offer(refinedClosure);
                            }
                        }
                    }
                }
            }
        }

        return improved;
    }

    private Iterable<Integer> attributeIterable(final Instances instances,
                                                final LinkedHashSet<Integer> labelIndices,
                                                final Set<Integer> predictedLabels) {
        return () -> new Iterator<Integer>() {

            private final Iterator<Integer> labelIterator = predictedLabels.iterator();

            private final int numAttributes = instances.numAttributes() - labelIndices.size();

            private int i = 0;

            private int count = 0;

            @Override
            public boolean hasNext() {
                return count < numAttributes || labelIterator.hasNext();
            }

            @Override
            public Integer next() {
                int next;

                if (count < numAttributes) {
                    while (labelIndices.contains(i)) {
                        i++;
                    }
                    next = i;
                    i++;
                } else {
                    next = labelIterator.next();
                }

                count++;
                return next;
            }
        };

    }

    private Closure findBestHead(final Instances instances, final LinkedHashSet<Integer> labelIndices,
                                 final Closure closure) throws
            Exception {
        closure.rule.setHead(null);
        Characteristic characteristic = multiLabelEvaluation.getCharacteristic();

        if (characteristic == Characteristic.DECOMPOSABLE) {
            return decomposite(instances, labelIndices, closure);
        } else if (characteristic == Characteristic.ANTI_MONOTONOUS) {
            return prunedSearch(instances, labelIndices, closure, null, new HashSet<>(),
                    new LinkedList<>());
        } else {
            throw new RuntimeException(
                    "Only anti-monotonous or decomposable evaluation metrics are supported for " +
                            "learning multi-label head rules");
        }
    }

    private Closure decomposite(final Instances instances, final LinkedHashSet<Integer> labelIndices,
                                final Closure closure) {
        Closure result = null;

        for (int labelIndex : labelIndices) { // For all possible label conditions
            Closure currentClosure = null;

            for (double value = predictZero ? 0 : 1; value <= 1; value++) {
                Attribute labelAttribute = instances.attribute(labelIndex);
                Condition labelCondition = new NominalCondition(toSeCoAttribute(labelAttribute),
                        value);

                if (!closure.containsCondition(labelCondition)) {
                    MultiHeadRule singleHeadRule = (MultiHeadRule) closure.rule.copy();
                    Head head = new Head();
                    head.addCondition(labelCondition);
                    singleHeadRule.setHead(head);
                    Closure singleHeadClosure = new Closure(singleHeadRule, null);
                    multiLabelEvaluation
                            .evaluate(instances, labelIndices, singleHeadClosure.rule, null);

                    if (currentClosure == null ||
                            singleHeadClosure.rule.getRuleValue() >=
                                    currentClosure.rule.getRuleValue()) {
                        currentClosure = singleHeadClosure;
                    }
                }
            }

            if (currentClosure != null &&
                    currentClosure.rule.getStats().getNumberOfTruePositives() > 0) {
                if (result == null) {
                    result = currentClosure;
                } else {
                    if (currentClosure.rule.getRuleValue() == result.rule.getRuleValue()) {
                        result.rule.getHead()
                                .addCondition(currentClosure.rule.getHead().iterator().next());
                        result.rule.getStats()
                                .addTruePositives(
                                        currentClosure.rule.getStats().getNumberOfTruePositives());
                        result.rule.getStats().addFalsePositives(
                                currentClosure.rule.getStats().getNumberOfFalsePositives());
                        result.rule.getStats()
                                .addTrueNegatives(
                                        currentClosure.rule.getStats().getNumberOfTrueNegatives());
                        result.rule.getStats().addFalseNegatives(
                                currentClosure.rule.getStats().getNumberOfFalseNegatives());
                    } else if (currentClosure.compareTo(result) > 0) {
                        result = currentClosure;
                    }
                }
            }
        }

        return result;
    }

    private Closure prunedSearch(final Instances instances, final LinkedHashSet<Integer> labelIndices,
                                 final Closure closure, final Closure bestClosure,
                                 final Set<Integer> evaluatedHeads,
                                 final List<Head> prunedHeads) throws Exception {
        Closure result = bestClosure;
        SortedMap<Double, Closure> sortedMap = new TreeMap<>(Comparator.reverseOrder());

        for (int labelIndex : labelIndices) {
            if (closure.rule.getHead() == null ||
                    !closure.rule.getHead().containsCondition(labelIndex)) {
                Closure refinedClosure = null;

                for (double value = predictZero ? 0 : 1; value <= 1; value++) {
                    Attribute labelAttribute = instances.attribute(labelIndex);
                    Condition labelCondition = new NominalCondition(toSeCoAttribute(labelAttribute),
                            value);

                    if (closure.containsCondition(labelCondition)) {
                        break;
                    } else {
                        MultiHeadRule refinedRule = (MultiHeadRule) closure.rule.copy();
                        Head head = refinedRule.getHead();

                        if (head == null) {
                            head = new Head();
                            refinedRule.setHead(head);
                        }

                        head.addCondition(labelCondition);

                        // If head has not already been evaluated
                        if (isHeadPruned(prunedHeads, evaluatedHeads, head)) {
                            break;
                        } else {
                            boolean isRuleIndependent = multiLabelEvaluation
                                    .getEvaluationStrategy() instanceof RuleIndependentEvaluation;
                            Closure currentClosure = new Closure(refinedRule,
                                    isRuleIndependent ? closure.metaData : null);
                            MetaData metaData = multiLabelEvaluation
                                    .evaluate(instances, labelIndices, currentClosure.rule,
                                            isRuleIndependent ? currentClosure.metaData : null);

                            if (isRuleIndependent) {
                                currentClosure.metaData = metaData;
                            }

                            if (refinedClosure == null ||
                                    currentClosure.rule.getRuleValue() >=
                                            refinedClosure.rule.getRuleValue()) {
                                refinedClosure = currentClosure;
                            }
                        }
                    }
                }

                if (refinedClosure != null) {
                    if (refinedClosure.rule.getStats().getNumberOfTruePositives() > 0 &&
                            (result == null ||
                                    refinedClosure.rule.getRuleValue() >=
                                            result.rule.getRuleValue())) {
                        sortedMap.put(refinedClosure.rule.getRuleValue(), refinedClosure);
                        evaluatedHeads.add(hashCodeOfConditions(
                                refinedClosure.rule.getHead().getConditions()));
                    } else {
                        prunedHeads.add(refinedClosure.rule.getHead());
                    }
                }
            }
        }

        Iterator<Closure> iterator = sortedMap.values().iterator();
        boolean first = true;

        while (iterator.hasNext()) {
            Closure refinedClosure = iterator.next();

            if (first) {
                result = refinedClosure;
            }

            result = prunedSearch(instances, labelIndices, refinedClosure, result, evaluatedHeads,
                    prunedHeads);
            first = false;
        }

        return result;
    }

    private boolean isHeadPruned(final List<Head> prunedHeads, final Set<Integer> evaluatedHeads,
                                 final Head head) {
        if (evaluatedHeads.contains(hashCodeOfConditions(head.getConditions()))) {
            return true;
        } else {
            return isHeadPruned(prunedHeads, head);
        }
    }

    private boolean isHeadPruned(final List<Head> prunedHeads, final Head head) {
        for (Head prunedHead : prunedHeads) {
            boolean isSubset = true;

            for (Condition condition : prunedHead) {
                if (!head.containsCondition(condition.getAttr().index())) {
                    isSubset = false;
                    break;
                }
            }

            if (isSubset) {
                return true;
            }
        }

        return false;
    }

    public int hashCodeOfConditions(final Collection<Condition> conditions) {
        final int prime = 31;
        int result = 0;

        for (Condition condition : conditions) {
            result = prime * result + condition.getAttr().name().hashCode();
        }

        return result;
    }

    private MultiHeadRule getBestRule(final Queue<Closure> closures) {
        MultiHeadRule bestRule = null;

        while (!closures.isEmpty()) {
            bestRule = closures.poll().rule;
        }

        return bestRule;
    }

    private Iterable<Closure> beamWidthIterable(final Queue<Closure> closures) {
        return () -> new Iterator<Closure>() {

            private final Object[] items = closures.toArray();

            private int i = 0;

            @Override
            public boolean hasNext() {
                return i == 0 || i < items.length;
            }

            @Override
            public Closure next() {
                if (i < items.length) {
                    Closure next = (Closure) items[i];
                    i++;
                    return next;
                } else {
                    i++;
                    return null;
                }
            }

        };
    }

    private Iterable<Condition> numericConditionsIterable(final Instances instances,
                                                          final LinkedHashSet<Integer> labelIndices,
                                                          final Attribute attribute) {
        instances.sort(attribute.index());

        return () -> new Iterator<Condition>() {

            private final de.tu_darmstadt.ke.seco.models.Attribute secoAttribute =
                    toSeCoAttribute(attribute);

            private int i = 2; // Start at second instance, because it makes no sense to use the smallest value as a split point

            private Instance lastInstance = instances.size() > 0 ? instances.get(0) : null;

            private double[] lastLabelVector =
                    instances.size() > 0 ? getLabelVector(instances.get(0), labelIndices) : null;

            private NumericCondition next = null;

            @Override
            public boolean hasNext() {
                while (i < (instances.size()) * 2) {
                    if (i % 2 > 0) {
                        double splitPoint = next.getValue();
                        next = new NumericCondition(secoAttribute, splitPoint, true);
                        i++;
                        return true;
                    } else {
                        Instance instance = instances.get(i / 2);
                        double[] labelVector = getLabelVector(instance, labelIndices);

                        if (!Arrays.equals(labelVector, lastLabelVector)) {
                            double value = instance.value(attribute.index());
                            double lastValue = lastInstance.value(attribute.index());
                            double splitPoint = lastValue + (value - lastValue) / 2.0;
                            next = new NumericCondition(secoAttribute, splitPoint, false);
                            lastInstance = instance;
                            lastLabelVector = labelVector;
                            i++;
                            return true;
                        } else {
                            lastInstance = instance;
                            lastLabelVector = labelVector;
                            i += 2;
                        }
                    }
                }

                next = null;
                return false;
            }

            @Override
            public Condition next() {
                return next;
            }

        };
    }

    private double[] getLabelVector(final Instance instance, final LinkedHashSet<Integer> labelIndices) {
        Instance wrappedInstance =
                instance instanceof DenseInstanceWrapper ?
                        ((DenseInstanceWrapper) instance).getWrappedInstance() :
                        ((SparseInstanceWrapper) instance).getWrappedInstance();
        double[] labelVector = new double[labelIndices.size()];
        int i = 0;

        for (int labelIndex : labelIndices) {
            labelVector[i] = wrappedInstance.value(labelIndex);
            i++;
        }

        return labelVector;
    }

    private Iterable<Condition> nominalConditionsIterable(final Attribute attribute) {
        return () -> new Iterator<Condition>() {

            private final de.tu_darmstadt.ke.seco.models.Attribute secoAttribute =
                    toSeCoAttribute(attribute);

            private int i = 0;

            @Override
            public boolean hasNext() {
                return i < attribute.numValues();
            }

            @Override
            public Condition next() {
                Condition condition = new NominalCondition(secoAttribute, i);
                i++;
                return condition;
            }

        };
    }

}