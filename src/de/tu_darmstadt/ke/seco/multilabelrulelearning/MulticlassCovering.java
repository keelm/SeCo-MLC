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
     * An increasingly sorted queue with a fixed size. If adding a new element extends the maximum size, the smallest
     * element is thrown away.
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
         * The indices of the attributes, which are used as the rule's conditions, mapped to the conditions.
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
            return conditions != null && (condition instanceof NominalCondition || conditions.contains(condition));
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


    private static HashSet<Integer> labelIndicesHash;
    private static Hashtable<Integer,Boolean> coveringCache;

    private final MultiLabelEvaluation multiLabelEvaluation;

    private final boolean predictZero;

    public MulticlassCovering(final MultiLabelEvaluation multiLabelEvaluation, final boolean predictZero, int[] labelIndices) {
        this.multiLabelEvaluation = multiLabelEvaluation;
        this.predictZero = predictZero;
        labelIndicesHash=new HashSet<Integer>(labelIndices.length);
        for (int i = 0; i < labelIndices.length; i++) {
        	labelIndicesHash.add(labelIndices[i]);
		}
        coveringCache=new Hashtable<>();
    }

    public static boolean isLabelIndex(int index){
    	return labelIndicesHash.contains(index);
    }


    static int nHashed=0;
    static int nNonHashed=0;
    
    public static boolean cachedCovers(Condition c, Instance inst) {
    	if(false && !isLabelIndex(c.getAttr().index())){
    		//https://stackoverflow.com/questions/11742593/what-is-the-hashcode-for-a-custom-class-having-just-two-int-properties
//    		int hashCode = (17*31+c.hashCode())*31+inst.hashCode(); // this apparently is not very efficient
    		int hashCode = getHashCodeTemp(c, inst);
    		Boolean res = getCachedResultTemp(hashCode);
    		if(res==null){
    			boolean resComputed = c.covers(inst);
    			coveringCache.put(hashCode,resComputed);
//    			System.out.println(coveringCache.size()+" "+nHashed+" "+nNonHashed+ " put into hash "+resComputed+" "+c+" "+inst);
    			return resComputed;
    		}else{
//    			System.out.println(coveringCache.size()+" "+nHashed+" "+nNonHashed + " already in hash "+c+" "+inst+" ");
    			nHashed++;
    			return res;
    		}
    	}else{
    		nNonHashed++;
    		//because this might change (labelfeatures might change)
    		return c.covers(inst);
    	}
	}

	/**
	 * @param hashCode
	 * @return
	 */
	private static Boolean getCachedResultTemp(int hashCode) {
		Boolean res=coveringCache.get(hashCode);
		return res;
	}

	/**
	 * @param c
	 * @param inst
	 * @return
	 */
	private static int getHashCodeTemp(Condition c, Instance inst) {
		int hashCode = 17*31+c.getAttr().index();
		hashCode = (int) (hashCode * 31 + Double.doubleToLongBits(c.getValue()));
		hashCode = hashCode* 31 + System.identityHashCode(inst);
		hashCode = hashCode* 31 + (c.cmp() ? 1: 0);
		return hashCode;
	}

    
    /**
     * @param beamWidthPercentage The beam width as a percentage of the number of attributes
     */
    public final MultiHeadRule findBestGlobalRule(final Instances instances, final int[] labelIndices,
                                                  final Set<Integer> predictedLabels,
                                                  final float beamWidthPercentage) throws Exception {
        if (beamWidthPercentage < 0)
            throw new IllegalArgumentException("Beam width must be at least 0.0");
        else if (beamWidthPercentage > 1)
            throw new IllegalArgumentException("Beam width must be at maximum 1.0");
        int numAttributes = instances.numAttributes();
        int beamWidth = Math.max(1, Math.min(numAttributes, Math.round(numAttributes * beamWidthPercentage)));
        return findBestGlobalRule(instances, labelIndices, predictedLabels, beamWidth);
    }

    public final MultiHeadRule findBestGlobalRule(final Instances instances, final int[] labelIndices,
                                                  final Set<Integer> predictedLabels,
                                                  final int beamWidth) throws
            Exception {
        if (beamWidth < 1)
            throw new IllegalArgumentException("Beam width must be at least 1");
        else if (beamWidth > instances.numAttributes())
            throw new IllegalArgumentException("Beam width must be at maximum " + instances.numAttributes());
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

    private boolean refineRule(final Instances instances, final int[] labelIndices, final Set<Integer> predictedLabels,
                               final Queue<Closure> closures) throws
            Exception {
        boolean improved = false;

        for (Closure closure : beamWidthIterable(closures)) {
            if (closure == null || closure.refineFurther) {
                if (closure != null) {
                    closure.refineFurther = false;
                }

                for (int i : attributeIterable(instances, labelIndices, predictedLabels)) { // For all attributes
                    Attribute attribute = instances.attribute(i);

                    for (Condition condition : attribute.isNumeric() ?
                            numericConditionsIterable(instances, labelIndices, attribute) :
                            nominalConditionsIterable(attribute)) {

                        // If condition is not part of the rule
                        if (closure == null || !closure.containsCondition(condition)) {
                            MultiHeadRule refinedRule = closure != null ? (MultiHeadRule) closure.rule.copy() :
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

    private Iterable<Integer> attributeIterable(final Instances instances, final int[] labelIndices,
                                                final Set<Integer> predictedLabels) {
        return () -> new Iterator<Integer>() {

            private final Iterator<Integer> labelIterator = predictedLabels.iterator();

            private int i = 0;

            @Override
            public boolean hasNext() {
                return i < (instances.numAttributes() - labelIndices.length) || labelIterator.hasNext();
            }

            @Override
            public Integer next() {
                if (i < (instances.numAttributes() - labelIndices.length)) {
                    return i++;
                } else {
                    return labelIterator.next();
                }
            }
        };

    }

    private Closure findBestHead(final Instances instances, final int[] labelIndices, final Closure closure) throws
            Exception {
        closure.rule.setHead(null);
        Characteristic characteristic = multiLabelEvaluation.getCharacteristic();

        if (characteristic == Characteristic.DECOMPOSABLE) {
            return decomposite(instances, labelIndices, closure);
        } else if (characteristic == Characteristic.ANTI_MONOTONOUS) {
            return prunedSearch(instances, labelIndices, closure, null, new LinkedList<>());
        } else {
            throw new RuntimeException("Only anti-monotonous or decomposable evaluation metrics are supported for " +
                    "learning multi-label head rules");
        }
    }

    private Closure decomposite(final Instances instances, final int[] labelIndices, final Closure closure) {
        Closure result = null;

        for (int labelIndex : labelIndices) { // For all possible label conditions
            Closure currentClosure = null;

            for (double value = predictZero ? 0 : 1; value <= 1; value++) {
                Attribute labelAttribute = instances.attribute(labelIndex);
                Condition labelCondition = new NominalCondition(toSeCoAttribute(labelAttribute), value);

                if (!closure.containsCondition(labelCondition)) {
                    MultiHeadRule singleHeadRule = (MultiHeadRule) closure.rule.copy();
                    Head head = new Head();
                    head.addCondition(labelCondition);
                    singleHeadRule.setHead(head);
                    Closure singleHeadClosure = new Closure(singleHeadRule, null);
                    multiLabelEvaluation.evaluate(instances, labelIndices, singleHeadClosure.rule, null);
//                    System.out.println(singleHeadClosure);

                    if (currentClosure == null ||
                            singleHeadClosure.rule.getRuleValue() >= currentClosure.rule.getRuleValue()) {
                        currentClosure = singleHeadClosure;
                    }
                }
            }

            if (currentClosure != null && currentClosure.rule.getStats().getNumberOfTruePositives() > 0) {
                if (result == null) {
                    result = currentClosure;
                } else {
                    if (currentClosure.rule.getRuleValue() == result.rule.getRuleValue()) {
                        result.rule.getHead().addCondition(currentClosure.rule.getHead().iterator().next());
                        result.rule.getStats()
                                .addTruePositives(currentClosure.rule.getStats().getNumberOfTruePositives());
                        result.rule.getStats().addFalsePositives(
                                currentClosure.rule.getStats().getNumberOfFalsePositives());
                        result.rule.getStats()
                                .addTrueNegatives(currentClosure.rule.getStats().getNumberOfTrueNegatives());
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

    private Closure prunedSearch(final Instances instances, final int[] labelIndices, final Closure closure,
                                 final Closure bestClosure, final List<Head> prunedHeads) throws Exception {
        Closure result = bestClosure;

        for (int labelIndex : labelIndices) { // For all possible label conditions
            if (closure.rule.getHead() == null || !closure.rule.getHead()
                    .containsCondition(labelIndex)) { // If label is not already contained in head
                Closure refinedClosure = null;

                for (double value = predictZero ? 0 : 1; value <= 1; value++) {
                    Attribute labelAttribute = instances.attribute(labelIndex);
                    Condition labelCondition = new NominalCondition(toSeCoAttribute(labelAttribute), value);

                    // If label is not included in the rule's conditions
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
                        if (isHeadPruned(prunedHeads, head)) {
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
                                    currentClosure.rule.getRuleValue() >= refinedClosure.rule.getRuleValue()) {
                                refinedClosure = currentClosure;
                            }
                        }
                    }
                }

                if (refinedClosure != null) {
                    if (refinedClosure.rule.getStats().getNumberOfTruePositives() > 0) {
                        if (result == null || refinedClosure.rule.getRuleValue() >= result.rule.getRuleValue()) {
                            Closure x = prunedSearch(instances, labelIndices, refinedClosure, refinedClosure,
                                    prunedHeads);



                            if (result == null || x.rule.getRuleValue() > result.rule.getRuleValue() ||
                                    (x.rule.getRuleValue() == result.rule.getRuleValue() &&
                                            x.rule.getHead().size() > result.rule.getHead().size())) {
                                result = x;
                            }
                        }
                    }

                    prunedHeads.add(refinedClosure.rule.getHead());
                }
            }
        }

        return result;
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

    private Iterable<Condition> numericConditionsIterable(final Instances instances, final int[] labelIndices,
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

    private double[] getLabelVector(final Instance instance, final int[] labelIndices) {
        Instance wrappedInstance =
                instance instanceof DenseInstanceWrapper ? ((DenseInstanceWrapper) instance).getWrappedInstance() :
                        ((SparseInstanceWrapper) instance).getWrappedInstance();
        double[] labelVector = new double[labelIndices.length];

        for (int i = 0; i < labelIndices.length; i++) {
            int labelIndex = labelIndices[i];
            labelVector[i] = wrappedInstance.value(labelIndex);
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