package de.tu_darmstadt.ke.seco.multilabelrulelearning;


import com.google.common.collect.SortedSetMultimap;
import com.google.common.collect.TreeMultimap;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic.Characteristic;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.RuleIndependentEvaluation;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
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
    public class Closure implements Comparable<Closure> {

        /**
         * The rule.
         **/
        private MultiHeadRule rule;

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

    private static final boolean DEBUG_STEP_BY_STEP = false;
    private static final boolean DEBUG_STEP_BY_STEP_V = false;

    private static HashSet<Integer> labelIndicesHash;
    private static Hashtable<Integer,Boolean> coveringCache;

    private final MultiLabelEvaluation multiLabelEvaluation;

    private final boolean predictZero;

    public MulticlassCovering(final MultiLabelEvaluation multiLabelEvaluation,
                              final boolean predictZero,
                              final BoostingStrategy boostingStrategy,
                              final boolean useRelaxedPruning,
                              final boolean useBoostedHeuristicForRules,
                              final int pruningDepth) {
        this.multiLabelEvaluation = multiLabelEvaluation;
        this.predictZero = predictZero;
        this.boostingStrategy = boostingStrategy;
        this.useRelaxedPruning = useRelaxedPruning;
        this.useBoostedHeuristicForChoosingRules = useBoostedHeuristicForRules;
        this.pruningDepth = pruningDepth;
    }

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
    	if(true && !isLabelIndex(c.getAttr().index())){
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

    public static boolean finished = false;
    boolean fixHead = false;
    Head fixedHead = null;

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
        fixHead = false;

        while (improved) { // Until no improvement possible
            improved = refineRule(instances, labelIndices, predictedLabels, bestClosures);
            if (!bestClosures.isEmpty()) {
                fixHead = true;
                fixedHead = ((Closure) bestClosures.toArray()[0]).rule.getHead();
            }


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
                    MultiHeadRule refinedRule = new MultiHeadRule(multiLabelEvaluation.getHeuristic());
                    Closure refinedClosure = new Closure(refinedRule, null);
                    refinedClosure = findBestHead(instances, labelIndices, refinedClosure);

                    if (refinedClosure != null) {
                        improved |= closures.offer(refinedClosure);
                    }
                }

                for (int i : attributeIterable(instances, labelIndices, predictedLabels)) { // For all attributes
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
                            //System.out.println(refinedClosure);
                            //System.out.println(refinedClosure);
                            increaseEvaluationCount();
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
                                 final Closure closure) throws Exception {
        //System.out.println(fixHead + " " + closure.rule.getBody());
        if (fixHead) {
            multiLabelEvaluation.evaluate(instances, labelIndices, closure.rule, null);
            if (closure.rule.getStats().getNumberOfTruePositives() <= 0)
                return null;
            if (closure.rule.getStats().getNumberOfTruePositives() <closure.rule.getStats().getNumberOfFalsePositives())
                return null;
            return closure;
        }


        closure.rule.setHead(null);
        Characteristic characteristic = multiLabelEvaluation.getCharacteristic();

        if (useRelaxedPruning) {
            if (characteristic == Characteristic.DECOMPOSABLE) {
                //return findBestRelaxedHeadAntiMonotonic(instances, labelIndices, closure, null, new HashSet<>(), new LinkedList<>());

                return findBestRelaxedHeadDecomposable(instances, labelIndices, closure);
            } else if (characteristic == Characteristic.ANTI_MONOTONOUS) {
                return findBestRelaxedHeadAntiMonotonic(instances, labelIndices, closure, null, new HashSet<>(), new LinkedList<>());
            } else {
                throw new RuntimeException("Only anti-monotonous or decomposable evaluation metrics are supported for learning multi-label head rules");
            }
        } else {
            if (characteristic == Characteristic.DECOMPOSABLE) {
                return decomposite(instances, labelIndices, closure);
            } else if (characteristic == Characteristic.ANTI_MONOTONOUS) {
                return prunedSearch(instances, labelIndices, closure, null, new HashSet<>(), new LinkedList<>());
            } else {
                throw new RuntimeException("Only anti-monotonous or decomposable evaluation metrics are supported for learning multi-label head rules");
            }
        }
    }

    /*************************************************
     * EvaluationSetting for learning more multi-label heads.
     *************************************************/

    /**
     * True if relaxed pruning is to applied. False otherwise, correspond to pruning as implemented by [Rapp, 2016].
     */
    public boolean useRelaxedPruning = true;

    /**
     * The boosting strategy to use. Defines how much boost to apply depending on the number of labels in the head.
     */
    public BoostingStrategy boostingStrategy;

    /**
     * True if the boosted heuristic value is to be used for evaluating rules.
     * False otherwise, corresponds to the unboosted heuristic value being used for assessing rules.
     */
    public boolean useBoostedHeuristicForChoosingRules = true;

    /**
     * Variables for tracking the number of evaluations per execution of findBestHead().
     * Corresponds approximately to the number of different heads evaluated in the search space.
     */
    public static int evaluations = 0;
    public static int evaluatedHeads = 0;
    public static double evaluationsPerHead = 0;

    /**
     * Trade off between finding the best possible head and efficiency.
     * A higher value implies higher chances of finding the best possible head
     * but also reduced efficiency. Number of subsequent boost function values checked.
     * If set to -1 the best possible head is guaranteed!
     */
    public int pruningDepth = -1;

    // TODO: refactor
    private Closure findBestRelaxedHeadDecomposable(final Instances instances, final LinkedHashSet<Integer> labelIndices, final Closure closure) {

        // save all single label heads heuristic values and closures
        SortedMultimap singleLabelHeads = findAllSingleLabelHeads(instances, labelIndices, closure);
        // data structures for keeping track of the so far induced heads
        SortedMap<Double, Closure> boostedMultiLabelHeads = new TreeMap<>(Comparator.reverseOrder());
        HashMap<Integer, Head> headsOfLengthN = new HashMap<>();
        HashMap<Integer, Closure> closuresWithHeadOfLengthN = new HashMap<>();
        // initialize the first head
        Double bestSingleLabelHeuristicValue = singleLabelHeads.firstKey();
        Closure bestSingleLabelHeuristicClosure = singleLabelHeads.get(bestSingleLabelHeuristicValue);
        boostedMultiLabelHeads.put(bestSingleLabelHeuristicValue, bestSingleLabelHeuristicClosure);
        Head bestSingleLabelHead = bestSingleLabelHeuristicClosure.rule.getHead();
        headsOfLengthN.put(1, bestSingleLabelHead);
        closuresWithHeadOfLengthN.put(1, bestSingleLabelHeuristicClosure);
        singleLabelHeads.remove(bestSingleLabelHeuristicValue, bestSingleLabelHeuristicClosure);
        Heuristic heuristic = multiLabelEvaluation.getHeuristic();
        // removes rules without any true positives
        if (bestSingleLabelHeuristicClosure.rule.getStats().getNumberOfTruePositives() <= 0)
            return null;

        boolean finish = false;
         // TODO: copy conditions?
        Closure topClosure = bestSingleLabelHeuristicClosure;
        // for all head lengths
        for (int n = 2; n <= labelIndices.size(); n++) {
            // TODO: copy properly
            Head bestHeadOfLessLength = headsOfLengthN.get(n-1).clone();
            Closure bestRemainingHeuristicClosure = null;
            Condition conditionToBeAdded = null;

            boolean labelAlreadyInHead = true;
            while (labelAlreadyInHead) {
                // no more single labels to add
                if (singleLabelHeads.isEmpty()) {
                    finish = true;
                    break;
                }
                Double bestRemainingHeuristicValue = singleLabelHeads.firstKey();
                bestRemainingHeuristicClosure = singleLabelHeads.get(bestRemainingHeuristicValue);
                Head bestRemainingHead = bestRemainingHeuristicClosure.rule.getHead();
                Condition bestRemainingCondition = (Condition) bestRemainingHead.getConditions().toArray()[0];
                int attributeIndex = bestRemainingCondition.getAttr().index();
                Collection<Integer> labelIndicesInHead = bestHeadOfLessLength.getLabelIndices();
                labelAlreadyInHead = labelIndicesInHead.contains(attributeIndex);
                singleLabelHeads.remove(bestRemainingHeuristicValue, bestRemainingHeuristicClosure);
                conditionToBeAdded = bestRemainingCondition;
            }
            if (finish)
                break;

            // add label to head
            bestHeadOfLessLength.addCondition(conditionToBeAdded);
            headsOfLengthN.put(n, bestHeadOfLessLength);

            // create new closure
            Closure multiClosure = new Closure((MultiHeadRule) closure.rule.copy(), null);
            multiClosure.rule.setStats(new TwoClassConfusionMatrix());

            // add statistics from best remaining single label head
            multiClosure.rule.getStats().addTruePositives(bestRemainingHeuristicClosure.rule.getStats().getNumberOfTruePositives());
            multiClosure.rule.getStats().addFalsePositives(bestRemainingHeuristicClosure.rule.getStats().getNumberOfFalsePositives());
            multiClosure.rule.getStats().addTrueNegatives(bestRemainingHeuristicClosure.rule.getStats().getNumberOfTrueNegatives());
            multiClosure.rule.getStats().addFalseNegatives(bestRemainingHeuristicClosure.rule.getStats().getNumberOfFalseNegatives());

            // add statistics from previous closure
            Closure prevClosure = closuresWithHeadOfLengthN.get(n-1);
            multiClosure.rule.getStats().addTruePositives(prevClosure.rule.getStats().getNumberOfTruePositives());
            multiClosure.rule.getStats().addFalsePositives(prevClosure.rule.getStats().getNumberOfFalsePositives());
            multiClosure.rule.getStats().addTrueNegatives(prevClosure.rule.getStats().getNumberOfTrueNegatives());
            multiClosure.rule.getStats().addFalseNegatives(prevClosure.rule.getStats().getNumberOfFalseNegatives());

            // set rule values and head
            multiClosure.rule.setHead(bestHeadOfLessLength);
            double rawRuleValue = heuristic.evaluateRule(multiClosure.rule);
            this.evaluations += 1;
            multiClosure.rule.setRawRuleValue(rawRuleValue);

            // apply boosting
            boostingStrategy.evaluate(multiClosure.rule);

            // set rule value depending on whether or not to use boosted heuristic
            double ruleValue = useBoostedHeuristicForChoosingRules ?  multiClosure.rule.getBoostedRuleValue() : multiClosure.rule.getRawRuleValue();
            multiClosure.rule.setRuleValue(heuristic, ruleValue);

            // update best closure
            if (multiClosure.rule.getBoostedRuleValue() >= topClosure.rule.getBoostedRuleValue())
                topClosure = multiClosure;



            // prune if the best value cannot be reached anymore
            double maximumBoostValue = boostingStrategy.getMaximumBoost(multiClosure.rule.getHead().size());
            if (multiClosure.rule.getRawRuleValue() * maximumBoostValue < topClosure.rule.getBoostedRuleValue()) {
                if (topClosure.rule.getStats().getNumberOfTruePositives() >= topClosure.rule.getStats().getNumberOfFalsePositives()) {
                    return topClosure;
                } else {
                    return null;
                }
            }

            // add to the sorted map
            if (boostedMultiLabelHeads.containsKey(multiClosure.rule.getBoostedRuleValue()))
                boostedMultiLabelHeads.remove(multiClosure.rule.getBoostedRuleValue());
            boostedMultiLabelHeads.put(multiClosure.rule.getBoostedRuleValue(), multiClosure);

            closuresWithHeadOfLengthN.put(n, multiClosure);
        }

        // filter out rules with fewer true positives than false positives
        Closure bestClosure = boostedMultiLabelHeads.get(boostedMultiLabelHeads.firstKey());
        if (bestClosure.rule.getStats().getNumberOfTruePositives() < bestClosure.rule.getStats().getNumberOfFalsePositives())
            return null;

        return boostedMultiLabelHeads.get(boostedMultiLabelHeads.firstKey());
    }

    /**
     * Evaluates all single label head rules and returns them sorted in descending order
     * in accordance to the heuristic value.
     */
    private SortedMultimap findAllSingleLabelHeads(final Instances instances, final LinkedHashSet<Integer> labelIndices, final Closure closure) {
        // save all single label heads heuristic values and closures
        SortedMultimap singleLabelHeads = new SortedMultimap();
        // for all possible labels
        for (int labelIndex : labelIndices) {
            // for both target, i.e. [red = 0] and [red = 1] (depending on whether or not to predict zero rules)
            for (double value = predictZero ? 0 : 1; value <= 1; value++) {
                Attribute labelAttribute = instances.attribute(labelIndex);
                Condition labelCondition = new NominalCondition(toSeCoAttribute(labelAttribute), value);
                // rule body may not contain the same label condition as in the head
                if (!closure.containsCondition(labelCondition)) {
                    MultiHeadRule singleHeadRule = (MultiHeadRule) closure.rule.copy();
                    Head head = new Head();
                    head.addCondition(labelCondition);
                    singleHeadRule.setHead(head);
                    Closure singleHeadClosure = new Closure(singleHeadRule, null);
                    multiLabelEvaluation.evaluate(instances, labelIndices, singleHeadClosure.rule, null);
                    singleHeadClosure.rule.setBoostedRuleValue(singleHeadClosure.rule.getRawRuleValue());
                    this.evaluations += 1;
                    singleLabelHeads.put(singleHeadClosure.rule.getRawRuleValue(), singleHeadClosure);
                }
            }
        }
        //System.out.println(singleLabelHeads);
        return singleLabelHeads;
    }

    private Closure findBestRelaxedHeadAntiMonotonic(final Instances instances, final LinkedHashSet<Integer> labelIndices,
                                                     final Closure closure, final Closure bestClosure,
                                                     final Set<Integer> evaluatedHeads,
                                                     final List<Head> prunedHeads) throws Exception {
        // best head so far
        Closure result = bestClosure;
        // variable naming?
        SortedSetMultimap<Double, Closure> improvedHeads = TreeMultimap.create(Comparator.reverseOrder(), Comparator.reverseOrder());
        // for all possible label conditions
        for (int labelIndex : labelIndices) {
            // if head empty or it does not contain condition yet
            if (closure.rule.getHead() == null || !closure.rule.getHead().containsCondition(labelIndex)) {
                Closure refinedClosure = null;

                for (double value = predictZero ? 0 : 1; value <= 1; value++) {
                    Attribute labelAttribute = instances.attribute(labelIndex);
                    Condition labelCondition = new NominalCondition(toSeCoAttribute(labelAttribute), value);
                    // rule contains condition -> break
                    if (closure.containsCondition(labelCondition)) {
                        break;
                    } else {
                        // copy old rule
                        MultiHeadRule refinedRule = (MultiHeadRule) closure.rule.copy();
                        Head head = refinedRule.getHead();
                        // if empty head -> create new head
                        if (head == null) {
                            head = new Head();
                            refinedRule.setHead(head);
                        }
                        // add label
                        head.addCondition(labelCondition);

                        // if head has not already been pruned
                        if (isHeadPruned(prunedHeads, evaluatedHeads, head)) {
                            break;
                        } else {
                            boolean isRuleIndependent = multiLabelEvaluation.getEvaluationStrategy() instanceof RuleIndependentEvaluation;
                            Closure currentClosure = new Closure(refinedRule, isRuleIndependent ? closure.metaData : null);
                            // evaluate rule with head
                            /*if (currentClosure.rule.getBody().toString().contains("outlook = sunny"))
                                System.out.println("blub");*/
                            MetaData metaData = multiLabelEvaluation.evaluate(instances, labelIndices, currentClosure.rule, isRuleIndependent ? currentClosure.metaData : null);
                            //System.out.println(currentClosure +"" + currentClosure.rule.getHead().getLabelIndices() + " best: " + bestClosure);
                            boostingStrategy.evaluate(currentClosure.rule);
                            evaluations += 1;

                            if (useBoostedHeuristicForChoosingRules)
                                currentClosure.rule.setRuleValue(currentClosure.rule.getHeuristic(), currentClosure.rule.getBoostedRuleValue());
                            else
                                currentClosure.rule.setRuleValue(currentClosure.rule.getHeuristic(), currentClosure.rule.getRawRuleValue());

                            if (isRuleIndependent) {
                                currentClosure.metaData = metaData;
                            }

                            // if new closure better than refined closure -> set new refined closure
                            if (refinedClosure == null ||  currentClosure.rule.getRuleValue() >= refinedClosure.rule.getRuleValue()) {
                                refinedClosure = currentClosure;
                            }
                        }
                    }
                }

                if (refinedClosure != null) {
                    // if refined closure better than result -> set new result
                    if (refinedClosure.rule.getStats().getNumberOfTruePositives() > 0 &&
                            (result == null || refinedClosure.rule.getBoostedRuleValue() >= result.rule.getBoostedRuleValue()) &&
                            closure.rule.getStats().getNumberOfTruePositives() >= closure.rule.getStats().getNumberOfFalsePositives()) {
                        // add refined closure
                        // map value to closure
                        improvedHeads.put(refinedClosure.rule.getRuleValue(), refinedClosure);
                        // add to evaluatedHeads heads
                        evaluatedHeads.add(hashCodeOfConditions(refinedClosure.rule.getHead().getConditions()));
                    } else {
                        // add to pruned heads
                        MultiHeadRule rule = refinedClosure.rule;
                        Head head = rule.getHead();
                        int numberOfLabelsInTheHead = head.size();
                        double maxBoost = pruningDepth == -1 ? boostingStrategy.getMaximumBoost(numberOfLabelsInTheHead) : boostingStrategy.getMaximumLookaheadBoost(numberOfLabelsInTheHead, pruningDepth);
                        double nextMaxRuleValue = rule.getRawRuleValue() * maxBoost;
                        if (result != null && nextMaxRuleValue < result.rule.getBoostedRuleValue())
                            prunedHeads.add(refinedClosure.rule.getHead());
                    }


                }
            }
        }


        // terminates if no rules in sorted map, i.e. no rules/heads were found to be an improvement
        // go through rules/heads in sorted map
        Iterator<Closure> iterator = improvedHeads.values().iterator();
        boolean first = true;
        // for all
        while (iterator.hasNext()) {
            // get refined rule
            Closure refinedClosure = iterator.next();

            if (first) {
                result = refinedClosure;
            }

            result = findBestRelaxedHeadAntiMonotonic(instances, labelIndices, refinedClosure, result, evaluatedHeads, prunedHeads);
            first = false;
        }

        return result;
    }

    /**
     * Calculates the metrics for tracking the number of evaluations per findBestHead().
     */
    private void increaseEvaluationCount() {
        double tmp = evaluationsPerHead * MulticlassCovering.evaluatedHeads;
        MulticlassCovering.evaluatedHeads++;
        tmp += evaluations;
        evaluationsPerHead = tmp / MulticlassCovering.evaluatedHeads;
        evaluations = 0;
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
                    this.evaluations += 1;

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

        //System.out.println(result);

        /*if (result != null && result.rule.getStats().getNumberOfTruePositives() < result.rule.getStats().getNumberOfFalsePositives())
            return null;*/

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

                        // If head has not already been evaluatedHeads
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
                            this.evaluations += 1;

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