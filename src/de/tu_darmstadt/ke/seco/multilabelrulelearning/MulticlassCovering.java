package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic.Characteristic;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.models.Random;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.RuleIndependentEvaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;

import javax.annotation.Nonnull;

import com.sun.corba.se.pept.transport.Acceptor;

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

    private static final boolean DEBUG_STEP_BY_STEP = false;
    private static final boolean DEBUG_STEP_BY_STEP_V = false;


    private static HashSet<Integer> labelIndicesHash;
    private static Hashtable<Integer,Boolean> coveringCache;

    private final MultiLabelEvaluation multiLabelEvaluation;

    private final boolean predictZero;
    
    protected Random random;
    private HashMap<Attribute, ArrayList<Double>> sorted;

    public MulticlassCovering(final MultiLabelEvaluation multiLabelEvaluation,
                              final boolean predictZero) {
        this.multiLabelEvaluation = multiLabelEvaluation;
        this.predictZero = predictZero;
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

	public final MultiHeadRule findBestRuleBottomUp(final Instances instances,
            								final LinkedHashSet<Integer> labelIndices,
            								final Set<Integer> predictedLabels,
            								final float beamWidthPercentage,
            								final boolean acceptEqual,
            								final boolean useSeCo,
            								final int inst,
            								final int n_step,
            								final String numericGeneralization, 
            								final boolean useRandom,
            								final boolean[] instanceStatus) throws Exception {
		if (beamWidthPercentage < 0)
			throw new IllegalArgumentException("Beam width must be at least 0.0");
		else if (beamWidthPercentage > 1)
			throw new IllegalArgumentException("Beam width must be at maximum 1.0");
		int numAttributes = instances.numAttributes();
		int beamWidth = Math
				.max(1, Math.min(numAttributes, Math.round(numAttributes * beamWidthPercentage)));
		return findBestRuleBottomUp(instances, labelIndices, predictedLabels, beamWidth, acceptEqual, useSeCo, inst, n_step, numericGeneralization, useRandom, instanceStatus);
	}
	
	public final MultiHeadRule findBestRuleBottomUp(final Instances instances,
											final LinkedHashSet<Integer> labelIndices,
											final Set<Integer> predictedLabels,
											final int beamWidth,
											final boolean acceptEqual,
											final boolean useSeCo,
											final int inst,
											final int n_step,
											final String numericGeneralization,
											final boolean useRandom,
											final boolean[] instanceStatus) throws Exception {
		Queue<Closure> bestClosures = new FixedPriorityQueue<>(beamWidth);
		boolean improved = true;
		sorted = sortNumericAttributes(instances, labelIndices);
		current_instance = inst;
		seco = useSeCo;
		boolean steps;
		Queue<Closure> bestClosureOverall = null;
		
		if (n_step != 0) {
			bestClosureOverall = new FixedPriorityQueue<Closure>(beamWidth);
		}
		
		// standard process if no n-step lookahead is used
		if (n_step == 0) {
			while (improved) {
				steps = false;
				improved = refineRuleBottomUp(instances, labelIndices, predictedLabels, bestClosures, acceptEqual, useSeCo, steps, bestClosureOverall, numericGeneralization, useRandom, instanceStatus);
				if (improved && DEBUG_STEP_BY_STEP_V) {
					System.out.println(
                        "Generalized rule conditions (beam width = " + beamWidth + "): " +
                                Arrays.toString(bestClosures.toArray()));
				}
			}
		} else {
			// start with -1, because the first improve step initializes a new rule
			int step = -1;
			
			// TODO for n-Step: to continue search after n-step is finished, set improved=true in refineRuleBottomUp, if the last step was an improvement
			// otherwise, search will always terminate after n steps
			while (improved) {
				steps = step < n_step;
				improved = refineRuleBottomUp(instances, labelIndices, predictedLabels, bestClosures, acceptEqual, useSeCo, steps, bestClosureOverall, numericGeneralization, useRandom, instanceStatus);
				step++;
				if (improved && DEBUG_STEP_BY_STEP_V) {
					System.out.println(
                        "Generalized rule conditions (beam width = " + beamWidth + "): " +
                                Arrays.toString(bestClosures.toArray()));
				}
			}
        }	

        MultiHeadRule bestRule = getBestRule(bestClosures);
        if (DEBUG_STEP_BY_STEP) {
            System.out.println("Found best rule: " + bestRule + "\n");
        }
        return bestRule;
	}
	
	private int current_instance = 0;
	private boolean seco = true;
	boolean newRule = true;
	int n = 0;
	private boolean previous_improved = false;
	
	public boolean refineRuleBottomUp(final Instances instances,
									  final LinkedHashSet<Integer> labelIndices,
            						  final Set<Integer> predictedLabels,
            						  final Queue<Closure> closures,
            						  final boolean acceptEqual,
            						  final boolean useSeCo,
            						  final boolean steps,
            						  final Queue<Closure> bestClosureOverAll,
            						  final String numericGeneralization,
            						  final boolean useRandom,
            						  final boolean[] instanceStatus) throws
			Exception {
		boolean improved = false;
		boolean better = false;
		
		for (Closure closure : beamWidthIterable(closures)) {
			if (closure == null || closure.refineFurther) {
				if (closure != null) {
					closure.refineFurther = false;
				}
				
				// new rule is an example transformed into a rule
				if (closure == null) {
					Closure refinedClosure;
					// evaluate the rule, no need to compare since body and head are initialized from new rule
					if (!useSeCo) {
						// refinedRule here is the new rule
						MultiHeadRule refinedRule = createMultiHeadRuleFromRandomInstance(instances, labelIndices, instanceStatus);
						refinedClosure = new Closure(refinedRule, null);
						multiLabelEvaluation.evaluate(instances, labelIndices, refinedClosure.rule, null);
					} else {
						// refinedRule here is the new rule						
						MultiHeadRule refinedRule = createMultiHeadRuleFromRandomInstance(instances, labelIndices, instanceStatus);
						refinedClosure = new Closure(refinedRule, null);
						multiLabelEvaluation.evaluate(instances, labelIndices, refinedClosure.rule, null);
					}
					
					if (refinedClosure != null) {
						improved |= closures.offer(refinedClosure);
					}
					newRule = true;
				}	
				// iterate over conditions of the rule
				if (closure != null) {
					
					if (!steps && bestClosureOverAll != null) {
						for (Closure cl : beamWidthIterable(bestClosureOverAll)) {
							if (cl != null) {
								closures.offer(cl);
							}
						}
						
						// return if after exactly n steps the rule shouldn't be further generalized, because the last generalization was not an improvement
						/* (!previous_improved) {
							return false;
						}
							*/
						bestClosureOverAll.clear();
					}
					
					
					// only iterate over the body since the head remains the same
					Iterator<Condition> c = closure.rule.getBody().iterator();
					while (c.hasNext()) {
						Condition cond = c.next();
						
						// choose a random condition to be removed
						if (useRandom) {
							int randIndex = random.nextInt(closure.rule.getBody().size());
							cond = closure.rule.getBody().get(randIndex);
						}
							
						// don't iterate if it's a label, redundant, can be removed
						if (!labelIndices.contains(cond.getAttr().index())) {
							int index = closure.rule.getBody().indexOf(cond);
							
								
								// can't change a value in the copy without changing the original value
								// instead remove and then readd condition with new value
								// order of conditions in the rule will change, but this doesn't have any effect
								MultiHeadRule refinedRule = (MultiHeadRule) closure.rule.copy();
								
								// remove condition
								if (cond.getAttr().isNumeric()) {
									// set minimum lower
									if (cond.cmp()==false) {
										try {
											
											double value = sorted.get(cond.getAttr()).get(sorted.get(cond.getAttr()).indexOf(cond.getValue()) - 1);
											int currentIndex = sorted.get(cond.getAttr()).indexOf(value);
											while (value==sorted.get(cond.getAttr()).get(sorted.get(cond.getAttr()).indexOf(cond.getValue()))) {
												currentIndex--;
												value = sorted.get(cond.getAttr()).get(currentIndex); 
											}
											// chose random smaller index between 0 and currentIndex											
											if (numericGeneralization.equals("random") && currentIndex > 0) {
												currentIndex -= random.nextInt(currentIndex+1); // at least 0, but at least 1 step smaller
												value = sorted.get(cond.getAttr()).get(currentIndex);
											}											
											if (currentIndex != 0) {
												refinedRule = (MultiHeadRule) refinedRule.generalizeNumeric(index, value);
											} else {											
												refinedRule = (MultiHeadRule) refinedRule.generalize(index);
											}											
										} catch(Exception e) {
											refinedRule = (MultiHeadRule) refinedRule.generalize(index);
										}
									}
									// set maximum higher
									if (cond.cmp()==true) {
										try {											
											double value = sorted.get(cond.getAttr()).get(sorted.get(cond.getAttr()).indexOf(cond.getValue()) + 1);
											int currentIndex = sorted.get(cond.getAttr()).indexOf(value);
											while (value==sorted.get(cond.getAttr()).get(sorted.get(cond.getAttr()).indexOf(cond.getValue()))) {
												currentIndex++;												
												value = sorted.get(cond.getAttr()).get(currentIndex);
											}
											if(numericGeneralization.equals("random") && currentIndex < sorted.get(cond.getAttr()).size() - 1) {
												// same
												int max = sorted.get(cond.getAttr()).size() - 1;
												currentIndex += random.nextInt(max - currentIndex + 1);
												value = sorted.get(cond.getAttr()).get(currentIndex);
											}
											if (currentIndex != sorted.get(cond.getAttr()).size() - 1) {
												refinedRule = (MultiHeadRule) refinedRule.generalizeNumeric(index, value);
											} else {									
												refinedRule = (MultiHeadRule) refinedRule.generalize(index);
											}												
										} catch(Exception e) {
											refinedRule = (MultiHeadRule) refinedRule.generalize(index);
										}
									}
								} else {
									refinedRule = (MultiHeadRule) refinedRule.generalize(index);
								}
								
								refinedRule.increaseGeneralizationCount();
								
								Closure refinedClosure = new Closure(refinedRule, closure != null ? closure.metaData : null);
								// evaluate the rule
								multiLabelEvaluation.evaluate(instances, labelIndices, refinedClosure.rule, closure != null ? closure.metaData : null);

								
								// if it's a new iteration and there are still n_steps left, the old rule will always be replaced by the best rule of the new iteration
								if (!improved && steps) {
									closures.remove(closure);
									if (!newRule) {
										bestClosureOverAll.offer(closure);
									}
									newRule = false;
	                			}
								
								// ruleComparison: > or >=
								// DO NOT USE !ACCEPT_EQUAL WITH BEAM WIDTH >1
								if (steps) {
									improved |= closures.offer(refinedClosure);
								} else {
									if (DEBUG_STEP_BY_STEP_V) {
										System.out.println("changed condition: " + cond);
										System.out.println("-> refinement candidate: " + refinedClosure);
									}
									better |= acceptEqual ? refinedClosure.rule.getRuleValue() >= closure.rule.getRuleValue() : refinedClosure.rule.getRuleValue() > closure.rule.getRuleValue();
									if (refinedClosure != null && (acceptEqual || better)) {
										improved |= closures.offer(refinedClosure);
									}
								}
								
								// return after first randomly removed condition
								if (useRandom) {
									return improved;
								}
							
						}
					}
				}			
			}
		}
		/*
		if (bestClosure == null) {
			
		}
		*/
		return improved;
	}
	
	
	/*
	 * creates a MultiHeadRule from a randomly chosen instance
	 * @param instances all uncovered instances
	 * @param labelIndices the indices of the possible labels for the Head
	 * @return the MultiHeadRule which represents the randomly chosen instance
	 */
	public MultiHeadRule createMultiHeadRuleFromRandomInstance(final Instances instances,
															   final LinkedHashSet<Integer> labelIndices,
															   final boolean[] instanceStatus) throws Exception {
		// choose a random instance
		int seed = 1;
		int i = 0;
		//TODO: initialize new Random(seed) only once, not in every function call, since all the same first numbers get checked every time
		random = new Random(seed);
		do {
			i = random.nextInt(instanceStatus.length);
			//System.out.println(i);
		}
		// true means not yet covered and can therefore be chosen
		while (instanceStatus[i]==false);
		
		int actualIndex = 0;
		for (int allIndex = 0; allIndex < i; allIndex++) {
			if (instanceStatus[allIndex]==true) {
				actualIndex++;
			}
		}
		
		//// TESTING: ALWAYS CHOSE IN THE SAME ORDER FOR THE SAME DATASET
		//i = 0;
		//// DELETE FOR REAL RESULTS
		
		Instance inst = instances.instance(actualIndex);
		
		if (!seco) {
			inst = instances.instance(current_instance);
		}

		// adapted from AveragingStrategy
		Instance wrappedInstance =
                inst instanceof DenseInstanceWrapper ? ((DenseInstanceWrapper) inst).getWrappedInstance() :
                        ((SparseInstanceWrapper) inst).getWrappedInstance();
		
		// create MultiHeadRule from chosen instance
		MultiHeadRule rule = new MultiHeadRule(multiLabelEvaluation.getHeuristic());
		
		// set head
		Head head = new Head();
		
		///// TESTING ONLY ONE LABEL SET
		/*
		int index = labelIndices.iterator().next();
		Attribute attribute = inst.attribute(index);
		double value = wrappedInstance.value(index);
		if (value != 0) {
			head.addCondition(new NominalCondition(toSeCoAttribute(attribute), value));
		}
		*/
		///// END OF TESTING
		
		
		for (int labelIndex : labelIndices) {
			
			Attribute attribute = inst.attribute(labelIndex);
			double value = wrappedInstance.value(labelIndex);
			
			if (!predictZero) {
				if (value!=0) {
					if (attribute.isNominal())
						head.addCondition(new NominalCondition(toSeCoAttribute(attribute), value));
					else if (attribute.isNumeric())
						head.addCondition(new NumericCondition(toSeCoAttribute(attribute), value));
					else
						throw new Exception("only numeric and nominal attributes supported !");
				}
			} else {
				if (attribute.isNominal())
					head.addCondition(new NominalCondition(toSeCoAttribute(attribute), value));
				else if (attribute.isNumeric())
					head.addCondition(new NumericCondition(toSeCoAttribute(attribute), value));
				else
					throw new Exception("only numeric and nominal attributes supported !");
			}
		}
		
		//////////////////
		/*
		if (head.size() == 0) {
			//System.out.println(instances.size());
			return createMultiHeadRuleFromRandomInstance(instances, labelIndices);
		}
		//if (instances.size() < 50) System.exit(0);
		*/
		 
		rule.setHead(head);
		
		///////////////////
		
		// set body
		final Instances dataset = (Instances) inst.dataset();
		
		final Enumeration<de.tu_darmstadt.ke.seco.models.Attribute> atts = dataset.enumerateAttributesWithoutClass();
		
		while (atts.hasMoreElements()) {
			final Attribute att = atts.nextElement();
			
			// don't add condition if it's a label
			if (!labelIndices.contains(att.index())) {			
				if (att.isNominal())
					rule.addCondition(new NominalCondition((de.tu_darmstadt.ke.seco.models.Attribute) att, inst.value(att)));
				// add two conditions (smaller and bigger) if attribute is numeric
				else if (att.isNumeric()) {
					double min = 0;
					double max = 0;
					double val = inst.value(att);
					boolean minFound = false;
					for (int j = 0; j < sorted.get(att).size(); j++) {
						if (sorted.get(att).get(j)>=val && !minFound) {
							min = sorted.get(att).get(j-1);
							minFound = true;
							if (j-1 != 0) {
								rule.addCondition(new NumericCondition((de.tu_darmstadt.ke.seco.models.Attribute) att, min, false));
							}
						}
						if (sorted.get(att).get(j)>val) {
							max = sorted.get(att).get(j);
							if (j != sorted.get(att).size() - 1) {
								rule.addCondition(new NumericCondition((de.tu_darmstadt.ke.seco.models.Attribute) att, max, true));
							}
							break;
						}
					}
					
					// use true for <=, false for >=
					// conditions are only added if they are not the lowest/highest value, in this case they would be deleted anyway
					
					
				} else
					throw new Exception("only numeric and nominal attributes supported !");
			}
		}
		
		return rule;
	}
	
	public HashMap<Attribute, ArrayList<Double>> sortNumericAttributes(final Instances instances,
																	final LinkedHashSet<Integer> labelIndices) {
		final Instance inst = instances.instance(0);
		final Instances dataset = (Instances) inst.dataset();   
		final Enumeration<de.tu_darmstadt.ke.seco.models.Attribute> atts = dataset.enumerateAttributesWithoutClass();
		HashMap<Attribute, ArrayList<Double>> sortedListForAttribute = new HashMap<Attribute, ArrayList<Double>>();
		while (atts.hasMoreElements()) {
			final Attribute att = atts.nextElement();
			if (!labelIndices.contains(att.index())) {
				if (att.isNumeric()) {
					ArrayList<Double> sortedAttribute = new ArrayList<Double>(instances.size());
					for (Instance in : instances) {
						sortedAttribute.add(in.value(att));
					}
					sortedAttribute.sort(new Comparator<Double>() {
						@Override
						public int compare(Double a, Double b) {
							if (a > b)
								return 1;
							else if (a < b)
								return -1;
							else
								return 0;
						}
					});
					ArrayList<Double> intervalsAttribute = new ArrayList<Double>(instances.size()+1);
					for (int index = 0; index < instances.size()-1; index++) {
						if (index == 0) {
							intervalsAttribute.add(index, sortedAttribute.get(index) - 1);
						}
						intervalsAttribute.add(index+1, (sortedAttribute.get(index) + sortedAttribute.get(index+1)) / 2);
						if (index == sortedAttribute.size()-2) {
							intervalsAttribute.add(sortedAttribute.get(index+1) + 1);
						}
					}
					if (instances.size() == 1) {
						intervalsAttribute.add(0, sortedAttribute.get(0) - 1);
						intervalsAttribute.add(1, sortedAttribute.get(0) + 1);
					}
					sortedListForAttribute.put(att, intervalsAttribute);
				}
			}
		}
		return sortedListForAttribute;
	}
	
    public MultiHeadRuleSet sortTheory (MultiHeadRuleSet theory, final Instances examples, final LinkedHashSet<Integer> labelIndices) {
    	MultiHeadRuleSet sortedTheory = new MultiHeadRuleSet();
    	ArrayList<MultiHeadRule> rules = new ArrayList<MultiHeadRule>(theory.size());
    	for (MultiHeadRule rule : theory) {
    		multiLabelEvaluation.evaluate(examples, labelIndices, rule, null);
    		rules.add(rule);
    	}
    	// sort by heuristic value over all examples
    	rules.sort(new Comparator<MultiHeadRule>() {
			@Override
			public int compare(MultiHeadRule a, MultiHeadRule b) {
				return b.compareTo(a);
			}
		});
    	sortedTheory.addAllRules(rules);
    	return sortedTheory;
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