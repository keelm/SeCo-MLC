package de.tu_darmstadt.ke.seco.algorithm;

import de.tu_darmstadt.ke.seco.algorithm.components.candidateselectors.CandidateSelector;
import de.tu_darmstadt.ke.seco.algorithm.components.candidateselectors.SelectAllCandidatesSelector;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.FMeasure;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.MEstimate;
import de.tu_darmstadt.ke.seco.algorithm.components.postprocessors.NoOpPostProcessor;
import de.tu_darmstadt.ke.seco.algorithm.components.postprocessors.PostProcessor;
import de.tu_darmstadt.ke.seco.algorithm.components.postprocessors.PostProcessorRipper;
import de.tu_darmstadt.ke.seco.algorithm.components.rulefilters.BeamWidthFilter;
import de.tu_darmstadt.ke.seco.algorithm.components.rulefilters.RuleFilter;
import de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers.RuleInitializer;
import de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers.TopDownRuleInitializer;
import de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners.RuleRefiner;
import de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners.TopDownRefiner;
import de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions.CoverageRuleStop;
import de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions.RuleStoppingCriterion;
import de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions.NoNegativesCoveredStop;
import de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions.StoppingCriterion;
import de.tu_darmstadt.ke.seco.algorithm.components.weightmodels.NoOpWeight;
import de.tu_darmstadt.ke.seco.algorithm.components.weightmodels.WeightModel;
import de.tu_darmstadt.ke.seco.models.Attribute;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.models.Random;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.DenseInstanceWrapper;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.JRipOneRuler;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.MulticlassCovering;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.SparseInstanceWrapper;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import de.tu_darmstadt.ke.seco.stats.RuleStats;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import de.tu_darmstadt.ke.seco.utils.Logger;
import de.tu_darmstadt.ke.seco.utils.SeCoLogger;
import weka.core.*;

import java.io.File;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.*;
import java.util.Map.Entry;

/**
 * @author eneldo
 */
public class SeCoAlgorithm implements Serializable {

    private static final long serialVersionUID = 2895086346619413864L;

    /**
     * the name of the classifier
     */
    private String name;

    private Hashtable<String, String> attributes;

    /**
     * used to determine the mode of the induction and the use of the rules
     */
    private final static int UNSORTED = 1;

    /**
     * used to determine the mode of the induction and the use of the rules
     */
    private final static int SORTED = 2;

    /**
     * The percentage size of the growing set. I.e. a value of 1 means "use 100% for growing (no pruning)", a value of
     * 0.8 means "use 80% for growing and 20% for pruning"
     */
    private double growingSetSize = 1;

    /**
     * The minimum number of examples a rule has to cover <p> TODO by m.zopf: should very likely be placed in another
     * class (maybe in the RuleFilter)
     */
    private int minNo = 1;

    /**
     * used for stratifying the folds of a cross validation in a random way <p> TODO by m.zopf: should be the only
     * source for randomness
     */
    private Random random = null;

    private boolean strictlyGreater = false;

    /**
     * a copy of the instances
     */
    private static Instances newInstances;

    private CandidateSelector candidateSelector;
    private Heuristic heuristic;
    private PostProcessor postProcessor;
    private RuleFilter ruleFilter;
    private RuleInitializer ruleInitializer;
    private RuleRefiner ruleRefiner;
    private RuleStoppingCriterion ruleStoppingCriterion;
    private StoppingCriterion stoppingCriterion;
    private WeightModel weightModel;

    public SeCoAlgorithm(String name) {
        this.name = name;
        initDefaultComponents();
    }

    private void initDefaultComponents() {
        candidateSelector = new SelectAllCandidatesSelector();
        heuristic = new MEstimate();
        postProcessor = new NoOpPostProcessor();
        ruleFilter = new BeamWidthFilter();
        ruleInitializer = new TopDownRuleInitializer();
        ruleRefiner = new TopDownRefiner();
        ruleStoppingCriterion = new CoverageRuleStop();
        stoppingCriterion = new NoNegativesCoveredStop();
        weightModel = new NoOpWeight();

    }

    private void setProperty(final String name, final String value) {
        if (name.equalsIgnoreCase("growingSetSize")) {
            double size;
            if (value.contains("/")) {
                final String[] fractionParts = value.replaceAll(" ", "").split("/");
                if (fractionParts.length != 2)
                    throw new NumberFormatException(
                            "Could not parse growingSetSize. growingSetSize was '" + value + "'.");
                else
                    size = Double.parseDouble(fractionParts[0]) / Double.parseDouble(fractionParts[1]);
            } else
                size = Double.parseDouble(value);

            if (size > 1)
                throw new IllegalArgumentException("growingSetSize must be <= 1. growingSetSize was '" + size + "'.");
            else
                growingSetSize = size;
        } else if (name.equalsIgnoreCase("minNo")) {
            Integer size = Integer.parseInt(value);
            if (size < 1)
                size = 1;
            minNo = size;
        } else if (name.equalsIgnoreCase("seed"))
            random = new Random(Long.parseLong(value));

        else if (name.equalsIgnoreCase("strictlyGreater"))
            strictlyGreater = Boolean.parseBoolean(value);
    }

    public void setAttributes(final Hashtable<String, String> attributes) {
        this.attributes = attributes;
        for (final Entry<String, String> attribute : attributes.entrySet())
            setProperty(attribute.getKey(), attribute.getValue());

        if (random == null) // if random was not initialized with a seed, generate a new Random with a random seed
            random = new Random(1);
    }

    public void setCandidateSelector(final CandidateSelector candidateSelector) {
        this.candidateSelector = candidateSelector;
    }

    public void setHeuristic(final Heuristic heuristic) {
        this.heuristic = heuristic;
    }

    public void setPostProcessor(final PostProcessor postProcessor) {
        this.postProcessor = postProcessor;
    }

    public void setRuleFilter(final RuleFilter ruleFilter) {
        this.ruleFilter = ruleFilter;
    }

    public void setRuleInitializer(final RuleInitializer ruleInitializer) {
        this.ruleInitializer = ruleInitializer;
    }

    public void setRuleRefiner(final RuleRefiner ruleRefiner) {
        this.ruleRefiner = ruleRefiner;
    }

    public void setRuleStoppingCriterion(final RuleStoppingCriterion ruleStoppingCriterion) {
        this.ruleStoppingCriterion = ruleStoppingCriterion;
    }

    public void setStoppingCriterion(final StoppingCriterion stoppingCriterion) {
        this.stoppingCriterion = stoppingCriterion;
    }

    public void setWeightModel(final WeightModel weightModel) {
        this.weightModel = weightModel;
    }

    public CandidateSelector getCandidateSelector() {
        return candidateSelector;
    }

    public Heuristic getHeuristic() {
        return heuristic;
    }

    public PostProcessor getPostProcessor() {
        return postProcessor;
    }

    public RuleFilter getRuleFilter() {
        return ruleFilter;
    }

    public RuleInitializer getRuleInitializer() {
        return ruleInitializer;
    }

    public RuleRefiner getRuleRefiner() {
        return ruleRefiner;
    }

    public RuleStoppingCriterion getRuleStoppingCriterion() {
        return ruleStoppingCriterion;
    }

    public StoppingCriterion getStoppingCriterion() {
        return stoppingCriterion;
    }

    public WeightModel getWeightModel() {
        return weightModel;
    }

    @Override
    public String toString() {
        final StringBuilder stringBuilder = new StringBuilder();
        if (attributes != null && attributes.size() != 0)
            stringBuilder.append("seCoAlgorithm........: " + attributes.toString() + "\n");

        stringBuilder.append("candidateSelector....: " + candidateSelector + "\n");
        stringBuilder.append("heuristic............: " + heuristic + "\n");
        stringBuilder.append("postProcessor........: " + postProcessor + "\n");
        stringBuilder.append("ruleFilter...........: " + ruleFilter + "\n");
        stringBuilder.append("ruleInitializer......: " + ruleInitializer + "\n");
        stringBuilder.append("ruleRefiner..........: " + ruleRefiner + "\n");
        stringBuilder.append("ruleStoppingCriterion: " + ruleStoppingCriterion + "\n");
        stringBuilder.append("stoppingCriterion....: " + stoppingCriterion + "\n");
        stringBuilder.append("weightModel..........: " + weightModel);
        return stringBuilder.toString();
    }

    public int getMinNo() {
        return minNo;
    }

    public void setMinNo(final int minNo) {
        this.minNo = minNo;
    }

    public Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        this.random = random;
    }

    /**
     * This will determine the best CandidateRule for the given training set. This procedure can grow a rule from an
     * empty rule or a given candidate rule.
     *
     * @param examples The growing examples.
     * @param r        The given candidate rule.
     * @return The best CandidateRule.
     */
    public SingleHeadRule findBestRule(final Instances examples, final SingleHeadRule r, final double classValue) throws
            Exception {
        Logger.debug("findBestRule(), initialize rule");
        TreeSet<SingleHeadRule> rules = new TreeSet<SingleHeadRule>(Collections.reverseOrder());
        // If there are already antecedents existing
        SingleHeadRule[] initRules = null;
        if (r != null) {
            initRules = new SingleHeadRule[1];
            initRules[0] = r;
        } else
            initRules = ruleInitializer.initializeRule(heuristic, examples, classValue);
        for (final SingleHeadRule initRule : initRules) {
            initRule.evaluateRuleForMultilabel(examples, classValue, heuristic);

            rules.add(initRule);
        }
        SingleHeadRule bestRule = rules.first();

        while (rules.size() > 0) {
            // System.out.println("rules in ruleset: " + rules);
            // Logger.debug("selecting candidates from " + rules.toString());
            final SingleHeadRuleSet candidateRules = candidateSelector.selectCandidates(rules, examples);
            removeRuleSetFromTreeSet(candidateRules, rules);

            for (int i = 0; i < candidateRules.numRules(); i++) {
                final SingleHeadRule candidateRule = candidateRules.getRule(i);

                // Stopping criterion if the candRule does not cover any
                // negatives it is not refined any longer

                // SHOULD THIS STAY IN???
                if (stoppingCriterion.checkForStop(candidateRule, examples) ||
                        candidateRule.getStats().getNumberOfTruePositives() < minNo)
                    continue;

                // Logger.debug("refining candidate " + c.toString());

                final SingleHeadRuleSet refinedRules = ruleRefiner.refineRule(candidateRule, examples, classValue);
                for (int j = 0; j < refinedRules.numRules(); j++) {

                    final SingleHeadRule refinedRule = refinedRules.getRule(j);
                    // if (refinements.size() > 1)
                    // System.out.println("more than one refinement!");
                    // m_ruleEvaluator.evaluateRule(refinement, examples);
                    // System.out.println("stats of refinement after: "+refinement.toString());
                    // Logger.debug("adding refinement " + refinement.toString());
                    // if refinement is the CandidateRule c, then don't add
                    if (refinedRule != candidateRule) {
                        // Forward Pruning
                        boolean fprune = true;
                        // get a copy of the current refinement
                        final SingleHeadRule currPruned = (SingleHeadRule) refinedRule.copy();
                        final double FP = currPruned.getStats().getNumberOfFalsePositives();
                        // create a virtual rule that does not loose a true
                        // positive but excludes all
                        // false positives, i.e., they become true negatives
                        currPruned.getStats().setNumberOfFalsePositives(0);
                        currPruned.getStats()
                                .setNumberOfTrueNegatives(currPruned.getStats().getNumberOfTrueNegatives() + FP);
                        // evaluate the virtual rule
                        currPruned.computeRuleValue(heuristic);
                        // if the virtual rule is better than the best rule
                        // add it because this means that the refinement can
                        // potentially be refined into a better rule than
                        // the current best one
                        if (bestRule.compareTo(currPruned) == 1)
                            fprune = false;

                        // some additional criterion determining if the rule
                        // is added to the rules list: if it does not cover
                        // enough positives and if forward pruning holds don't
                        // add
                        // TODO by m.zopf: should we use strictlyGreater here or should the refiner produce only strictlyGreater rules?
                        // if ((refinement.getStats().getNumberOfTruePositives() >= minNo && !stoppingCriterion.checkForStop(refinement, examples) && fprune && (!strictlyGreater || (refinement.compareTo(c) == 1))))
                        if ((fprune && (!strictlyGreater || (refinedRule.compareTo(candidateRule) == 1))))
                            rules.add(refinedRule);
                    }
                }
            }

            // VALUE HEURISTICS
            // for value heuristics, the best rule during the whole
            // refinement process is stored and returned but only if it
            // covers enough examples
            if (rules.size() > 0)
                if (heuristic.isValueHeuristic() && rules.first().getStats().getNumberOfTruePositives() >= minNo)

                    bestRule = (SingleHeadRule) Rule.getBetterRule(bestRule, rules.first());
                    // if ((m_stopCriterion.checkForStop(refinement, examples))) {
                    // break;
                    // }
                else // for gain heuristics, the last refinement is returned
                    if (stoppingCriterion.checkForStop(rules.first(), examples) &&
                            rules.first().getStats().getNumberOfTruePositives() >= minNo) {

                        rules.add(rules.first()); // TODO by m.zopf: why is the refinement added twice to the rules?
                        if (rules.size() > 0)
                            bestRule = rules.first();

                    }

            Logger.debug("filtering rules...");

            // System.out.println("rules in ruleset before filtering: " + rules);
            rules = ruleFilter.filterRules(rules, examples);

        }

        // System.out.println("best rule found: " + bestRule);
        return bestRule;
    }

    public SingleHeadRule findBestRuleForMultilabel(final Instances examples, final SingleHeadRule r,
                                                    final double classValue) throws Exception {
        Logger.debug("findBestRule(), initialize rule");
        TreeSet<SingleHeadRule> rules = new TreeSet<SingleHeadRule>(Collections.reverseOrder());
        // If there are already antecedents existing
        SingleHeadRule[] initRules = null;
        if (r != null) {
            initRules = new SingleHeadRule[1];
            initRules[0] = r;
        } else
            initRules = ruleInitializer.initializeRule(heuristic, examples, classValue);
        for (final SingleHeadRule initRule : initRules) {
            initRule.evaluateRuleForMultilabel(examples, classValue, heuristic);

            rules.add(initRule);
        }
        SingleHeadRule bestRule = rules.first();

        while (rules.size() > 0) {
            // System.out.println("rules in ruleset: " + rules);
            // Logger.debug("selecting candidates from " + rules.toString());
            final SingleHeadRuleSet candidateRules = candidateSelector.selectCandidates(rules, examples);
            removeRuleSetFromTreeSet(candidateRules, rules);

            for (int i = 0; i < candidateRules.numRules(); i++) {
                final SingleHeadRule candidateRule = candidateRules.getRule(i);

                // Stopping criterion if the candRule does not cover any
                // negatives it is not refined any longer

                // SHOULD THIS STAY IN???
//				if (stoppingCriterion.checkForStop(candidateRule, examples) || candidateRule.getStats().getNumberOfTruePositives() < minNo)
//					continue;

                // Logger.debug("refining candidate " + c.toString());

                final SingleHeadRuleSet refinedRules = ruleRefiner.refineRule(candidateRule, examples, classValue);
                for (int j = 0; j < refinedRules.numRules(); j++) {

                    final SingleHeadRule refinedRule = refinedRules.getRule(j);
                    // if (refinements.size() > 1)
                    // System.out.println("more than one refinement!");
                    // m_ruleEvaluator.evaluateRule(refinement, examples);
                    // System.out.println("stats of refinement after: "+refinement.toString());
                    // Logger.debug("adding refinement " + refinement.toString());
                    // if refinement is the CandidateRule c, then don't add
                    if (refinedRule != candidateRule) {
                        // Forward Pruning
                        boolean fprune = true;
                        // get a copy of the current refinement
                        final SingleHeadRule currPruned = (SingleHeadRule) refinedRule.copy();
                        final double FP = currPruned.getStats().getNumberOfFalsePositives();
                        // create a virtual rule that does not loose a true
                        // positive but excludes all
                        // false positives, i.e., they become true negatives
                        currPruned.getStats().setNumberOfFalsePositives(0);
                        currPruned.getStats()
                                .setNumberOfTrueNegatives(currPruned.getStats().getNumberOfTrueNegatives() + FP);
                        // evaluate the virtual rule
                        currPruned.computeRuleValue(heuristic);
                        // if the virtual rule is better than the best rule
                        // add it because this means that the refinement can
                        // potentially be refined into a better rule than
                        // the current best one
                        if (bestRule.compareTo(currPruned) == 1)
                            fprune = false;

                        // some additional criterion determining if the rule
                        // is added to the rules list: if it does not cover
                        // enough positives and if forward pruning holds don't
                        // add
                        // TODO by m.zopf: should we use strictlyGreater here or should the refiner produce only strictlyGreater rules?
                        // if ((refinement.getStats().getNumberOfTruePositives() >= minNo && !stoppingCriterion.checkForStop(refinement, examples) && fprune && (!strictlyGreater || (refinement.compareTo(c) == 1))))
                        if ((fprune && (!strictlyGreater || (refinedRule.compareTo(candidateRule) == 1))))
                            rules.add(refinedRule);
                    }
                }
            }

            // VALUE HEURISTICS
            // for value heuristics, the best rule during the whole
            // refinement process is stored and returned but only if it
            // covers enough examples
            if (rules.size() > 0)
                if (heuristic.isValueHeuristic() && rules.first().getStats().getNumberOfTruePositives() >= minNo)

                    bestRule = (SingleHeadRule) SingleHeadRule.getBetterRule(bestRule, rules.first());
                    // if ((m_stopCriterion.checkForStop(refinement, examples))) {
                    // break;
                    // }
                else // for gain heuristics, the last refinement is returned
                    if (stoppingCriterion.checkForStop(rules.first(), examples) &&
                            rules.first().getStats().getNumberOfTruePositives() >= minNo) {

                        rules.add(rules.first()); // TODO by m.zopf: why is the refinement added twice to the rules?
                        if (rules.size() > 0)
                            bestRule = rules.first();

                    }

            Logger.debug("filtering rules...");

            // System.out.println("rules in ruleset before filtering: " + rules);
            rules = ruleFilter.filterRules(rules, examples);

        }

        // System.out.println("best rule found: " + bestRule);
        return bestRule;
    }

    /**
     * Prunes a rule after it was learned. The code is adapted from JRip, the weka implementation of the RIPPER
     * algorithm. The pruning measures used in the building phase and postprocessing phase are different. The parameter
     * useWhole is a flag to indicate the pruning measure.
     *
     * @param examples The pruning examples.
     * @param r        The candidate rule.
     * @param useWhole false: calculates the error rate of the examples covered true: calculates the error rate of the
     *                 whole pruning examples
     * @return the pruned candidate rule.
     */
    public SingleHeadRule pruneRule(final Instances examples, final SingleHeadRule r, final boolean useWhole,
                                    final double classValue) {

        Instances pruningExamples = examples;
        final ArrayList<Condition> m_Antds = r.getBody();

        final double total = pruningExamples.sumOfWeights();

        // defAccu is the maximum number of instances in
        // the pruning data that can be covered by rule r
        double defAccu = 0;

        try {
            defAccu = computeDefAccu(pruningExamples, classValue);
        } catch (final UnassignedClassException e1) {

            e1.printStackTrace();
        } catch (final UnassignedDatasetException e1) {

            e1.printStackTrace();
        }

        final int size = m_Antds.size();
        if (size == 0)
            return r; // Default rule before pruning

        // coverage is tp+fp
        // worthValue is tp
        // worthRt is tp+1/tp+fp+2
        final double[] coverage = new double[size];
        final double[] worthValue = new double[size];
        final double[] worthRt = new double[size];

        for (int w = 0; w < size; w++)
            worthRt[w] = coverage[w] = worthValue[w] = 0.0;

		/*
         * Calculate accuracy parameters for all the antecedents in this rule
		 */
        double tn = 0.0; // True negative value if useWhole is used

        for (int x = 0; x < size; x++) {
            final Condition antd = m_Antds.get(x);
            final Instances newPruningExamples = pruningExamples;
            // Make pruningExamples empty
            pruningExamples = new Instances(newPruningExamples, 0);

            for (int y = 0; y < newPruningExamples.numInstances(); y++) {
                final Instance ins = newPruningExamples.instance(y);

                // Covered by this antecedent
                if (antd.covers(ins)) {
                    coverage[x] += ins.weight();
                    // Add to data for further pruning
                    pruningExamples.add(ins);
                    try {
                        // TODO by m.zopf: casting classValue to int will result in false behavior if we have a classValue like 1.5?
                        if ((int) ins.classValue() == (int) classValue) // Accurate prediction
                            worthValue[x] += ins.weight();
                    } catch (final UnassignedClassException e) {

                        e.printStackTrace();
                    } catch (final UnassignedDatasetException e) {

                        e.printStackTrace();
                    }
                } else if (useWhole)
                    try {
                        if ((int) ins.classValue() != (int) classValue)
                            tn += ins.weight();
                    } catch (final UnassignedClassException e) {

                        e.printStackTrace();
                    } catch (final UnassignedDatasetException e) {

                        e.printStackTrace();
                    }
            }
            if (useWhole) {

                // worthValue is tp+tn
                worthValue[x] += tn;
                // worthRt is tp+tn / total
                worthRt[x] = worthValue[x] / total;

            } else
                // Note if coverage is 0, accuracy is 0.5
                worthRt[x] = (worthValue[x] + 1.0) / (coverage[x] + 2.0);

        }

        double maxValue = (defAccu + 1.0) / (total + 2.0);

        int maxIndex = -1;
        for (int i = 0; i < worthRt.length; i++)
            if (worthRt[i] > maxValue) { // Prefer to the
                maxValue = worthRt[i]; // shorter rule
                maxIndex = i;
            }

		/* Prune the antecedents according to the accuracy parameters */
        for (int z = size - 1; z > maxIndex; z--)
            m_Antds.remove(z);

        // save antds back to rule
        r.initBody();
        for (final Condition c : m_Antds)
            r.addCondition(c);

        return r;

    }

    /**
     * Prunes a rule after it was learned. The code is adapted from JRip, the weka implementation of the RIPPER
     * algorithm. The pruning measures used in the building phase and postprocessing phase are different. The parameter
     * useWhole is a flag to indicate the pruning measure.
     *
     * @param examples The pruning examples.
     * @param r        The candidate rule.
     * @param useWhole false: calculates the error rate of the examples covered true: calculates the error rate of the
     *                 whole pruning examples
     * @return the pruned candidate rule.
     */
    public SingleHeadRule pruneRuleForMultilabel(final Instances examples, final SingleHeadRule r,
                                                 final boolean useWhole, final double classValue) {

        Instances pruningExamples = examples;
        final ArrayList<Condition> m_Antds = r.getBody();

        final double total = pruningExamples.sumOfWeights();

        // defAccu is the maximum number of instances in
        // the pruning data that can be covered by rule r
        double defAccu = 0;

        try {
            defAccu = computeDefAccu(pruningExamples, classValue);
        } catch (final UnassignedClassException e1) {

            e1.printStackTrace();
        } catch (final UnassignedDatasetException e1) {

            e1.printStackTrace();
        }

        final int size = m_Antds.size();
        if (size == 0)
            return r; // Default rule before pruning

        // coverage is tp+fp
        // worthValue is tp
        // worthRt is tp+1/tp+fp+2
        final double[] coverage = new double[size];
        final double[] worthValue = new double[size];
        final double[] worthRt = new double[size];

        for (int w = 0; w < size; w++)
            worthRt[w] = coverage[w] = worthValue[w] = 0.0;

		/*
         * Calculate accuracy parameters for all the antecedents in this rule
		 */
        double tn = 0.0; // True negative value if useWhole is used

        for (int x = 0; x < size; x++) {
            final Condition antd = m_Antds.get(x);
            final Instances newPruningExamples = pruningExamples;
            // Make pruningExamples empty
            pruningExamples = new Instances(newPruningExamples, 0);

            for (int y = 0; y < newPruningExamples.numInstances(); y++) {
                final Instance ins = newPruningExamples.instance(y);

                // Covered by this antecedent
                if (antd.covers(ins)) {
                    coverage[x] += ins.weight();
                    // Add to data for further pruning
                    pruningExamples.addDirectly(ins);
                    try {
                        // TODO by m.zopf: casting classValue to int will result in false behavior if we have a classValue like 1.5?
                        if ((int) ins.classValue() == (int) classValue) // Accurate prediction
                            worthValue[x] += ins.weight();
                    } catch (final UnassignedClassException e) {

                        e.printStackTrace();
                    } catch (final UnassignedDatasetException e) {

                        e.printStackTrace();
                    }
                } else if (useWhole)
                    try {
                        if ((int) ins.classValue() != (int) classValue)
                            tn += ins.weight();
                    } catch (final UnassignedClassException e) {

                        e.printStackTrace();
                    } catch (final UnassignedDatasetException e) {

                        e.printStackTrace();
                    }
            }
            if (useWhole) {

                // worthValue is tp+tn
                worthValue[x] += tn;
                // worthRt is tp+tn / total
                worthRt[x] = worthValue[x] / total;

            } else
                // Note if coverage is 0, accuracy is 0.5
                worthRt[x] = (worthValue[x] + 1.0) / (coverage[x] + 2.0);

        }

        double maxValue = (defAccu + 1.0) / (total + 2.0);

        int maxIndex = -1;
        for (int i = 0; i < worthRt.length; i++)
            if (worthRt[i] > maxValue) { // Prefer to the
                maxValue = worthRt[i]; // shorter rule
                maxIndex = i;
            }

		/* Prune the antecedents according to the accuracy parameters */
        for (int z = size - 1; z > maxIndex; z--)
            m_Antds.remove(z);

        // save antds back to rule
        r.initBody();
        for (final Condition c : m_Antds)
            r.addCondition(c);

        return r;

    }

    /**
     * Removes all rules of the rset from the TreeSet.
     *
     * @param rset The set of rules that is to be removed.
     * @param tset The TreeSet that is to be modified.
     */
    protected void removeRuleSetFromTreeSet(final SingleHeadRuleSet rset, final TreeSet<SingleHeadRule> tset) {
        for (int i = 0; i < rset.numRules(); i++) {
            final SingleHeadRule r = rset.getRule(i);
            tset.remove(r);
        }
    }

    /**
     * Private function to compute default number of accurate instances in the specified data for the consequent of the
     * rule
     *
     * @param data The data in question.
     * @return The default accuracy.
     * @throws UnassignedDatasetException
     * @throws UnassignedClassException
     */
    private double computeDefAccu(final Instances data, final double classValue) throws UnassignedClassException,
            UnassignedDatasetException {
        double defAccu = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            final Instance inst = data.instance(i);
            if ((int) inst.classValue() == (int) classValue)
                defAccu += inst.weight();
        }
        return defAccu;
    }

    /**
     * The Separate and Conquer algorithm..
     *
     * @param examples The training set.
     * @return The theory that has been learned.
     */
    public SingleHeadRuleSet separateAndConquer(Instances examples, final double classValue) throws Exception {
        // System.out.println("entering separateAndConquer for class attribute: " + examples.classAttribute());
        // newExamples used only in postprocessor
        final Instances newExamples = examples;
        Instances tempExamples = examples;
        // for counting how many rules cover a example
        Instances m_covered = new Instances(examples);

        m_covered.setWeightsTo0();

        Instances growingExamples, pruningExamples;
        SingleHeadRule r;

        SingleHeadRuleSet theory = new SingleHeadRuleSet();

        while (examples.containsPositive(classValue)) {
            // System.out.println("number of examples: " + examples.size() + "; number of positive examples: " + examples.countInstances(classValue));

            if (growingSetSize != 1) {
                tempExamples = examples;
                examples = RuleStats.stratify(examples, growingSetSize, random);
                // use growing and pruning set
                final SplittedInstances splitInst = new SplittedInstances(examples);
                splitInst.splitInstances(growingSetSize);
                growingExamples = splitInst.getGrowingSet();
                pruningExamples = splitInst.getPruningSet();

                r = findBestRule(growingExamples, null, classValue);

                r.evaluateRule(examples, classValue, heuristic);

                r = pruneRule(pruningExamples, r, false, classValue);
                // and recompute stats
                r.evaluateRule(examples, classValue, heuristic);
            } else
                r = findBestRule(examples, null, classValue);

            if (ruleStoppingCriterion.checkForRuleStop(theory, r, examples, m_covered, classValue, this)) {
                Logger.debug("SingleHeadRule stop !");
                if (growingSetSize != 1)
                    newInstances = tempExamples;
                else
                    newInstances = examples;
                break;
            }

            // here takes the weighting place
            final Instances[] m_container = weightModel.changeWeights(examples, m_covered, r);
            examples = m_container[0];
            m_covered = m_container[1];

            examples = r.uncoveredInstances(
                    examples); // TODO by m.zopf: not just uncovered instances must be removed, also unimportant instances (following the weight model) must be removed

            newInstances = examples;
            theory.addRule(r);
            // System.out.println("best rule found: " + r);
            // System.out.println("uncoverd instances: " + newInstances.size());
        }

        // perform post processing by cloning the AbstractSeco and run the post processor
        if (postProcessor != null && growingSetSize != 1) {
            Logger.debug("performing post processing...");
            if (postProcessor instanceof NoOpPostProcessor) {
                final NoOpPostProcessor dpp = (NoOpPostProcessor) postProcessor;
                dpp.clone(this);
            } else if (postProcessor instanceof PostProcessorRipper) {
                final PostProcessorRipper dpp = (PostProcessorRipper) postProcessor;
                dpp.clone(this);
            }
            theory = postProcessor.postProcessTheory(theory, newExamples, classValue);

        }

        // Logger.info("separate and conquer finished for class " + classValue);
        // Logger.info("Theory:\n" + theory.toString());

        // System.out.println("theory complete: " + theory);
        // System.out.println("number of examples: " + examples.size() + "; number of positive examples: " + examples.countInstances(classValue));
        return theory;
    }

    /**
     * Setter for the instances
     *
     * @param examples The instances to be set.
     */
    public void setInstances(final Instances examples) {
        newInstances = examples;
    }

    public Instances getNewInstances() {
        return newInstances;
    }

    public double getGrowingSetSize() {
        return growingSetSize;
    }

    public void setGrowingSetSize(final double growingSetSize) {
        this.growingSetSize = growingSetSize;
    }

    public String getName() {
        return name;
    }

    // multiclass:

    public SingleHeadRuleSet separateAndConquerUnordered(Instances examples) throws Exception {
        SeCoLogger.debug("entering separateAndConquer");
        // newExamples used only in postprocessor
        Instances newExamples = examples;
        Instances tempExamples = examples;
        // for counting how many rules cover a example
        Instances m_covered = new Instances(examples);
        m_covered.setWeightsTo0();
        Instances growingExamples, pruningExamples;
        SingleHeadRule r = null;
        SingleHeadRule bestRuleOfMulti = null;
        SingleHeadRuleSet theory = new SingleHeadRuleSet();
        int TrainingDataSize = examples.getInstances().size();

        double classToLearn;

        outerloop:
        while (examples.getInstances().size() > TrainingDataSize * noNeedForClassification) {
            bestRuleOfMulti = null;
            for (double val : examples.getDistinctClassValues(true, false)) {
                classToLearn = val;
                if (growingSetSize != 1) {
                    tempExamples = examples;
                    examples = RuleStats.stratify(examples, growingSetSize, random);
                    // use growing and pruning set
                    SplittedInstances splitInst = new SplittedInstances(examples);
                    splitInst.splitInstances(growingSetSize);
                    growingExamples = splitInst.getGrowingSet();
                    pruningExamples = splitInst.getPruningSet();

                    r = findBestRule(growingExamples, null, classToLearn);

                    r.evaluateRule(examples, classToLearn, heuristic);

                    r = pruneRule(pruningExamples, r, false, classToLearn);
                    // and recompute stats
                    r.evaluateRule(examples, classToLearn, heuristic);
                } else
                    r = findBestRule(examples, null, classToLearn);

                // System.out.println("best rule found for class '" + val + "':" + r);

                if (bestRuleOfMulti == null)
                    bestRuleOfMulti = r;
                else if (bestRuleOfMulti.compareTo(r) < 0)
                    bestRuleOfMulti = r;
            }

            // System.out.println("bestRuleOfMulti found:" + bestRuleOfMulti);

            if (ruleStoppingCriterion
                    .checkForRuleStop(theory, bestRuleOfMulti, examples, m_covered, bestRuleOfMulti.getPredictedValue(),
                            this)) {
                Logger.debug("SingleHeadRule stop !");
                if (growingSetSize != 1)
                    newInstances = tempExamples;
                else
                    newInstances = examples;
                break;
            }
            // if the rule has no condition, then it will be learned via
            // setDefaultRule call in createClassifier
            if (bestRuleOfMulti.length() == 0)
                break outerloop;

            examples = bestRuleOfMulti.uncoveredInstances(examples);

            // System.out.println("remaining examples: " + examples.size());

            newInstances = examples;
            theory.addRule(bestRuleOfMulti);
        }

        return theory;
    }

    public SingleHeadRuleSet separateAndConquerMultiClass(Instances examples) throws Exception {
        SeCoLogger.debug("entering separateAndConquer");
        // newExamples used only in postprocessor
        Instances newExamples = examples;
        Instances tempExamples = examples;
        // for counting how many rules cover a example
        Instances m_covered = new Instances(examples);
        m_covered.setWeightsTo0();
        Instances growingExamples, pruningExamples;
        SingleHeadRule r = null;
        SingleHeadRule bestRuleOfMulti = null;
        SingleHeadRuleSet theory = new SingleHeadRuleSet();
        int TrainingDataSize = examples.getInstances().size();

        double classToLearn;

        outerloop:
        while (examples.getInstances().size() > TrainingDataSize * noNeedForClassification) {
            bestRuleOfMulti = null;
            for (double val : examples.getDistinctClassValues(true, false)) {
                classToLearn = val;
                if (growingSetSize != 1) {
                    tempExamples = examples;
                    examples = RuleStats.stratify(examples, growingSetSize, random);
                    // use growing and pruning set
                    SplittedInstances splitInst = new SplittedInstances(examples);
                    splitInst.splitInstances(growingSetSize);
                    growingExamples = splitInst.getGrowingSet();
                    pruningExamples = splitInst.getPruningSet();

                    r = findBestRule(growingExamples, null, classToLearn);

                    r.evaluateRule(examples, classToLearn, heuristic);

                    r = pruneRule(pruningExamples, r, false, classToLearn);
                    // and recompute stats
                    r.evaluateRule(examples, classToLearn, heuristic);
                } else
                    r = findBestRule(examples, null, classToLearn);

                // System.out.println("best rule found for class '" + val + "':" + r);

                if (bestRuleOfMulti == null)
                    bestRuleOfMulti = r;
                else if (bestRuleOfMulti.compareTo(r) < 0)
                    bestRuleOfMulti = r;
            }

            // System.out.println("bestRuleOfMulti found:" + bestRuleOfMulti);

            if (ruleStoppingCriterion
                    .checkForRuleStop(theory, bestRuleOfMulti, examples, m_covered, bestRuleOfMulti.getPredictedValue(),
                            this)) {
                Logger.debug("SingleHeadRule stop !");
                if (growingSetSize != 1)
                    newInstances = tempExamples;
                else
                    newInstances = examples;
                break;
            }
            // if the rule has no condition, then it will be learned via
            // setDefaultRule call in createClassifier
            if (bestRuleOfMulti.length() == 0)
                break outerloop;

            examples = bestRuleOfMulti.uncoveredInstances(examples);

            // System.out.println("remaining examples: " + examples.size());

            newInstances = examples;
            theory.addRule(bestRuleOfMulti);
        }

        return theory;
    }


    protected double noNeedForClassification = 0.05;


    public double getUncoveredInstancesPercentage() {
        return noNeedForClassification;
    }

    public void setUncoveredInstancesPercentage(double noNeedForClassification) {
        this.noNeedForClassification = noNeedForClassification;
    }

    public boolean readdAllCovered = false;

    public boolean isPredictZero() {
        return predictZero;
    }

    public void setPredictZero(boolean predictZero) {
        this.predictZero = predictZero;
    }

    public boolean isReaddAllCovered() {
        return readdAllCovered;
    }

    public void setReaddAllCovered(boolean readdAllCovered) {
//		this.predictZero = predictZero;
        this.readdAllCovered = readdAllCovered;
    }

    private boolean useMultilabelHeads = false;

    public void setUseMultilabelHeads(boolean useMultilabelHeads) {
        this.useMultilabelHeads = useMultilabelHeads;
    }

    public boolean areMultilabelHeadsUsed() {
        return useMultilabelHeads;
    }

    private String beamWidth = "1";

    public void setBeamWidth(String beamWidth) {
        this.beamWidth = beamWidth;
    }

    public String getBeamWidth() {
        return beamWidth;
    }

    private String evaluationStrategy = EvaluationStrategy.RULE_DEPENDENT;

    public void setEvaluationStrategy(String evaluationStrategy) {
        this.evaluationStrategy = evaluationStrategy;
    }

    public String getEvaluationStrategy() {
        return evaluationStrategy;
    }

    private String averagingStrategy = AveragingStrategy.MICRO_AVERAGING;

    public void setAveragingStrategy(String averagingStrategy) {
        this.averagingStrategy = averagingStrategy;
    }

    public String getAveragingStrategy() {
        return averagingStrategy;
    }

    public double getSkipThresholdPercentage() {
        return skipThresholdPercentage;
    }

    public void setSkipThresholdPercentage(double skipThresholdPercentage) {
        if (skipThresholdPercentage < 0) {
            useSkippingRules = false;
            this.skipThresholdPercentage = -1;
        } else {
            useSkippingRules = true;
            this.skipThresholdPercentage = skipThresholdPercentage;
        }
    }

    private boolean useBottomUp = true;
    
    public void setUseBottomUp(boolean useBottomUp) {
    	this.useBottomUp = useBottomUp;
    }
    
    public boolean isBottomUpUsed() {
    	return useBottomUp;
    }
    
    private boolean acceptEqual = true;
    
    public void setAcceptEqual(boolean acceptEqual) {
    	this.acceptEqual = acceptEqual;
    }
    
    public boolean isEqualAccepted() {
    	return acceptEqual;
    }
    
    private boolean useSeCo = true;
   
    public void setUseSeCo(boolean useSeCo) {
    	this.useSeCo = useSeCo;
    }
    
    public boolean isSeCoUsed() {
    	return useSeCo;
    }    
    
    private int n_step = 1;
    
    public void setNStep(int n_step) {
    	this.n_step = n_step;
    }
    
    public int getNStep() {
    	return n_step;
    }
    
    private boolean useRandom = false;
    
    public boolean isRandomUsed() {
		return useRandom;
	}

	public void setUseRandom(boolean useRandom) {
		this.useRandom = useRandom;
	}

	private double betaValue = 0.5;
	
	public void setBetaValue(double betaValue) {
		this.betaValue = betaValue;
	}
	
	public double getBetaValue() {
		return betaValue;
	}
	
	private String numericGeneralization = "random"; 
	
	public void setNumericGeneralization(String numGen) {
		this.numericGeneralization = numGen;
	}
	
	public String getNumericGeneralization() {
		return numericGeneralization;
	}
	
	private boolean coverAllLabels = false;
	
	public void setCoverAllLabels(boolean cover) {
		this.coverAllLabels = cover;
	}
	
	public boolean getAllLabelsCovered() {
		return coverAllLabels;
	}
	
	String evaluationMethod = "DNF";
	
	public String getEvaluationMethod() {
		return evaluationMethod;
	}
	
	public void setEvaluationMethod(String evaluationMethod) {
		this.evaluationMethod = evaluationMethod;
	}

	boolean predictZero = true;

    boolean useSkippingRules = true;

    double skipThresholdPercentage = 0.00;

    private JRipOneRuler ripper;


    public static boolean DEBUG_STEP_BY_STEP = true;
    public static boolean DEBUG_STEP_BY_STEP_V = false;

    public RuleSet<?> separateAndConquerMultilabel(Instances examples, int labelIndices[]) throws Exception {
        if (useMultilabelHeads) {
            return multiclassCoveringSeparateAndConquerMultilabel(examples, labelIndices);
        } else {
            return standardSeparateAndConquerMultilabel(examples, labelIndices);
        }
    }
    
    public MultiHeadRuleSet multiclassCoveringSeparateAndConquerMultilabel(Instances examples,
                                                                           int labelIndices[]) throws Exception {
        SeCoLogger.debug("entering separateAndConquerMultilabel");
        LinkedHashSet<Integer> labelIndicesAsSet = new LinkedHashSet<>(labelIndices.length);
        Arrays.stream(labelIndices).forEach(labelIndicesAsSet::add);
        Instances originalExamples = examples; // newExamples used only in postprocessor
        examples = new Instances(originalExamples,
                originalExamples.numInstances()); //so that I can do what I want on this
        ArrayList<Instance> examplesReferences = null; // only used for debugging
        
        if (DEBUG_STEP_BY_STEP)
            examplesReferences = new ArrayList<>();

        for (int i = 0; i < originalExamples.size(); i++) {
            Instance inst = originalExamples.get(i);
            Instance wrappedInstance;

            if (inst instanceof SparseInstance) {
                wrappedInstance = new SparseInstanceWrapper(inst, labelIndices);
            } else {
                wrappedInstance = new DenseInstanceWrapper(inst, labelIndices);
            }

            examples.addDirectly(wrappedInstance); //now secured

            if (DEBUG_STEP_BY_STEP)
                examplesReferences.add(wrappedInstance);
        }

        
        if (betaValue!=-1) {
        	setHeuristic(new FMeasure(betaValue));
        }
        
        MultiHeadRule r;
        MultiHeadRule bestRuleOfMulti;
        MultiHeadRuleSet theory = new MultiHeadRuleSet();
        theory.setLabelIndices(labelIndices); //so that tostring prints out mlc statistics
        int trainingDataSize = examples.getInstances().size();
        boolean[] instanceStatus = new boolean[trainingDataSize];
        Arrays.fill(instanceStatus, true);
        Set<Integer> predictedLabelIndices = new HashSet<>();
        
        int count = examples.getInstances().size();
        
        if (!isSeCoUsed()) {
        	while (count > trainingDataSize * noNeedForClassification) {
        		bestRuleOfMulti = null;
        		EvaluationStrategy evaluationStrategy = EvaluationStrategy.create(getEvaluationStrategy());
                AveragingStrategy averagingStrategy = AveragingStrategy.create(getAveragingStrategy());
                MultiLabelEvaluation multiLabelEvaluation = new MultiLabelEvaluation(getHeuristic(), evaluationStrategy,
                        averagingStrategy);
                MulticlassCovering multiclassCovering = new MulticlassCovering(multiLabelEvaluation, isPredictZero());
                
                if (useBottomUp) {
                	try {
                        int beamWidth = Integer.valueOf(getBeamWidth());
                        bestRuleOfMulti = multiclassCovering
                                .findBestRuleBottomUp(examples, labelIndicesAsSet, predictedLabelIndices, beamWidth, isEqualAccepted(), isSeCoUsed(), trainingDataSize - count, n_step, getNumericGeneralization(), useRandom, instanceStatus);
                    } catch (NumberFormatException e) {
                        float beamWidthPercentage = Float.valueOf(getBeamWidth());
                        bestRuleOfMulti = multiclassCovering
                                .findBestRuleBottomUp(examples, labelIndicesAsSet, predictedLabelIndices, beamWidthPercentage, isEqualAccepted(), isSeCoUsed(), trainingDataSize - count, n_step, getNumericGeneralization(), useRandom, instanceStatus);
                    }
                } else {
                	try {
                		int beamWidth = Integer.valueOf(getBeamWidth());
                		bestRuleOfMulti = multiclassCovering
                            .findBestGlobalRule(examples, labelIndicesAsSet, predictedLabelIndices, beamWidth);
                	} catch (NumberFormatException e) {
                		float beamWidthPercentage = Float.valueOf(getBeamWidth());
                		bestRuleOfMulti = multiclassCovering
                				.findBestGlobalRule(examples, labelIndicesAsSet, predictedLabelIndices, beamWidthPercentage);
                	}
                }
                theory.addRule(bestRuleOfMulti);
                count--;
        	}
        	return theory;
        }
        
        // Continue until a certain percentage of the training data is covered
        outerloop:
        while (examples.getInstances().size() > trainingDataSize * noNeedForClassification) {
            bestRuleOfMulti = null;

            if (DEBUG_STEP_BY_STEP) {
                System.out.println("########remaining training set (" + examples.size() + ")");
                if (DEBUG_STEP_BY_STEP_V) for (Instance inst : examples) System.out.println(inst);
                else System.out.println(examples.size());
                System.out.println("########candidate rules");
            }

            EvaluationStrategy evaluationStrategy = EvaluationStrategy.create(getEvaluationStrategy());
            AveragingStrategy averagingStrategy = AveragingStrategy.create(getAveragingStrategy());
            MultiLabelEvaluation multiLabelEvaluation = new MultiLabelEvaluation(getHeuristic(), evaluationStrategy,
                    averagingStrategy);
            MulticlassCovering multiclassCovering = new MulticlassCovering(multiLabelEvaluation, isPredictZero());
            
            if (useBottomUp) {
            	try {
                    int beamWidth = Integer.valueOf(getBeamWidth());
                    bestRuleOfMulti = multiclassCovering
                            .findBestRuleBottomUp(examples, labelIndicesAsSet, predictedLabelIndices, beamWidth, isEqualAccepted(), isSeCoUsed(), 0, getNStep(), getNumericGeneralization(), useRandom, instanceStatus);
                } catch (NumberFormatException e) {
                    float beamWidthPercentage = Float.valueOf(getBeamWidth());
                    bestRuleOfMulti = multiclassCovering
                            .findBestRuleBottomUp(examples, labelIndicesAsSet, predictedLabelIndices, beamWidthPercentage, isEqualAccepted(), isSeCoUsed(), 0, getNStep(), getNumericGeneralization(), useRandom, instanceStatus);
                }
            } else {
            	try {
            		int beamWidth = Integer.valueOf(getBeamWidth());
            		bestRuleOfMulti = multiclassCovering
                        .findBestGlobalRule(examples, labelIndicesAsSet, predictedLabelIndices, beamWidth);
            	} catch (NumberFormatException e) {
            		float beamWidthPercentage = Float.valueOf(getBeamWidth());
            		bestRuleOfMulti = multiclassCovering
            				.findBestGlobalRule(examples, labelIndicesAsSet, predictedLabelIndices, beamWidthPercentage);
            	}
            }

            if (bestRuleOfMulti != null) {
                ArrayList<Instance> coveredInstances = bestRuleOfMulti.coveredInstances(examples);
                ArrayList<Instance> coveredButLabelsNotFullyCoveredInstances = new ArrayList<Instance>();
                examples = bestRuleOfMulti.uncoveredInstances(examples); //maintain these in any case

                if (DEBUG_STEP_BY_STEP) {
                    System.out.println("########uncovered by rule (" + examples.size() + ")");
                    if (DEBUG_STEP_BY_STEP_V) for (Instance inst : examples) System.out.println(inst);
                    else System.out.println(examples.size());
                }

                Head head = bestRuleOfMulti.getHead();

                for (Instance covered : coveredInstances) {
                    for (Map.Entry<Integer, Condition> entry : head.entries()) {
                        int labelIndex = entry.getKey();
                        predictedLabelIndices.add(labelIndex);

                        if (Utils.isMissingValue(covered.value(labelIndex))) {
                            covered.setValue(labelIndex, entry.getValue().getValue());
                        }
                    }

                    if (predictZero) {
                        int uncoveredLabels = getUncoveredLabels(covered, labelIndices);

                        if (uncoveredLabels > 0) {
                            coveredButLabelsNotFullyCoveredInstances.add(covered);
                        }
                    } else {
                        int uncoveredPosLabels = getUncoveredPosLabels(covered, labelIndices);

                        if (uncoveredPosLabels > 0) {
                            // there are still label attributes to fill up, so continue
                            coveredButLabelsNotFullyCoveredInstances.add(covered);
                        }
                    }
                }

                // TODO: make this more efficient! + Testing
                // true means, the instance is not yet covered by a rule
                for (int i = 0; i < instanceStatus.length; i++) {
                	if (instanceStatus[i] == true) {
                		if (bestRuleOfMulti.covers(originalExamples.get(i))) {
                			instanceStatus[i] = false;
                		}
                	}
                	System.out.println(instanceStatus[i]);
                }
                
                theory.addRule(bestRuleOfMulti);
                if (DEBUG_STEP_BY_STEP) {
                    System.out.println(
                            "########covered by rule (and predicted written) (" + coveredInstances.size() + ")");
                    if (DEBUG_STEP_BY_STEP_V) for (Instance inst : coveredInstances) System.out.println(inst);
                    else System.out.println(coveredInstances.size());
                    System.out.println(
                            "########readdition candidates (" + coveredButLabelsNotFullyCoveredInstances.size() + ")");
                    if (DEBUG_STEP_BY_STEP_V)
                        for (Instance inst : coveredButLabelsNotFullyCoveredInstances) System.out.println(inst);
                    else System.out.println(coveredButLabelsNotFullyCoveredInstances.size());
                }
                
                if (useSkippingRules) {
                    if (DEBUG_STEP_BY_STEP)
                        System.out.println(
                                coveredButLabelsNotFullyCoveredInstances.size() + " " + coveredInstances.size() + " " +
                                        (double) coveredButLabelsNotFullyCoveredInstances.size() /
                                                (double) coveredInstances.size() + " " +
                                        ((double) coveredButLabelsNotFullyCoveredInstances.size() /
                                                (double) coveredInstances.size() > skipThresholdPercentage));
                    if ((double) coveredButLabelsNotFullyCoveredInstances.size() / (double) coveredInstances.size() >
                            skipThresholdPercentage) {
                        //most covered examples were not fully label-covered
                        if (!readdAllCovered) {
                            for (int i = 0; i < coveredButLabelsNotFullyCoveredInstances.size(); i++) {
                                examples.addDirectly(coveredButLabelsNotFullyCoveredInstances
                                        .get(i)); //covered but labels not fully covered
                            }
                        } else {
                            for (int i = 0; i < coveredInstances.size(); i++) {
                                examples.addDirectly(
                                        coveredInstances.get(i)); //covered, labels fully and not fully covered
                            }
                        }
                    } else {
                        //don't add examples and add a special rule (since most covered examples were fully label-covered)
                        MultiHeadRule skipRule = new MultiHeadRule(null);
                        Head skipHead = new Head();
                        skipHead.addCondition(new NominalCondition(new Attribute("magicSkipHead"), 0));
                        skipRule.setHead(skipHead);
                        skipRule.setStats(new TwoClassConfusionMatrix(
                                coveredInstances.size() - coveredButLabelsNotFullyCoveredInstances.size(), 0,
                                coveredButLabelsNotFullyCoveredInstances.size(), 0));
                        theory.addRule(skipRule);
                    }
                } else {
                    // currently only exactly covered heads are used
                	// Re-add instances to training set for next iteration
                	
                	if (getAllLabelsCovered()) {
                	
                		for (int i = 0; i < coveredButLabelsNotFullyCoveredInstances.size(); i++) {
                			examples.addDirectly(coveredButLabelsNotFullyCoveredInstances.get(i));
                		}
                	}
                }
            } else {
                break;
            }
        }
        
        // use a _sorted_ Decision List as aggregation function
        
        if (false && getEvaluationMethod().equals("DecisionList")) {
        	Instances evalExamples = new Instances(originalExamples,
                    originalExamples.numInstances()); //so that I can do what I want on this
            examplesReferences = null; // only used for debugging
            
            if (DEBUG_STEP_BY_STEP)
                examplesReferences = new ArrayList<>();

            for (int i = 0; i < originalExamples.size(); i++) {
                Instance inst = originalExamples.get(i);
                Instance wrappedInstance;

                if (inst instanceof SparseInstance) {
                    wrappedInstance = new SparseInstanceWrapper(inst, labelIndices);
                } else {
                    wrappedInstance = new DenseInstanceWrapper(inst, labelIndices);
                }

                evalExamples.addDirectly(wrappedInstance); //now secured

                if (DEBUG_STEP_BY_STEP)
                    examplesReferences.add(wrappedInstance);
            }
        	EvaluationStrategy evaluationStrategy = EvaluationStrategy.create(getEvaluationStrategy());
            AveragingStrategy averagingStrategy = AveragingStrategy.create(getAveragingStrategy());
            MultiLabelEvaluation multiLabelEvaluation = new MultiLabelEvaluation(getHeuristic(), evaluationStrategy,
                    averagingStrategy);
            MulticlassCovering multiclassCovering = new MulticlassCovering(multiLabelEvaluation, isPredictZero());
            theory = multiclassCovering.sortTheory(theory, evalExamples, labelIndicesAsSet);
            System.out.println(theory);
        }
        
        return theory;
    }

    public SingleHeadRuleSet standardSeparateAndConquerMultilabel(Instances examples, int labelIndices[]) throws
            Exception {
        SeCoLogger.debug("entering separateAndConquerMultilabel");
        Instances originalExamples = examples; // newExamples used only in postprocessor
        examples = new Instances(originalExamples,
                originalExamples.numInstances()); //so that I can do what I want on this
        ArrayList<Instance> examplesReferences = null; // only used for debugging

        if (DEBUG_STEP_BY_STEP)
            examplesReferences = new ArrayList<>();

        for (int i = 0; i < originalExamples.size(); i++) {
            Instance inst = originalExamples.get(i);
            Instance wrappedInstance;

            if (inst instanceof SparseInstance) {
                wrappedInstance = new SparseInstanceWrapper(inst, labelIndices);
            } else {
                wrappedInstance = new DenseInstanceWrapper(inst, labelIndices);
            }

            examples.addDirectly(wrappedInstance); //now secured

            if (DEBUG_STEP_BY_STEP)
                examplesReferences.add(wrappedInstance);
        }

        SingleHeadRule r;
        SingleHeadRule bestRuleOfMulti;
        SingleHeadRuleSet theory = new SingleHeadRuleSet();
        theory.setLabelIndices(labelIndices); //so that tostring prints out mlc statistics
        int trainingDataSize = examples.getInstances().size();
        double classValueToLearn; // 0 if zero rules are used, 1 otherwise

        // Continue until a certain percentage of the training data is covered
        outerloop:
        while (examples.getInstances().size() > trainingDataSize * noNeedForClassification) {
            bestRuleOfMulti = null;

            if (DEBUG_STEP_BY_STEP) {
                System.out.println("########remaining training set (" + examples.size() + ")");
                if (DEBUG_STEP_BY_STEP_V) for (Instance inst : examples) System.out.println(inst);
                else System.out.println(examples.size());
                System.out.println("########candidate rules");
            }

            // Iterate over all labels
            for (int labelIndex = 0; labelIndex < labelIndices.length; labelIndex++) {
                examples.setClassIndex(labelIndices[labelIndex]);
                classValueToLearn = predictZero ? 0.0 : 1.0;

                for (; classValueToLearn <= 1; classValueToLearn++) {
                    if (growingSetSize != 1) {
                        if (ripper == null) {
                            ripper = new JRipOneRuler();

                            if (postProcessor instanceof NoOpPostProcessor)
                                ripper.setUsePruning(false);
                            if (postProcessor instanceof PostProcessorRipper) {
                                ripper.setUsePruning(true);
                                ripper.setOptimizations(((PostProcessorRipper) postProcessor).getNumOptimizations());
                            }
                        }

                        Instances examplesForRipper = new Instances(examples, examples.numInstances());
                        for (int i = 0; i < examples.numInstances(); i++) {
                            Instance inst = examples.instance(i);

                            if (Utils.isMissingValue(
                                    inst.value(inst.classIndex()))) //otherwise the class was already set
                                examplesForRipper.add(inst);
                        }

                        r = null;

                        if (examplesForRipper.size() != 0) {
                            try {
                                r = ripper.getBestRuleForClass((int) classValueToLearn, examplesForRipper);
                            } catch (Exception e) {
                                //most of the time it's that not enough training with class labels were found
                                r = null;
                            }
                        }

                        if (r == null) {
                            //create default rule
                            r = new SingleHeadRule(heuristic,
                                    new NominalCondition(examples.classAttribute(), classValueToLearn));
                        }

                        r.setHeuristic(heuristic);
                        r.evaluateRuleForMultilabel(examples, classValueToLearn, heuristic);

                        if (DEBUG_STEP_BY_STEP && DEBUG_STEP_BY_STEP_V)
                            System.out.println(
                                    "SingleHeadRule for label " + labelIndex + " = " + ((int) classValueToLearn) +
                                            ": " + r);
                    } else {
                        r = findBestRuleForMultilabel(examples, null, classValueToLearn);
                    }

                    // Keep learned rule, if it is currently the best rule for predicting the considered label
                    if (bestRuleOfMulti == null || bestRuleOfMulti.compareTo(r) < 0) {
                        bestRuleOfMulti = r;
                    }
                }
            }

            ArrayList<Instance> coveredInstances = bestRuleOfMulti.coveredInstances(examples);
            ArrayList<Instance> coveredButLabelsNotFullyCoveredInstances = new ArrayList<Instance>();
            examples = bestRuleOfMulti.uncoveredInstances(examples); //maintain these in any case

            if (DEBUG_STEP_BY_STEP) {
                System.out.println("########uncovered by rule (" + examples.size() + ")");
                if (DEBUG_STEP_BY_STEP_V) for (Instance inst : examples) System.out.println(inst);
                else System.out.println(examples.size());
            }

            Condition head = bestRuleOfMulti.getHead();
            int coveredLabelIndex = head.getAttr().index();

            for (int i = 0; i < coveredInstances.size(); i++) {
                Instance covered = coveredInstances.get(i);
                covered.dataset().setClassIndex(
                        coveredLabelIndex); //this is obligatory due to that setvalue executed toDoubleArray, and here the classIndex features is overwritten with the real value. so, before this bugfix, an additional label features (the last one) was always overwritten

                if (Utils.isMissingValue(covered.value(coveredLabelIndex))) {
                    covered.setValue(coveredLabelIndex, head.getValue());
                }

                if (predictZero) {
                    int uncoveredLabels = getUncoveredLabels(covered, labelIndices);

                    if (uncoveredLabels > 0) {
                        coveredButLabelsNotFullyCoveredInstances.add(covered);
                    }
                } else {
                    int uncoveredPosLabels = getUncoveredPosLabels(covered, labelIndices);

                    if (uncoveredPosLabels > 0) {
                        // there are still label attributes to fill up, so continue
                        coveredButLabelsNotFullyCoveredInstances.add(covered);
                    }
                }
            }

            theory.addRule(bestRuleOfMulti);

            if (DEBUG_STEP_BY_STEP) {
                System.out.println("########covered by rule (and predicted written) (" + coveredInstances.size() + ")");
                if (DEBUG_STEP_BY_STEP_V) for (Instance inst : coveredInstances) System.out.println(inst);
                else System.out.println(coveredInstances.size());
                System.out.println(
                        "########readdition candidates (" + coveredButLabelsNotFullyCoveredInstances.size() + ")");
                if (DEBUG_STEP_BY_STEP_V)
                    for (Instance inst : coveredButLabelsNotFullyCoveredInstances) System.out.println(inst);
                else System.out.println(coveredButLabelsNotFullyCoveredInstances.size());
            }

            if (useSkippingRules) {
                if (DEBUG_STEP_BY_STEP)
                    System.out.println(
                            coveredButLabelsNotFullyCoveredInstances.size() + " " + coveredInstances.size() + " " +
                                    (double) coveredButLabelsNotFullyCoveredInstances.size() /
                                            (double) coveredInstances.size() + " " +
                                    ((double) coveredButLabelsNotFullyCoveredInstances.size() /
                                            (double) coveredInstances.size() > skipThresholdPercentage));
                if ((double) coveredButLabelsNotFullyCoveredInstances.size() / (double) coveredInstances.size() >
                        skipThresholdPercentage) {
                    //most covered examples were not fully label-covered
                    if (!readdAllCovered) { //TODO: eigentlich m�sste auch useSkipping=false && hackAddAll gehen, oder?
                        for (int i = 0; i < coveredButLabelsNotFullyCoveredInstances.size(); i++) {
                            examples.addDirectly(coveredButLabelsNotFullyCoveredInstances
                                    .get(i)); //covered but labels not fully covered
                        }
                    } else {
                        for (int i = 0; i < coveredInstances.size(); i++) {
                            examples.addDirectly(coveredInstances.get(i)); //covered, labels fully and not fully covered
                        }
                    }
                } else {
                    //dont add examples and add a special rule (since most covered examples were fully label-covered)
                    SingleHeadRule skipRule = new SingleHeadRule(null,
                            new NominalCondition(new Attribute("magicSkipHead"), 1));
                    skipRule.setStats(new TwoClassConfusionMatrix(
                            coveredInstances.size() - coveredButLabelsNotFullyCoveredInstances.size(), 0,
                            coveredButLabelsNotFullyCoveredInstances.size(), 0));
                    theory.addRule(skipRule);
                }
            } else {
                // Re-add instances to training set for next iteration
                for (int i = 0; i < coveredButLabelsNotFullyCoveredInstances.size(); i++) {
                    examples.addDirectly(coveredButLabelsNotFullyCoveredInstances.get(i));
                }
            }

            if (DEBUG_STEP_BY_STEP)
                System.out.println(theory);
        }

        return theory;
    }

    private Instances getAsInstances(Instances examples, List<Instance> growingExamples) {
        Instances result = new Instances(examples, 0);
        for (Instance instance : growingExamples) {
            result.addDirectly(instance);
        }
        return result;
    }

    static List<Instance> getGrowingSet(List<Instance> stratifiedTrainset, double growingSetSize) {
        final int split = (int) (stratifiedTrainset.size() * growingSetSize);

//		rt[0] = new Instances(this, 0, splits);
//		rt[1] = new Instances(this, splits, this.numInstances() - splits);
//		growingSet = rt[0];
//		pruningSet = rt[1];

        ArrayList<Instance> growingSet = new ArrayList<Instance>();
        for (int i = 0; i < split; i++) {
            growingSet.add(stratifiedTrainset.get(i));
        }
        return growingSet;
    }

    static List<Instance> getPruningSet(List<Instance> stratifiedTrainset, double growingSetSize) {
        final int split = (int) (stratifiedTrainset.size() * growingSetSize);

//		rt[0] = new Instances(this, 0, splits);
//		rt[1] = new Instances(this, splits, this.numInstances() - splits);
//		growingSet = rt[0];
//		pruningSet = rt[1];

        ArrayList<Instance> pruningSet = new ArrayList<Instance>();
        for (int i = split; i < stratifiedTrainset.size(); i++) {
            pruningSet.add(stratifiedTrainset.get(i));
        }
        return pruningSet;
    }

    private int getUncoveredLabels(Instance covered, int[] labelIndices) {
        int count = 0;

        for (int i = 0; i < labelIndices.length; i++) {
            int labelIndex = labelIndices[i];
            if (covered.isMissing(labelIndex))
                count++;

        }
        return count;
    }


    private int getUncoveredPosLabels(Instance covered, int[] labelIndices) {
        int count = 0;
        Instance parent = null;
        if (covered instanceof DenseInstanceWrapper)
            parent = ((DenseInstanceWrapper) covered).getWrappedInstance(); //TODO: may work, may not work...
        else
            parent = ((SparseInstanceWrapper) covered).getWrappedInstance(); //TODO: may work, may not work...

        for (int i = 0; i < labelIndices.length; i++) {
            int labelIndex = labelIndices[i];
            if (parent.value(labelIndex) == 1.0) {
                count++;
//				if(Double.compare(covered.value(labelIndex),Utils.missingValue())!=0)
                if (!covered.isMissing(labelIndex))
                    count--; //it does not matter, if it is correct, but it should be somehow set, then it was somehow covered.
            }
        }
        return count;
    }

    /**
     * Stratify the given data into the given number of bags based on the class values. It differs from the
     * <code>Instances.stratify(int fold)</code> that before stratification it sorts the instances according to the
     * class order in the header file. It assumes no missing values in the class.
     *
     * @param data           the given data
     * @param growingSetSize the growing set size
     * @param rand           the random object used to randomize the instances
     * @return the stratified instances
     * @throws UnassignedClassException
     * @throws UnassignedDatasetException
     */
    public static final List<Instance> stratifyForMultilabel(final Instances data, final double growingSetSize,
                                                             final Random rand) throws UnassignedClassException,
            UnassignedDatasetException {

        final int folds = (int) Math.round(1 / (1 - growingSetSize));
        if (!data.classAttribute().isNominal())
            return data;

        final ArrayList<Instance> result = new ArrayList<>(data.size());
        final ArrayList<Instance>[] bagsByClasses = new ArrayList[data.numClasses()];
        for (int i = 0; i < bagsByClasses.length; i++)
            bagsByClasses[i] = new ArrayList<>(data.size());

        // Sort by class
        for (int j = 0; j < data.numInstances(); j++) {
            final Instance datum = data.instance(j);
            bagsByClasses[(int) datum.classValue()].add(datum);
        }

        // bagsByClasses is sorted by the number of instances, very important for RIPPER

        final int n = bagsByClasses.length;
        final ArrayList<Instance>[] temp = new ArrayList[1];
        for (int pass = 1; pass < n; pass++)
            // This next loop becomes shorter and shorter
            for (int i = 0; i < n - pass; i++)
                if (bagsByClasses[i].size() > bagsByClasses[i + 1].size()) {
                    // exchange elements
                    temp[0] = bagsByClasses[i];
                    bagsByClasses[i] = bagsByClasses[i + 1];
                    bagsByClasses[i + 1] = temp[0];
                }

        // Randomize each class
        for (final ArrayList<Instance> bagsByClasse : bagsByClasses)
//			bagsByClasse.randomize(rand);
            Collections.shuffle(bagsByClasse, rand);

        for (int k = 0; k < folds; k++) {
            int offset = k, bag = 0;
            oneFold:
            while (true) {
                while (offset >= bagsByClasses[bag].size()) {
                    offset -= bagsByClasses[bag].size();
                    if (++bag >= bagsByClasses.length)// Next bag
                        break oneFold;
                }

                result.add(bagsByClasses[bag].get(offset));
                offset += folds;
            }
        }

        return result;
    }

}