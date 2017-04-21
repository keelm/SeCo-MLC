/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * PostProcessorRipper.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Jiawei Du Created on 12.10.2009, 16:12
 */

package de.tu_darmstadt.ke.seco.algorithm.components.postprocessors;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.util.ArrayList;

import de.tu_darmstadt.ke.seco.models.*;
import weka.core.Instance;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.ErrorRate;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.stats.RuleStats;
import de.tu_darmstadt.ke.seco.utils.Logger;

/**
 * This postprocessor will optimize the rule set by constructing two alternative rules. (replacement & revision)
 *
 * @author Knowledge Engineering Group
 */
public class PostProcessorRipper extends PostProcessor implements Serializable {

    // default constructor needed by the factory
    public PostProcessorRipper() {
        super();
    }

    public PostProcessorRipper(final SeCoAlgorithm seCoAlgorithm) {
        super(seCoAlgorithm);
    }

    private static final long serialVersionUID = 1L;

    // Default optimization iteration.
    @ConfigurableProperty
    private int optimizations = 2;

    public int getNumOptimizations() {
        return optimizations;
    }

    // Generates the variant abridgment, if m_abridgment = 1.
    @ConfigurableProperty
    private boolean useAbridgment = false;

    // Selection criterion for variants.
    // Default is MDL. // TODO by m.zopf: As MDL is the default, selectionCriterion should not be null. It should be MDL. MDL is not implemented as a heuristic!
    @ConfigurableProperty
    private Heuristic selectionCriterion = null;

    public void clone(final SeCoAlgorithm seCoAlgorithm) {
        this.seCoAlgorithm.setCandidateSelector(seCoAlgorithm.getCandidateSelector());
        this.seCoAlgorithm.setRuleInitializer(seCoAlgorithm.getRuleInitializer());
        this.seCoAlgorithm.setHeuristic(seCoAlgorithm.getHeuristic());
        this.seCoAlgorithm.setRuleRefiner(seCoAlgorithm.getRuleRefiner());
        this.seCoAlgorithm.setRuleFilter(seCoAlgorithm.getRuleFilter());
        this.seCoAlgorithm.setStoppingCriterion(seCoAlgorithm.getStoppingCriterion());
        this.seCoAlgorithm.setRuleStoppingCriterion(seCoAlgorithm.getRuleStoppingCriterion());
        this.seCoAlgorithm.setGrowingSetSize(seCoAlgorithm.getGrowingSetSize());
        this.seCoAlgorithm.setRandom(seCoAlgorithm.getRandom());
        this.seCoAlgorithm.setMinNo(seCoAlgorithm.getMinNo());
    }

    /**
     * Optimizes the initial rule set.
     *
     * @param theory   the initial rule set
     * @param examples the training examples
     */
    @Override
    public SingleHeadRuleSet postProcessTheory(SingleHeadRuleSet theory, Instances examples, final double classValue) throws Exception {
        if (optimizations == 0)
            return theory;

        Instances newExamples, growingExamples, pruningExamples;
        // Calculate expFPRate & defDL
        final double all_m_classVal = examples.countInstances(classValue);
        final double all = examples.numInstances();
        final double expFPRate = all_m_classVal / all;

        double classYWeights = 0, totalWeights = 0;
        for (int j = 0; j < examples.numInstances(); j++) {
            final Instance datum = examples.instance(j);
            totalWeights += datum.weight();
            if ((int) datum.classValue() == classValue)
                classYWeights += datum.weight();
        }
        final double defDL = RuleStats.dataDL(expFPRate, 0.0, totalWeights, 0.0, classYWeights);

        final double m_Total = RuleStats.numAllConditions(examples);
        boolean stop = false;
        // Check whether data have positive examples
        final boolean defHasPositive = true; // No longer used
        boolean hasPositive = defHasPositive;
        double dl = defDL, minDL = defDL;
        RuleStats rstats = null;
        double[] rst;

        RuleStats finalRulesetStat = null;

        for (int z = 0; z < optimizations; z++) {
            newExamples = examples;
            finalRulesetStat = new RuleStats();
            finalRulesetStat.setData(newExamples);
            finalRulesetStat.setNumAllConds(m_Total);
            int position = 0;
            stop = false;
            boolean isResidual = false;
            hasPositive = defHasPositive;
            dl = minDL = defDL;

            oneRule:
            while (!stop && hasPositive) {

                isResidual = (position >= theory.size());

                newExamples = RuleStats.stratify(newExamples, seCoAlgorithm.getGrowingSetSize(), seCoAlgorithm.getRandom());
                final SplittedInstances splitInst = new SplittedInstances(newExamples);
                splitInst.splitInstances(seCoAlgorithm.getGrowingSetSize());
                growingExamples = splitInst.getGrowingSet();
                pruningExamples = splitInst.getPruningSet();

                SingleHeadRule finalRule = null;

                // Check if all the learned Rules are optimized?
                if (isResidual) {
                    // Find a new SingleHeadRule to cover the residual positive examples.
                    SingleHeadRule newRule;
                    newRule = seCoAlgorithm.findBestRule(growingExamples, null, classValue);
                    newRule = seCoAlgorithm.pruneRule(pruningExamples, newRule, false, classValue);

                    newRule.evaluateRule(pruningExamples, all_m_classVal, seCoAlgorithm.getHeuristic());

                    finalRule = newRule;
                } else {
                    final SingleHeadRule oldRule = theory.getRule(position);
                    oldRule.evaluateRule(newExamples, all_m_classVal, seCoAlgorithm.getHeuristic());

                    boolean covers = false;

                    // Check the Coverage of the old SingleHeadRule.
                    for (int i = 0; i < newExamples.numInstances(); i++)
                        if (oldRule.covers(newExamples.instance(i))) {
                            covers = true;
                            break;
                        }

                    // Null coverage, this rule can be removed from the theory.
                    if (!covers) {
                        theory.deleteRule(position);
                        finalRulesetStat.addAndUpdate(oldRule);
                        position++;
                        continue oneRule;
                    }

                    // Variant Abridgment
                    SingleHeadRule abridgmentRule = null;
                    if (useAbridgment)
                        abridgmentRule = pruneOldRule(newExamples, oldRule);

                    // Variant Replacement
                    SingleHeadRule replace;
                    replace = seCoAlgorithm.findBestRule(growingExamples, null, classValue);
                    replace.evaluateRule(newExamples, all_m_classVal, seCoAlgorithm.getHeuristic());

                    // Remove the pruning data covered by the following
                    // rules, then simply compute the error rate of the
                    // current rule to prune it. According to Ripper,
                    // it's equivalent to computing the error of the
                    // whole ruleset
                    pruningExamples = RuleStats.rmCoveredBySuccessives(pruningExamples, theory, position);
                    replace = seCoAlgorithm.pruneRule(pruningExamples, replace, true, classValue);

                    replace.evaluateRule(newExamples, all_m_classVal, seCoAlgorithm.getHeuristic());

                    // Variant Revision
                    SingleHeadRule revision = (SingleHeadRule) oldRule.copy();

                    // For revision, first rm the data covered by the old rule
                    final Instances newGrowingExamples = new Instances(growingExamples, 0);
                    for (int b = 0; b < growingExamples.numInstances(); b++) {
                        final Instance inst = growingExamples.instance(b);
                        if (revision.covers(inst))
                            newGrowingExamples.add(inst);
                    }

                    revision = seCoAlgorithm.findBestRule(newGrowingExamples, revision, classValue);

                    revision.evaluateRule(growingExamples, all_m_classVal, seCoAlgorithm.getHeuristic());
                    revision = seCoAlgorithm.pruneRule(pruningExamples, revision, true, classValue);

                    revision.evaluateRule(newExamples, all_m_classVal, seCoAlgorithm.getHeuristic());

                    // Default evualtion values of variants
                    final double[] value = {0, 0, 0, 0};
                    // the amount of variants
                    double count = 0;

                    // other metrics
                    if (!(selectionCriterion == null)) {
                        // value[0] = -1 * oldRule.getStats().getAccuracy();
                        value[0] = -1 * selectionCriterion.evaluateRule(oldRule);
                        value[1] = -1 * selectionCriterion.evaluateRule(revision);
                        value[2] = -1 * selectionCriterion.evaluateRule(replace);
                        count = 3;
                        if (abridgmentRule != null) {
                            value[3] = -1 * selectionCriterion.evaluateRule(abridgmentRule);
                            count = 4;
                        }
                    }
                    // metric MDL
                    else {
                        count = 3;
                        final double[][] prevRuleStats = new double[position][6];
                        for (int c = 0; c < position; c++)
                            prevRuleStats[c] = finalRulesetStat.getSimpleStats(c);

                        // Calculate the relative DL of Abrigment.
                        if (abridgmentRule != null) {
                            theory.replaceCondition(abridgmentRule, position);
                            final RuleStats abrStat = new RuleStats(examples, theory);
                            abrStat.setNumAllConds(m_Total);
                            abrStat.countData(position, newExamples, prevRuleStats);
                            // repStat.countData();
                            rst = abrStat.getSimpleStats(position);

                            Logger.debug("Abridgment rule covers: " + rst[0] + " | pos = " + rst[2] + " | neg = " + rst[4] + "\nThe rule doesn't cover: " + rst[1] + " | pos = " + rst[5]);

                            value[3] = abrStat.relativeDL(position, expFPRate, true);

                            if (Double.isNaN(value[3]) || Double.isInfinite(value[3]))
                                throw new Exception("Should never happen: repDL" + "in optmz. stage NaN or " + "infinite!");
                            count = 4;
                        }

                        // Calculate the relative DL of Replacement.
                        theory.replaceCondition(replace, position);
                        final RuleStats repStat = new RuleStats(examples, theory);
                        repStat.setNumAllConds(m_Total);
                        repStat.countData(position, newExamples, prevRuleStats);
                        // repStat.countData();
                        rst = repStat.getSimpleStats(position);

                        Logger.debug("Replace rule covers: " + rst[0] + " | pos = " + rst[2] + " | neg = " + rst[4] + "\nThe rule doesn't cover: " + rst[1] + " | pos = " + rst[5]);

                        value[2] = repStat.relativeDL(position, expFPRate, true);

                        if (Double.isNaN(value[2]) || Double.isInfinite(value[2]))
                            throw new Exception("Should never happen: repDL" + "in optmz. stage NaN or " + "infinite!");

                        // Calculate the relative DL of Revision.
                        theory.replaceCondition(revision, position);
                        final RuleStats revStat = new RuleStats(examples, theory);
                        revStat.setNumAllConds(m_Total);
                        revStat.countData(position, newExamples, prevRuleStats);
                        // revStat.countData();
                        rst = revStat.getSimpleStats(position);

                        Logger.debug("Revision rule covers: " + rst[0] + " | pos = " + rst[2] + " | neg = " + rst[4] + "\nThe rule doesn't cover: " + rst[1] + " | pos = " + rst[5]);

                        value[1] = revStat.relativeDL(position, expFPRate, true);

                        if (Double.isNaN(value[1]) || Double.isInfinite(value[1]))
                            throw new Exception("Should never happen: revDL" + "in optmz. stage NaN or " + "infinite!");

                        // Calculate the relative DL of old SingleHeadRule.
                        theory.replaceCondition(oldRule, position);
                        rstats = new RuleStats(examples, theory);
                        rstats.setNumAllConds(m_Total);
                        rstats.countData(position, newExamples, prevRuleStats);
                        // rstats.countData();
                        rst = rstats.getSimpleStats(position);

                        Logger.debug("Old rule covers: " + rst[0] + " | pos = " + rst[2] + " | neg = " + rst[4] + "\nThe rule doesn't cover: " + rst[1] + " | pos = " + rst[5]);

                        value[0] = rstats.relativeDL(position, expFPRate, true);

                        if (Double.isNaN(value[0]) || Double.isInfinite(value[0]))
                            throw new Exception("Should never happen: oldDL" + "in optmz. stage NaN or " + "infinite!");
                    }

                    // Select the best Variant based on the different Metric.
                    switch (min(value, count)) {
                        case 0:
                            finalRule = oldRule;
                            break;
                        case 1:
                            finalRule = revision;
                            break;
                        case 2:
                            finalRule = replace;
                            break;
                        case 3:
                            finalRule = abridgmentRule;
                            break;
                    }
                }

                finalRulesetStat.addAndUpdate(finalRule);
                rst = finalRulesetStat.getSimpleStats(position);

				/*
                 * System.out.println("Final rule covers: " + finalRule +rst[0]+ " | pos = " + rst[2] + " | neg = " + rst[4]+ "\nThe rule doesn't cover: "+rst[1]+ " | pos = " + rst[5]);
				 */

                // if(rst[2]<=rst[4]){
                // finalRulesetStat.removeLast();
                // break;
                // }
                if (isResidual) {

                    dl += finalRulesetStat.relativeDL(position, expFPRate, true);

                    Logger.debug("ERROR: After optimization: the dl" + "=" + dl + " | best: " + minDL);

                    if (dl < minDL)
                        minDL = dl; // The best dl so far

                    // stop = m_rsc.checkForRuleStop(theory, finalRule,
                    // newExamples, null);
                    // stop = m_rsc.checkStop(rst, minDL, dl);
                    stop = checkStop(rst, minDL, dl);

                    if (!stop)
                        theory.addRule(finalRule); // Accepted
                    else {
                        finalRulesetStat.removeLast(); // Remove last to be
                        // re-used
                        position--;
                    }
                } else
                    theory.replaceCondition(finalRule, position); // Accepted

                // Data not covered
                if (finalRulesetStat.getRulesetSize() > 0)// If any rules
                    // newExamples = finalRule.uncoveredInstances(newExamples);
                    newExamples = finalRulesetStat.getFiltered(position)[1];

                hasPositive = newExamples.containsPositive(classValue); // Positives
                // remaining?

                position++;
            }

            if (theory.size() > (position + 1))
                for (int k = position + 1; k < theory.size(); k++)
                    finalRulesetStat.addAndUpdate(theory.getRule(k));
            Logger.debug("\nERROR:Deleting rules to decrease" + " DL of the whole ruleset ...");
            finalRulesetStat.reduceDL(expFPRate, true);
            Logger.debug("ERROR: " + (theory.size() - finalRulesetStat.getRulesetSize()) + " rules are deleted after DL reduction procedure");
            theory = finalRulesetStat.getRuleset();
            rstats = finalRulesetStat;
        }

        // System.out.println("Optimized SingleHeadRuleSet");
        for (int i = 0; i < theory.size(); i++) {
            final SingleHeadRule a = theory.getRule(i);

            a.evaluateRule(examples, classValue, seCoAlgorithm.getHeuristic());
            examples = a.uncoveredInstances(examples);
        }
        // System.out.println("----------------------------------");

        seCoAlgorithm.setInstances(examples);

        return theory;
    }

    /**
     * Compares the variants with each other.
     *
     * @param value  the value of variants.
     * @param length the amount of variants
     */
    private int min(final double[] value, final double length) {
        int flag = 0;
        double minValue = value[0];
        for (int i = 1; i < length; i++)
            if (value[i] < minValue) {
                minValue = value[i];
                flag = i;
            }
        return flag;
    }

    @Override
    public void setProperty(final String name, final String value) {
        if (name.equals("optimizations"))
            optimizations = Integer.parseInt(value);
        if (name.equals("abrigment"))
            useAbridgment = Boolean.parseBoolean(value);

        /** Creates a new instance of Heuristic, if selection!=MDL */
        if (name.equals("selection") && !value.equals("MDL"))
            try {
                createByClassname("seco.heuristics" + "." + value);
            } catch (final Exception e1) {
                e1.printStackTrace();
                System.exit(0);
            }
    }

    /**
     * This will create a new instance of the named class using a default constructor without arguments.
     *
     * @param name The classname.
     */
    private void createByClassname(final String name) throws Exception {
        Logger.debug("creating object by classname: " + name);
        final Class<?> c = Class.forName(name);
        final Constructor<?> co = c.getConstructor(new Class[0]);
        selectionCriterion = (Heuristic) co.newInstance(new Object[0]);
    }

    /**
     * Generates the variant Abridgment.
     *
     * @param newExamples The pruning examples
     * @param oldRule     The old rule of the initial rule set
     * @return the pruned old rule.
     */
    private SingleHeadRule pruneOldRule(final Instances newExamples, final SingleHeadRule oldRule) throws Exception {
        final ErrorRate errorRate = new ErrorRate();

        // Gets the default error rate of the old rule.
        final double defaultErrorRate = errorRate.evaluateRule(oldRule);
        int maxValuePosition;
        double maxValue = defaultErrorRate;
        final ArrayList<Condition> m_Antds = oldRule.getBody();
        ArrayList<Condition> m_Antds_oldRule = new ArrayList<Condition>(m_Antds);
        final ArrayList<Condition> m_Antds_tempRule = new ArrayList<Condition>(m_Antds);

        SingleHeadRule temp = (SingleHeadRule) oldRule.clone();
        SingleHeadRule temp2 = (SingleHeadRule) oldRule.clone();

        while (m_Antds_oldRule.size() > 1) {
            maxValuePosition = -1;
            for (int i = 0; i < m_Antds_oldRule.size(); i++) {
                m_Antds_oldRule.remove(i);

                temp.initBody();
                for (final Condition c : m_Antds_oldRule)
                    temp.addCondition(c);

                temp.evaluateRule(newExamples, i, seCoAlgorithm.getHeuristic());

                if (errorRate.evaluateRule(temp) <= maxValue) {
                    maxValuePosition = i;
                    maxValue = errorRate.evaluateRule(temp);
                    temp2.initBody();
                    for (final Condition c : m_Antds_oldRule)
                        temp2.addCondition(c);
                }

                m_Antds_oldRule = new ArrayList<Condition>(m_Antds_tempRule);
            }

            // Discards the one whose removal has the least detrimental effect
            // on the error rate of the rule
            if (maxValuePosition >= 0) {
                m_Antds_oldRule.remove(maxValuePosition);
                m_Antds_tempRule.remove(maxValuePosition);
            } else
                break;

        }
        return temp2;
    }

    /**
     * Check whether the stopping criterion meets.
     *
     * @param rst   the statistic of the ruleset
     * @param minDL the min description length so far
     * @param dl    the current description length of the ruleset
     * @return true if stop criterion meets, false otherwise
     */
    private boolean checkStop(final double[] rst, final double minDL, final double dl) {
        final boolean m_Check = true;
        if (dl > minDL + 64) {
            Logger.debug("ERROR: DL too large: " + dl + " | " + minDL);
            return true;
        } else if (!RuleStats.gr(rst[2], 0.0)) {// Covered positives
            Logger.debug("ERROR: Too few positives.");
            return true;
        } else if ((rst[4] / rst[0]) >= 0.5) {// Err rate
            if (m_Check) {
                Logger.debug("ERROR: Error too large: " + rst[4] + "/" + rst[0]);
                return true;
            } else
                return false;
        } else {// Not stops
            Logger.debug("ERROR: Continue.");
            return false;
        }
    }
}