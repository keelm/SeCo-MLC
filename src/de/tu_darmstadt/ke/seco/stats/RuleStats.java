/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * RuleStats.java Copyright (C) 2001 University of Waikato, Hamilton, New Zealand
 */

package de.tu_darmstadt.ke.seco.stats;

import de.tu_darmstadt.ke.seco.models.*;
import weka.core.Instance;
import weka.core.UnassignedClassException;
import weka.core.UnassignedDatasetException;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;

/**
 * This class implements the statistics functions used in the propositional rule learner, from the simpler ones like count of true/false positive/negatives, filter data based on the ruleset, etc. to the more sophisticated ones such as MDL calculation and rule variants generation for each rule in the ruleset.
 * <p>
 * <p>
 * Obviously the statistics functions listed above need the specific data and the specific ruleset, which are given in order to instantiate an object of this class.
 * <p>
 *
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 4608 $
 */
public class RuleStats implements Serializable {

    /**
     * for serialization
     */
    static final long serialVersionUID = -5708153367675298624L;

    /**
     * The data on which the stats calculation is based
     */
    private Instances m_Data;

    /**
     * The specific ruleset in question
     */
    private SingleHeadRuleSet m_Ruleset;

    /**
     * The simple stats of each rule
     */
    private ArrayList<Object> m_SimpleStats;

    /**
     * The set of instances filtered by the ruleset
     */
    private ArrayList<Object> m_Filtered;

    /**
     * The class distributions predicted by each rule
     */
    private ArrayList<Object> m_Distributions;
    /**
     * The total number of possible conditions that could appear in a rule
     */
    private double m_Total;

    /**
     * The redundancy factor in theory description length
     */
    private static double REDUNDANCY_FACTOR = 0.5;

    /**
     * The theory weight in the MDL calculation
     */
    private double MDL_THEORY_WEIGHT = 1.0;

    /**
     * The natural logarithm of 2.
     */
    public static double log2 = Math.log(2);

    /**
     * The small deviation allowed in double comparisons.
     */
    public static double SMALL = 1e-6;

    /**
     * Default constructor
     */
    public RuleStats() {
        m_Data = null;
        m_Ruleset = null;
        m_SimpleStats = null;
        m_Filtered = null;
        m_Distributions = null;
        m_Total = -1;
    }

    /**
     * Constructor that provides ruleset and data
     *
     * @param data   the data
     * @param theory the ruleset
     */
    public RuleStats(final Instances data, final SingleHeadRuleSet theory) {
        this();
        m_Data = data;
        m_Ruleset = theory;
    }

    /**
     * Frees up memory after classifier has been built.
     */
    public void cleanUp() {
        m_Data = null;
        m_Filtered = null;
    }

    /**
     * Set the number of all conditions that could appear in a rule in this RuleStats object, if the number set is smaller than 0 (typically -1), then it calcualtes based on the data store
     *
     * @param total the set number
     */
    public void setNumAllConds(final double total) {
        if (total < 0)
            m_Total = numAllConditions(m_Data);
        else
            m_Total = total;
    }

    /**
     * Set the data of the stats, overwriting the old one if any
     *
     * @param data the data to be set
     */
    public void setData(final Instances data) {
        m_Data = data;
    }

    /**
     * Get the data of the stats
     *
     * @return the data
     */
    public Instances getData() {
        return m_Data;
    }

    /**
     * Set the ruleset of the stats, overwriting the old one if any
     *
     * @param rules the set of rules to be set
     */
    public void setRuleset(final SingleHeadRuleSet rules) {
        m_Ruleset = rules;
    }

    /**
     * Get the ruleset of the stats
     *
     * @return the set of rules
     */
    public SingleHeadRuleSet getRuleset() {
        return m_Ruleset;
    }

    /**
     * Get the size of the ruleset in the stats
     *
     * @return the size of ruleset
     */
    public int getRulesetSize() {
        return m_Ruleset.size();
    }

    /**
     * Get the simple stats of one rule, including 6 parameters: 0: coverage; 1:uncoverage; 2: true positive; 3: true negatives; 4: false positives; 5: false negatives
     *
     * @param index the index of the rule
     * @return the stats
     */
    public double[] getSimpleStats(final int index) {
        if ((m_SimpleStats != null) && (index < m_SimpleStats.size()))
            return (double[]) m_SimpleStats.get(index);

        return null;
    }

    /**
     * Get the data after filtering the given rule
     *
     * @param index the index of the rule
     * @return the data covered and uncovered by the rule
     */
    public Instances[] getFiltered(final int index) {
        if ((m_Filtered != null) && (index < m_Filtered.size()))
            return (Instances[]) m_Filtered.get(index);

        return null;
    }

    /**
     * Get the class distribution predicted by the rule in given position
     *
     * @param index the position index of the rule
     * @return the class distributions
     */
    public double[] getDistributions(final int index) {

        if ((m_Distributions != null) && (index < m_Distributions.size()))
            return (double[]) m_Distributions.get(index);

        return null;
    }

    /**
     * Set the weight of theory in MDL calcualtion
     *
     * @param weight the weight to be set
     */
    public void setMDLTheoryWeight(final double weight) {
        MDL_THEORY_WEIGHT = weight;
    }

    /**
     * Compute the number of all possible conditions that could appear in a rule of a given data. For nominal attributes, it's the number of values that could appear; for numeric attributes, it's the number of values * 2, i.e. <= and >= are counted as different possible conditions.
     *
     * @param data the given data
     * @return number of all conditions of the data
     */
    public static double numAllConditions(final Instances data) {
        double total = 0;
        final Enumeration<?> attEnum = data.enumerateAttributes();

        for (int i = 0; i < data.getNumberOfAttributes() - 1; i++) {
            final Attribute att = Attribute.toSeCoAttribute((weka.core.Attribute) attEnum.nextElement());
            if (att.isNominal())
                total += att.numValues();
            else
                total += 2.0 * data.numDistinctValues(att);
        }

        return total;
    }

    /**
     * Filter the data according to the ruleset and compute the basic stats: coverage/uncoverage, true/false positive/negatives of each rule
     */
    public void countData() {
        if ((m_Filtered != null) || (m_Ruleset == null) || (m_Data == null))
            return;

        final int size = m_Ruleset.size();
        m_Filtered = new ArrayList<Object>(size);
        m_SimpleStats = new ArrayList<Object>(size);
        m_Distributions = new ArrayList<Object>(size);
        Instances data = new Instances(m_Data);

        for (int i = 0; i < size; i++) {
            final double[] stats = new double[6]; // 6 statistics parameters
            double[] classCounts = null;
            try {
                classCounts = new double[m_Data.classAttribute().numValues()];
            } catch (final UnassignedClassException e) {
                e.printStackTrace();
            }
            final Instances[] filtered = computeSimpleStats(i, data, stats, classCounts);
            m_Filtered.add(filtered);
            m_SimpleStats.add(stats);
            m_Distributions.add(classCounts);
            data = filtered[1]; // Data not covered
        }
    }

    /**
     * Count data from the position index in the ruleset assuming that given data are not covered by the rules in position 0...(index-1), and the statistics of these rules are provided.<br>
     * This procedure is typically useful when a temporary object of RuleStats is constructed in order to efficiently calculate the relative DL of rule in position index, thus all other stuff is not needed.
     *
     * @param index         the given position
     * @param uncovered     the data not covered by rules before index
     * @param prevRuleStats the provided stats of previous rules
     */
    public void countData(final int index, final Instances uncovered, final double[][] prevRuleStats) {
        if ((m_Filtered != null) || (m_Ruleset == null))
            return;

        final int size = m_Ruleset.size();
        m_Filtered = new ArrayList<Object>(size);
        m_SimpleStats = new ArrayList<Object>(size);
        Instances[] data = new Instances[2];
        data[1] = uncovered;

        for (int i = 0; i < index; i++) {
            m_SimpleStats.add(prevRuleStats[i]);
            if (i + 1 == index)
                m_Filtered.add(data);
            else
                m_Filtered.add(new Object()); // Stuff sth.
        }

        for (int j = index; j < size; j++) {
            final double[] stats = new double[6]; // 6 statistics parameters
            final Instances[] filtered = computeSimpleStats(j, data[1], stats, null);
            m_Filtered.add(filtered);
            m_SimpleStats.add(stats);
            data = filtered; // Data not covered
        }
    }

    /**
     * Find all the instances in the dataset covered/not covered by the rule in given index, and the correponding simple statistics and predicted class distributions are stored in the given double array, which can be obtained by getSimpleStats() and getDistributions().<br>
     *
     * @param index the given index, assuming correct
     * @param insts the dataset to be covered by the rule
     * @param stats the given double array to hold stats, side-effected
     * @param dist  the given array to hold class distributions, side-effected if null, the distribution is not necessary
     * @return the instances covered and not covered by the rule
     */
    private Instances[] computeSimpleStats(final int index, final Instances insts, final double[] stats, final double[] dist) {
        final SingleHeadRule rule = m_Ruleset.getRule(index);

        final Instances[] data = new Instances[2];
        data[0] = new Instances(insts, insts.numInstances());
        data[1] = new Instances(insts, insts.numInstances());
        try {
            for (int i = 0; i < insts.numInstances(); i++) {
                final Instance datum = insts.instance(i);
                final double weight = datum.weight();
                if (rule.covers(datum)) {
                    data[0].add(datum); // Covered by this rule
                    stats[0] += weight; // Coverage

                    if ((int) datum.classValue() == (int) rule.getHead().getValue())
                        stats[2] += weight; // True positives
                    else
                        stats[4] += weight;
                    // False positives
                    if (dist != null)
                        dist[(int) datum.classValue()] += weight;
                } else {
                    data[1].add(datum); // Not covered by this rule
                    stats[1] += weight;
                    if ((int) datum.classValue() != (int) rule.getHead().getValue())
                        stats[3] += weight; // True negatives
                    else
                        stats[5] += weight; // False negatives
                }
            }
        } catch (final UnassignedClassException e) {
            e.printStackTrace();
        } catch (final UnassignedDatasetException e) {
            e.printStackTrace();
        }

        return data;
    }

    /**
     * Add a rule to the ruleset and update the stats
     *
     * @param lastRule the rule to be added
     */
    public void addAndUpdate(final SingleHeadRule lastRule) {
        if (m_Ruleset == null)
            m_Ruleset = new SingleHeadRuleSet();
        m_Ruleset.addRule(lastRule);

        final Instances data = (m_Filtered == null) ? m_Data : ((Instances[]) m_Filtered.get(m_Filtered.size() - 1))[1];
        final double[] stats = new double[6];
        double[] classCounts = null;
        try {
            classCounts = new double[m_Data.classAttribute().numValues()];
        } catch (final UnassignedClassException e) {
            e.printStackTrace();
        }
        final Instances[] filtered = computeSimpleStats(m_Ruleset.size() - 1, data, stats, classCounts);

        if (m_Filtered == null)
            m_Filtered = new ArrayList<Object>();
        m_Filtered.add(filtered);

        if (m_SimpleStats == null)
            m_SimpleStats = new ArrayList<Object>();
        m_SimpleStats.add(stats);

        if (m_Distributions == null)
            m_Distributions = new ArrayList<Object>();
        m_Distributions.add(classCounts);
    }

    /**
     * Subset description length: <br>
     * S(t,k,p) = -k*log2(p)-(n-k)log2(1-p)
     * <p>
     * Details see Quilan: "MDL and categorical theories (Continued)",ML95
     *
     * @param t the number of elements in a known set
     * @param k the number of elements in a subset
     * @param p the expected proportion of subset known by recipient
     * @return the subset description length
     */
    public static double subsetDL(final double t, final double k, final double p) {
        double rt = gr(p, 0.0) ? (-k * log2(p)) : 0.0;
        rt -= (t - k) * log2(1 - p);
        return rt;
    }

    /**
     * The description length of the theory for a given rule. Computed as:<br>
     * 0.5* [||k||+ S(t, k, k/t)]<br>
     * where k is the number of antecedents of the rule; t is the total possible antecedents that could appear in a rule; ||K|| is the universal prior for k , log2*(k) and S(t,k,p) = -k*log2(p)-(n-k)log2(1-p) is the subset encoding length.
     * <p>
     * <p>
     * Details see Quilan: "MDL and categorical theories (Continued)",ML95
     *
     * @param index the index of the given rule (assuming correct)
     * @return the theory DL, weighted if weight != 1.0
     */
    public double theoryDL(final int index) {

        // SingleHeadRule r = ((SingleHeadRule)m_Ruleset.getRule(index));

        final double k = m_Ruleset.getRule(index).length();

        if (k == 0)
            return 0.0;

        double tdl = log2(k);
        if (k > 1) // Approximation
            tdl += 2.0 * log2(tdl); // of log2 star
        tdl += subsetDL(m_Total, k, k / m_Total);
        // System.out.println("!!!theory: "+MDL_THEORY_WEIGHT *
        // REDUNDANCY_FACTOR * tdl);
        return MDL_THEORY_WEIGHT * REDUNDANCY_FACTOR * tdl;
    }

    /**
     * The description length of data given the parameters of the data based on the ruleset.
     * <p>
     * Details see Quinlan: "MDL and categorical theories (Continued)",ML95
     * <p>
     *
     * @param expFPOverErr expected FP/(FP+FN)
     * @param cover        coverage
     * @param uncover      uncoverage
     * @param fp           False Positive
     * @param fn           False Negative
     * @return the description length
     */
    public static double dataDL(final double expFPOverErr, final double cover, final double uncover, final double fp, final double fn) {
        final double totalBits = log2(cover + uncover + 1.0); // how many data?
        double coverBits, uncoverBits; // What's the error?
        double expErr; // Expected FP or FN

        if (gr(cover, uncover)) {
            expErr = expFPOverErr * (fp + fn);
            coverBits = subsetDL(cover, fp, expErr / cover);
            uncoverBits = gr(uncover, 0.0) ? subsetDL(uncover, fn, fn / uncover) : 0.0;
        } else {
            expErr = (1.0 - expFPOverErr) * (fp + fn);
            coverBits = gr(cover, 0.0) ? subsetDL(cover, fp, fp / cover) : 0.0;
            uncoverBits = subsetDL(uncover, fn, expErr / uncover);
        }

		/*
         * System.err.println("!!!cover: " + cover + "|uncover" + uncover + "|coverBits: "+coverBits+"|uncBits: "+ uncoverBits+ "|FPRate: "+expFPOverErr + "|expErr: "+expErr+ "|fp: "+fp+"|fn: "+fn+"|total: "+totalBits);
		 */
        return (totalBits + coverBits + uncoverBits);
    }

    /**
     * Calculate the potential to decrease DL of the ruleset, i.e. the possible DL that could be decreased by deleting the rule whose index and simple statstics are given. If there's no potentials (i.e. smOrEq 0 && error rate < 0.5), it returns NaN.
     * <p>
     * <p>
     * The way this procedure does is copied from original RIPPER implementation and is quite bizzare because it does not update the following rules' stats recursively any more when testing each rule, which means it assumes after deletion no data covered by the following rules (or regards the deleted rule as the last rule). Reasonable assumption?
     * <p>
     *
     * @param index        the index of the rule in m_Ruleset to be deleted
     * @param expFPOverErr expected FP/(FP+FN)
     * @param rulesetStat  the simple statistics of the ruleset, updated if the rule should be deleted
     * @param ruleStat     the simple statistics of the rule to be deleted
     * @param checkErr     whether check if error rate >= 0.5
     * @return the potential DL that could be decreased
     */
    public double potential(final int index, final double expFPOverErr, final double[] rulesetStat, final double[] ruleStat, final boolean checkErr) {
        // System.out.println("!!!inside potential: ");
        // Restore the stats if deleted
        final double pcov = rulesetStat[0] - ruleStat[0];
        final double puncov = rulesetStat[1] + ruleStat[0];
        final double pfp = rulesetStat[4] - ruleStat[4];
        final double pfn = rulesetStat[5] + ruleStat[2];

        final double dataDLWith = dataDL(expFPOverErr, rulesetStat[0], rulesetStat[1], rulesetStat[4], rulesetStat[5]);
        final double theoryDLWith = theoryDL(index);
        final double dataDLWithout = dataDL(expFPOverErr, pcov, puncov, pfp, pfn);

        final double potential = dataDLWith + theoryDLWith - dataDLWithout;
        final double err = ruleStat[4] / ruleStat[0];
		/*
		 * System.out.println("!!!"+dataDLWith +" | "+ theoryDLWith + " | " +dataDLWithout+"|"+ruleStat[4] + " / " + ruleStat[0]);
		 */
        boolean overErr = grOrEq(err, 0.5);
        if (!checkErr)
            overErr = false;

        if (grOrEq(potential, 0.0) || overErr) {
            // If deleted, update ruleset stats. Other stats do not matter
            rulesetStat[0] = pcov;
            rulesetStat[1] = puncov;
            rulesetStat[4] = pfp;
            rulesetStat[5] = pfn;
            return potential;
        } else
            return Double.NaN;
    }

    /**
     * Compute the minimal data description length of the ruleset if the rule in the given position is deleted.<br>
     * The min_data_DL_if_deleted = data_DL_if_deleted - potential
     *
     * @param index     the index of the rule in question
     * @param expFPRate expected FP/(FP+FN), used in dataDL calculation
     * @param checkErr  whether check if error rate >= 0.5
     * @return the minDataDL
     */
    public double minDataDLIfDeleted(final int index, final double expFPRate, final boolean checkErr) {
        // System.out.println("!!!Enter without: ");
        final double[] rulesetStat = new double[6]; // Stats of ruleset if deleted
        final int more = m_Ruleset.size() - 1 - index; // How many rules after?
        final ArrayList<double[]> indexPlus = new ArrayList<double[]>(more); // Their stats

        // 0...(index-1) are OK
        for (int j = 0; j < index; j++) {
            // Covered stats are cumulative
            rulesetStat[0] += ((double[]) m_SimpleStats.get(j))[0];
            rulesetStat[2] += ((double[]) m_SimpleStats.get(j))[2];
            rulesetStat[4] += ((double[]) m_SimpleStats.get(j))[4];
        }

        // Recount data from index+1
        Instances data = (index == 0) ? m_Data : ((Instances[]) m_Filtered.get(index - 1))[1];
        // System.out.println("!!!without: " + data.sumOfWeights());

        for (int j = (index + 1); j < m_Ruleset.size(); j++) {
            final double[] stats = new double[6];
            final Instances[] split = computeSimpleStats(j, data, stats, null);
            indexPlus.add(stats);
            rulesetStat[0] += stats[0];
            rulesetStat[2] += stats[2];
            rulesetStat[4] += stats[4];
            data = split[1];
        }
        // Uncovered stats are those of the last rule
        if (more > 0) {
            rulesetStat[1] = indexPlus.get(indexPlus.size() - 1)[1];
            rulesetStat[3] = indexPlus.get(indexPlus.size() - 1)[3];
            rulesetStat[5] = indexPlus.get(indexPlus.size() - 1)[5];
        } else if (index > 0) {
            rulesetStat[1] = ((double[]) m_SimpleStats.get(index - 1))[1];
            rulesetStat[3] = ((double[]) m_SimpleStats.get(index - 1))[3];
            rulesetStat[5] = ((double[]) m_SimpleStats.get(index - 1))[5];
        } else { // Null coverage
            rulesetStat[1] = ((double[]) m_SimpleStats.get(0))[0] + ((double[]) m_SimpleStats.get(0))[1];
            rulesetStat[3] = ((double[]) m_SimpleStats.get(0))[3] + ((double[]) m_SimpleStats.get(0))[4];
            rulesetStat[5] = ((double[]) m_SimpleStats.get(0))[2] + ((double[]) m_SimpleStats.get(0))[5];
        }

        // Potential
        double potential = 0;
        for (int k = index + 1; k < m_Ruleset.size(); k++) {
            final double[] ruleStat = indexPlus.get(k - index - 1);
            final double ifDeleted = potential(k, expFPRate, rulesetStat, ruleStat, checkErr);
            if (!Double.isNaN(ifDeleted))
                potential += ifDeleted;
        }

        // Data DL of the ruleset without the rule
        // Note that ruleset stats has already been updated to reflect
        // deletion if any potential
        final double dataDLWithout = dataDL(expFPRate, rulesetStat[0], rulesetStat[1], rulesetStat[4], rulesetStat[5]);
        // System.out.println("!!!without: "+dataDLWithout + " |potential: "+
        // potential);
        // Why subtract potential again? To reflect change of theory DL??
        return (dataDLWithout - potential);
    }

    /**
     * Compute the minimal data description length of the ruleset if the rule in the given position is NOT deleted.<br>
     * The min_data_DL_if_n_deleted = data_DL_if_n_deleted - potential
     *
     * @param index     the index of the rule in question
     * @param expFPRate expected FP/(FP+FN), used in dataDL calculation
     * @param checkErr  whether check if error rate >= 0.5
     * @return the minDataDL
     */
    public double minDataDLIfExists(final int index, final double expFPRate, final boolean checkErr) {
        // System.out.println("!!!Enter with: ");
        final double[] rulesetStat = new double[6]; // Stats of ruleset if rule exists
        for (int j = 0; j < m_SimpleStats.size(); j++) {
            // Covered stats are cumulative
            rulesetStat[0] += ((double[]) m_SimpleStats.get(j))[0];
            rulesetStat[2] += ((double[]) m_SimpleStats.get(j))[2];
            rulesetStat[4] += ((double[]) m_SimpleStats.get(j))[4];
            if (j == m_SimpleStats.size() - 1) { // Last rule
                rulesetStat[1] = ((double[]) m_SimpleStats.get(j))[1];
                rulesetStat[3] = ((double[]) m_SimpleStats.get(j))[3];
                rulesetStat[5] = ((double[]) m_SimpleStats.get(j))[5];
            }
        }

        // Potential
        double potential = 0;
        for (int k = index + 1; k < m_SimpleStats.size(); k++) {
            final double[] ruleStat = getSimpleStats(k);
            final double ifDeleted = potential(k, expFPRate, rulesetStat, ruleStat, checkErr);
            if (!Double.isNaN(ifDeleted))
                potential += ifDeleted;
        }

        // Data DL of the ruleset without the rule
        // Note that ruleset stats has already been updated to reflect deletion
        // if any potential
        final double dataDLWith = dataDL(expFPRate, rulesetStat[0], rulesetStat[1], rulesetStat[4], rulesetStat[5]);
        // System.out.println("!!!with: "+dataDLWith + " |potential: "+
        // potential);
        return (dataDLWith - potential);
    }

    /**
     * The description length (DL) of the ruleset relative to if the rule in the given position is deleted, which is obtained by: <br>
     * MDL if the rule exists - MDL if the rule does not exist <br>
     * Note the minimal possible DL of the ruleset is calculated(i.e. some other rules may also be deleted) instead of the DL of the current ruleset.
     * <p>
     *
     * @param index     the given position of the rule in question (assuming correct)
     * @param expFPRate expected FP/(FP+FN), used in dataDL calculation
     * @param checkErr  whether check if error rate >= 0.5
     * @return the relative DL
     */
    public double relativeDL(final int index, final double expFPRate, final boolean checkErr) {

        final double d1 = minDataDLIfExists(index, expFPRate, checkErr);
        final double d2 = theoryDL(index);
        final double d3 = minDataDLIfDeleted(index, expFPRate, checkErr);

        // System.out.println(d1 + " + " + d2 + " - " + d3);

        return d1 + d2 - d3;
    }

    /**
     * Try to reduce the DL of the ruleset by testing removing the rules one by one in reverse order and update all the stats
     *
     * @param expFPRate expected FP/(FP+FN), used in dataDL calculation
     * @param checkErr  whether check if error rate >= 0.5
     */
    public void reduceDL(final double expFPRate, final boolean checkErr) {

        boolean needUpdate = false;
        final double[] rulesetStat = new double[6];
        for (int j = 0; j < m_SimpleStats.size(); j++) {
            // Covered stats are cumulative
            rulesetStat[0] += ((double[]) m_SimpleStats.get(j))[0];
            rulesetStat[2] += ((double[]) m_SimpleStats.get(j))[2];
            rulesetStat[4] += ((double[]) m_SimpleStats.get(j))[4];
            if (j == m_SimpleStats.size() - 1) { // Last rule
                rulesetStat[1] = ((double[]) m_SimpleStats.get(j))[1];
                rulesetStat[3] = ((double[]) m_SimpleStats.get(j))[3];
                rulesetStat[5] = ((double[]) m_SimpleStats.get(j))[5];
            }
        }

        // Potential
        for (int k = m_SimpleStats.size() - 1; k >= 0; k--) {

            final double[] ruleStat = (double[]) m_SimpleStats.get(k);

            // rulesetStat updated
            final double ifDeleted = potential(k, expFPRate, rulesetStat, ruleStat, checkErr);
            if (!Double.isNaN(ifDeleted))
                if (k == (m_SimpleStats.size() - 1))
                    removeLast();
                else {
                    m_Ruleset.deleteRule(k);
                    needUpdate = true;
                }
        }

        if (needUpdate) {
            m_Filtered = null;
            m_SimpleStats = null;
            countData();
        }
    }

    /**
     * Remove the last rule in the ruleset as well as it's stats. It might be useful when the last rule was added for testing purpose and then the test failed
     */
    public void removeLast() {
        final int last = m_Ruleset.size() - 1;
        m_Ruleset.deleteRule(last);
        m_Filtered.remove(last);
        m_SimpleStats.remove(last);
        if (m_Distributions != null)
            m_Distributions.remove(last);
    }

    /**
     * Static utility function to count the data covered by the rules after the given index in the given rules, and then remove them. It returns the data not covered by the successive rules.
     *
     * @param data   the data to be processed
     * @param theory the ruleset
     * @param index  the given index
     * @return the data after processing
     */
    public static Instances rmCoveredBySuccessives(final Instances data, final SingleHeadRuleSet theory, final int index) {
        final Instances rt = new Instances(data, 0);

        for (int i = 0; i < data.numInstances(); i++) {
            final Instance datum = data.instance(i);
            boolean covered = false;

            for (int j = index + 1; j < theory.numRules(); j++) {
                final SingleHeadRule rule = theory.getRule(j);
                if (rule.covers(datum)) {
                    covered = true;
                    break;
                }
            }

            if (!covered)
                rt.add(datum);
        }
        return rt;
    }

    /**
     * Stratify the given data into the given number of bags based on the class values. It differs from the <code>Instances.stratify(int fold)</code> that before stratification it sorts the instances according to the class order in the header file. It assumes no missing values in the class.
     *
     * @param data           the given data
     * @param growingSetSize the growing set size
     * @param rand           the random object used to randomize the instances
     * @return the stratified instances
     * @throws UnassignedClassException
     * @throws UnassignedDatasetException
     */
    public static final Instances stratify(final Instances data, final double growingSetSize, final Random rand) throws UnassignedClassException, UnassignedDatasetException {

        final int folds = (int) Math.round(1 / (1 - growingSetSize));
        if (!data.classAttribute().isNominal())
            return data;

        final Instances result = new Instances(data, 0);
        final Instances[] bagsByClasses = new Instances[data.numClasses()];
        for (int i = 0; i < bagsByClasses.length; i++)
            bagsByClasses[i] = new Instances(data, 0);

        // Sort by class
        for (int j = 0; j < data.numInstances(); j++) {
            final Instance datum = data.instance(j);
            bagsByClasses[(int) datum.classValue()].add(datum);
        }

        // bagsByClasses is sorted by the number of instances, very important for RIPPER

        final int n = bagsByClasses.length;
        final Instances[] temp = new Instances[1];
        for (int pass = 1; pass < n; pass++)
            // This next loop becomes shorter and shorter
            for (int i = 0; i < n - pass; i++)
                if (bagsByClasses[i].numInstances() > bagsByClasses[i + 1].numInstances()) {
                    // exchange elements
                    temp[0] = bagsByClasses[i];
                    bagsByClasses[i] = bagsByClasses[i + 1];
                    bagsByClasses[i + 1] = temp[0];
                }

        // Randomize each class
        for (final Instances bagsByClasse : bagsByClasses)
            bagsByClasse.randomize(rand);

        for (int k = 0; k < folds; k++) {
            int offset = k, bag = 0;
            oneFold:
            while (true) {
                while (offset >= bagsByClasses[bag].numInstances()) {
                    offset -= bagsByClasses[bag].numInstances();
                    if (++bag >= bagsByClasses.length)// Next bag
                        break oneFold;
                }

                result.add(bagsByClasses[bag].instance(offset));
                offset += folds;
            }
        }

        return result;
    }

    /**
     * Compute the combined DL of the ruleset in this class, i.e. theory DL and data DL. Note this procedure computes the combined DL according to the current status of the ruleset in this class
     *
     * @param expFPRate expected FP/(FP+FN), used in dataDL calculation
     * @param predicted the default classification if ruleset covers null
     * @return the combined class
     */
    public double combinedDL(final double expFPRate, final double predicted) {
        double rt = 0;

        if (getRulesetSize() > 0) {
            final double[] stats = (double[]) m_SimpleStats.get(m_SimpleStats.size() - 1);
            for (int j = getRulesetSize() - 2; j >= 0; j--) {
                stats[0] += getSimpleStats(j)[0];
                stats[2] += getSimpleStats(j)[2];
                stats[4] += getSimpleStats(j)[4];
            }
            rt += dataDL(expFPRate, stats[0], stats[1], stats[4], stats[5]); // Data
            // DL
        } else { // Null coverage ruleset
            double fn = 0.0;
            for (int j = 0; j < m_Data.numInstances(); j++)
                try {
                    if ((int) m_Data.instance(j).classValue() == (int) predicted)
                        fn += m_Data.instance(j).weight();
                } catch (final UnassignedClassException e) {
                    e.printStackTrace();
                } catch (final UnassignedDatasetException e) {
                    e.printStackTrace();
                }
            rt += dataDL(expFPRate, 0.0, m_Data.sumOfWeights(), 0.0, fn);
        }

        for (int i = 0; i < getRulesetSize(); i++)
            // Theory DL
            rt += theoryDL(i);

        return rt;
    }

    /**
     * Patition the data into 2, first of which has (numFolds-1)/numFolds of the data and the second has 1/numFolds of the data
     *
     * @param data     the given data
     * @param numFolds the given number of folds
     * @return the patitioned instances
     */
    public static final Instances[] partition(final Instances data, final int numFolds) {
        final Instances[] rt = new Instances[2];
        final int splits = data.numInstances() * (numFolds - 1) / numFolds;

        rt[0] = new Instances(data, 0, splits);
        rt[1] = new Instances(data, splits, data.numInstances() - splits);

        return rt;
    }

    public static double log2(final double a) {

        return Math.log(a) / log2;
    }

    public static boolean gr(final double a, final double b) {

        return (a - b > SMALL);
    }

    public static boolean grOrEq(final double a, final double b) {

        return (b - a < SMALL);
    }
}
