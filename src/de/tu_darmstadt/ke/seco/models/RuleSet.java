/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * RuleSet.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Johannes Fürnkranz Modified by Viktor Seifert
 */

package de.tu_darmstadt.ke.seco.models;

import weka.core.Instance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning.
 * <pruningDepth>
 * RuleSet implements the representation of a rule set.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 355 $
 */
public abstract class RuleSet<RuleType extends Rule> implements Serializable, Iterable<RuleType> {
    /**
     *
     */
    private static final long serialVersionUID = 1L;

    // the rules
    private ArrayList<RuleType> m_rules;

    // the default prediction;
    // TODO by m.zopf: the default rule here is something special. it may be better to put the default rule at the end of the m_rules list
    private RuleType m_default;

    // -------- Constructor ---------
    public RuleSet() {
        m_rules = new ArrayList<>();
    }

    // -------- Methods --------------

    /**
     * return all rules that cover a given instance
     *
     * @param inst the instance
     * @return a FastVector of rules that cover the instance
     */
    public abstract ArrayList<RuleType> getCoveringRules(final Instance inst);

    /**
     * return the first rule that covers a given instance or null if no instance covers the instance. The default rule is not tested.
     *
     * @param inst the instance
     * @return a the first rule that cover the instance
     */
    public abstract RuleType getFirstCoveringRule(final Instance inst);


    @Override
    public abstract String toString();

    /**
     * @return the rule set, a FastVector of Rules. Note that the default rule is <em>not</em> returned.
     */
    public final ArrayList<RuleType> getRules() {
        return m_rules;
    }

    /**
     * @return the final rule of the set
     */
    public final RuleType getLastRule() {
        return m_rules.get(m_rules.size() - 1);
    }

    /**
     * @param n the number of the rule should be returned
     * @return the nth rule
     */
    public final RuleType getRule(final int n) {
        return m_rules.get(n);
    }

    /**
     * @return the number of rules (excluding the default rule)
     */
    public final int numRules() {
        return m_rules.size();
    }

    /**
     * @return the number of conditions in the rules
     */
    public final int numConditions() {
        int n = 0;

        for (int i = 0; i < numRules(); i++)
            n += getRule(i).length();

        return n;
    }

    /**
     * Sums the number of attribute references in the rules which is the amount of different attributes that are to be tested when applying the rule.
     *
     * @return The number of attribute references in the rules.
     */
    public final int referredAttributes() {
        int n = 0;

        for (int i = 0; i < numRules(); i++)
            n += getRule(i).referredAttributes();

        return n;
    }

    // ==============================================================================================
    // not used any more

    /**
     * The sum of normalized rule lengths. The normaliation regards alternative rule representations that would shorten the rule.
     *
     * @return The sum of normalized rule lengths.
     */
    public final int normalizedLength() throws Exception {
        int n = 0;

        for (int i = 0; i < numRules(); i++)
            n += getRule(i).normalizedLength();

        return n;
    }

    /**
     * The average length of a rule is determined by dividing the sum of conditions by the sum of rules
     *
     * @return the average length of a rule (rounded to 2 decimal places)
     */

    public final double averageLength() {
        final int place = 2;
        final double temp = (double) numConditions() / (double) numRules();
        final double factor = Math.pow(10, place);
        return (Math.round(temp * factor) / factor);
    }

    /**
     * add a rule to the set
     *
     * @param c the rule to be added
     */
    public final void addRule(final RuleType c) {
        m_rules.add(c);
    }

    /**
     * add a collection of rules to the rule set
     *
     * @param c the rule to be added
     */
    public final void addAllRules(final Collection<RuleType> c) {
        m_rules.addAll(c);
    }

    /**
     * delete the last rule
     */
    public final void deleteLastRule() {
        m_rules.remove(m_rules.size() - 1);
    }

    /**
     * delete the nth rule
     *
     * @param n number of the rule to delete
     */
    public final void deleteRule(final int n) {
        m_rules.remove(n);
    }

    /**
     * replace the last rule with a new rule
     *
     * @param r new rule
     */
    public final void replaceLastRule(final RuleType r) {
        m_rules.set(m_rules.size() - 1, r);
    }

    /**
     * replace the nth rule with a new rule
     *
     * @param r new rule
     * @param n number of rule to replace
     */
    public final void replaceCondition(final RuleType r, final int n) {
        m_rules.set(n, r);
    }

    /**
     * get the default rule
     */

    public final RuleType getDefaultRule() {
        return m_default;
    }

    /**
     * set the default rule to a new rule
     *
     * @param r the new rule
     */
    public final void setDefaultRule(final RuleType r) {
        m_default = r;
    }

    protected int[] m_labelIndices = null;

    /**
     * is almost only used for the tostring method, in order to know if it actually was learned also on label features
     *
     * @param labelIndices
     */
    public final void setLabelIndices(int labelIndices[]) {
        m_labelIndices = labelIndices;
    }

    public final int[] getLabelIndices() {
        return m_labelIndices;
    }

    public final int size() {
        return m_rules.size();
    }

    /**
     * find out if two RuleSets are equal
     *
     * @param o the RuleSet this RuleSet will be compared to
     * @return true if and only if the two RuleSets are semantically equivalent, i.e. contain the same rules in arbitrary order
     */
    @Override
    public final boolean equals(final Object o) {

        if (o instanceof RuleSet) {
            final RuleSet compSet = (RuleSet) o;

            final ArrayList<RuleType> compRules = compSet.getRules();

            // two rule sets are equal if each of the two is a subset of the
            // other one
            return m_rules.containsAll(compRules) && compRules.containsAll(m_rules);
        }

        return false;
    }

    @Override
    public final Iterator<RuleType> iterator() {
        return m_rules.iterator();
    }

}