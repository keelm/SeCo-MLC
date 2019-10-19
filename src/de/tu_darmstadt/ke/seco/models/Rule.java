/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * CandidateRule.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Johannes Fürnkranz
 *
 * Parts of the code adapted from JRip Copyright (C) 2001 Xin Xu, Eibe Frank Prism Copyright (C) 1999 Ian H. Witten
 */

package de.tu_darmstadt.ke.seco.models;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.MulticlassCovering;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import weka.core.Instance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Iterator;

/**
 * CandidateRule is a subclass of Rule for candidate rules. A candidate rule contains additional information like - a slot for storing the result of the rule - the history of the rule (a pointer back to the predecessor)
 * <p>
 * Parts of it is based on code for JRip and for Prism.
 *
 * @author Xin Xu
 * @author Eibe Frank
 * @author Ian H. Witten
 * @author Knowledge Engineering Group
 * @version $Revision: 277 $
 */
public abstract class Rule implements Comparable<Rule>, Cloneable, Iterable<Condition>, Serializable {

    /**
     * the heuristic value of the rule
     */
    protected double m_val;

    /**
     * a random value for tie breaking
     */
    protected double m_tie;

    /**
     * a pointer to the predecessor of the candidate rule
     */
    //ELM: in again
    protected Rule m_pred;

    /**
     * the heuristic used to evaluate this CandidateRule
     */
    protected Heuristic heuristic;

    /**
     * The body of the rule, a list of conditions
     */
    protected ArrayList<Condition> m_body;

    /**
     * a 2x2 confusion matrix for maintaining coverage stats
     */
    protected TwoClassConfusionMatrix m_stats;

    /**
     * a 2x2 confusion matrix for maintaining recall stats    
     */
    protected TwoClassConfusionMatrix m_recall_stats;
    
    private Random m_rand = new Random(1);

    /* Constructor */
    public Rule(Heuristic heuristic) {
        initBody();
        m_val = Double.NaN;
        this.heuristic = heuristic;
        resetTieBreaker();
    }

    /**
     * @return a shallow copy of the candidate rule
     * <p>
     * The copy does not copy the conditions, while the clone does.
     */
    public abstract Object copy();

    /**
     * print out a candidate rule with coverage statistics and the heuristic value
     * <p>
     * `@return a printable representation of the rule
     */
    @Override
    public abstract String toString();

    /**
     * finds out whether this rule is semantically equivalent to another rule
     *
     * @param o the rule this rule is to be compared with
     * @return true if and only if the two rules have the same head (if both heads are null, it counts as the same head) and their body contains the same conditions (the order of the conditions within the body does not matter, and neither do duplicate conditions)
     */
    @Override
    public abstract boolean equals(final Object o);

    /**
     * check whether the rule covers an instance
     *
     * @param inst the instance to check
     * @return true if the rule covers the instance, false else
     */
	public final boolean covers(final Instance inst) {
		for (int i = 0; i < length(); i++) {
			final Condition c = getCondition(i);

			if (false) {
				if (!MulticlassCovering.cachedCovers(c, inst)) 
					return false;
			} else {
				if (!c.covers(inst))
					return false;
			}
		}

		return true;
	}

    /**
     * get the length of the rule, the number of conditions in the body
     */
    public final int length() {
        return m_body.size();
    }

    /**
     * @param n the number of the condition that should be returned
     * @return the nth condition of the body.
     */
    public final Condition getCondition(final int n) {
        return m_body.get(n);
    }

    /**
     * @return a deep copy of the candidate rule
     * <p>
     * the clone contains also contains fresh copies of the conditions and the coverage stats.
     */
    @Override
    public final Object clone() throws CloneNotSupportedException {
        final Object copy = super.clone();
        final Rule r = (Rule) copy;
        r.m_val = m_val;
        r.resetTieBreaker();
        return copy;
    }

    /**
     * reset the body of the rule to an empty vector and reset all statistics to 0.
     */
    public final void initBody() {
        m_body = new ArrayList<Condition>();
        m_stats = new TwoClassConfusionMatrix();
        m_val = Double.NEGATIVE_INFINITY;
        generalizationCount = 0;
        resetTieBreaker();
    }

    /**
     * return the better of two rules. It is assumed that computeRuleValue has been previously called! A rule is better if its evaluation is higher or if the evalution is the same, but it has shorter length, or the same length and a higher random tie break value. If one of the two is null, the other is returned.
     *
     * @param r1 the first candidate rule
     * @param r2 the second candidate rule
     * @return the better of the two rules
     */
    public static Rule getBetterRule(final Rule r1, final Rule r2) {

        // check that both are not null
        if (r1 == null)
            return r2;
        if (r2 == null)
            return r1;

        if (r1.compareTo(r2) > 0)
            return r1;
        else
            // we shouldn't have equal rules
            return r2;
    }

    
    protected int generalizationCount;
    
    public int getGeneralizationCount() {
		return generalizationCount;
	}

	public void setGeneralizationCount(int generalizationCount) {
		this.generalizationCount = generalizationCount;
	}
	
	public void increaseGeneralizationCount() {
		this.generalizationCount += 1;
	}

	/**
     * compare two rules. It is assumed that computeRuleValue has been previously called! A rule is better if its evaluation is higher or if the evalution is the same, but it has shorter length, or the same length and a higher random tie break value.
     *
     * @param r the rule to compare to
     * @return -1 if the r is better, 1 if the old rule is better, 0 else
     * @throws NullPointerException if the object is null
     */
    @Override
    public final int compareTo(final Rule r) throws NullPointerException {
        // type cast
        if (r == null)
            throw new ClassCastException("Null CandidateRule to compareTo.");

        // now compare the values
        final double val1 = this.getRuleValue();
        final double val2 = r.getRuleValue();
        if (val1 > val2)
            return 1;
        else if (val2 > val1)
            return -1;
        else if (this.getStats().getNumberOfTruePositives() > r.getStats().getNumberOfTruePositives())
            return 1;
        else if (this.getStats().getNumberOfTruePositives() < r.getStats().getNumberOfTruePositives())
            return -1;
        // which rule has been generalized more often (!= length for numerical attributes)
        else if (this.getGeneralizationCount() > r.getGeneralizationCount())
        	return 1;
        else if (this.getGeneralizationCount() < r.getGeneralizationCount())
        	return -1;
        // now tie break on length
        /*
        else if (this.length() < r.length())
            return 1;
        else if (r.length() < this.length())
            return -1;
            */
        // else
        // return this.toString().compareTo(r.toString()); // TODO by m.zopf: this version of compareTo is slower than the previous one, but also better. so find a fast and good solution.

        if (this.getTieBreaker() > r.getTieBreaker())
            return -1; // we are better
        else if (this.getTieBreaker() < r.getTieBreaker())
            return 1;
        else
            // we should only end up here if r and this are the same object
            return 0;
    }

    /*
     * set the recall TwoClassStats
     */
    public final TwoClassConfusionMatrix getRecallStats() {
    	return m_recall_stats;
    }    
    
    /*
     * set the recall TwoClassStats
     */
    public final void setRecallStats(final TwoClassConfusionMatrix confusionMatrix) {
    	m_recall_stats = confusionMatrix;
    }    
    
    /**
     * return the TwoClassStats object containing the coverage counts
     */
    public final TwoClassConfusionMatrix getStats() {
        return m_stats;
    }

    /**
     * set the TwoClassStats object to a new object
     */
    public final void setStats(final TwoClassConfusionMatrix confusionMatrix) {
        m_stats = confusionMatrix;
    }

    public final void evaluateRule(final Instances examples, final double m_classVal, Heuristic heuristic) throws Exception {
        final TwoClassConfusionMatrix confusionMatrix = new TwoClassConfusionMatrix();
        for (int i = 0; i < examples.numInstances(); i++) {
            final Instance inst = examples.instance(i);
            final boolean covered = covers(inst);
            final boolean shouldBeCovered = inst.classValue() == m_classVal;

            if (covered && shouldBeCovered)
                confusionMatrix.addTruePositives(inst.weight());
            else if (covered && !shouldBeCovered)
                confusionMatrix.addFalsePositives(inst.weight());
            else if (!covered && shouldBeCovered)
                confusionMatrix.addTrueNegatives(inst.weight());
            else
                confusionMatrix.addFalseNegatives(inst.weight());
        }
        setStats(confusionMatrix);
        computeRuleValue(heuristic);
    }

    public final void evaluateRuleForMultilabel(final Instances examples, final double m_classVal, Heuristic heuristic) throws Exception {
        final TwoClassConfusionMatrix confusionMatrix = new TwoClassConfusionMatrix();
        for (int i = 0; i < examples.numInstances(); i++) {
            final Instance inst = examples.instance(i);
            final boolean classValueAlreadySet = !weka.core.Utils.isMissingValue(inst.value(inst.classIndex())); //this should get something different then inst.classValue()
            //before, fn and tn did count, so that the consistency and especially the coverage was was somehow rewarded. now I try it out without, let's see
            if (classValueAlreadySet)
                continue;
            final boolean covered = covers(inst);
            final boolean shouldBeCovered = inst.classValue() == m_classVal;
            if (covered && shouldBeCovered) {
                if (!classValueAlreadySet)
                    confusionMatrix.addTruePositives(inst.weight());
            } else if (covered && !shouldBeCovered) {
                if (!classValueAlreadySet)
                    confusionMatrix.addFalsePositives(inst.weight());
            } else if (!covered && shouldBeCovered)
                confusionMatrix.addFalseNegatives(inst.weight());
            else
                confusionMatrix.addTrueNegatives(inst.weight());
        }
        setStats(confusionMatrix);
        computeRuleValue(heuristic);
    }

    public final double computeRuleValue(final Heuristic h) {
        this.setHeuristic(h);
        return m_val = h.evaluateRule(this);
    }

    /**
     * get the rule value that has been previously computed.
     */
    public final double getRuleValue() {
        return m_val;
    }

    public final void setRuleValue(final Heuristic h, final double value) {
        this.setHeuristic(h);
        this.m_val = value;
    }

    /**
     * reset the tie break value. This causes the next call to getTieBreaker to return a new value. resetTieBreaker should be called whenever the rule candidate changes (e.g., by adding or deleting conditions). This is *not* done automatically. resetTieBreaker is automatically called only when copying or constructing a rule candidate.
     */

    // currently m_tie is set to 1 in order to cancel out randomness
    // later we should make this configurable
    public final void resetTieBreaker() {
        m_tie = -1;
    }

    /**
     * get the predecessor of the candidate rule
     *
     * @return the predecessor or null
     */
    public final Rule getPredecessor() {
        //ELM: in again
//		return null; // TODO by m.zopf: because of performance reasons return here just null
        return m_pred;
    }

    /*
     * set the predecessor of the candidate rule
     *
     * @param r the predecessor
     */
    public final void setPredecessor(final Rule r) {
        //ELM: in again!
        m_pred = r;
    }

    public final void setHeuristic(final Heuristic heuristic) {
        this.heuristic = heuristic;
    }

    public final Heuristic getHeuristic() {
        return heuristic;
    }

    /**
     * return a specialization of the current rule. The specialization will be a fresh copy and the current rule will be its predecessor.
     *
     * @param c a condition
     * @return a specialization of the rule that results from adding c
     */
    public final Rule specialize(final Condition c) { // TODO by m.zopf: if a rule is specialized (or generalized) the stats (confusion matrix) should be set to zero or null because the stored value is no longer correct
        // initialize the candidate rule
        final Rule s = (Rule) this.copy();
        s.addCondition(c);
        s.setPredecessor(this);
        return s;
    }

    /**
     * add a condition to the body of the rule
     */
    public final void addCondition(final Condition c) {
        m_body.add(c);
    }

    @Override
    public final Iterator<Condition> iterator() {
        return m_body.iterator();
    }

    public final boolean containsCondition(final Condition cond) {
        for (final Condition c : this)
            if (c.equals(cond))
                return true;

        return false;
    }

    /**
     * return a generalization of this rule. The generalization will be a fresh copy and the current rule will be its predecessor.
     *
     * @param n number of the condition that will be deleted in order to generalize this rule
     * @return the generalization of this rule resulting from deletion of the n-th condition
     */
    public final Rule generalize(final int n) { // TODO by m.zopf: if a rule is specialized (or generalized) the stats (confusion matrix) should be set to zero or null because the stored value is no longer correct
        final Rule s = (Rule) this.copy();
        s.deleteCondition(n);
        s.setPredecessor(this);
        return s;
    }
    
    public final Rule generalizeNumeric (final int index, final double value) {
    	final Rule s = (Rule) this.copy();
    	Condition oldCond = new NumericCondition(s.getCondition(index).getAttr(), value, s.getCondition(index).cmp());
    	s.deleteCondition(index);
    	// oldCond.setValue(value);
    	s.addCondition(oldCond);
    	//s.getCondition(index).setValue(value);
    	s.setPredecessor(this);
    	return s;
    }

    /**
     * delete the nth condition from the body of the rule
     *
     * @param n number of the condition to delete
     */
    public final void deleteCondition(final int n) {
        m_body.remove(n);
    }

    /**
     * @return the body of the rule, a FastVector of Conditions
     */
    public final ArrayList<Condition> getBody() {
        return m_body;
    }

    /**
     * return the set of Instances that are not covered by the rule.
     *
     * @param data the set of instances
     * @return the set of instances taht are not covered.
     */
    public final Instances uncoveredInstances(final Instances data) {
        final Instances uncovd = new Instances(data, data.numInstances());
        final Enumeration<Instance> e = data.enumerateInstances();

        while (e.hasMoreElements()) {
            final Instance i = e.nextElement();
            if (!covers(i))
                uncovd.addDirectly(i);
        }

        uncovd.compactify();
        return uncovd;
    }

    /**
     * The set of nominal attributes that are to be tested by this rule. numeric attributes can be used more than once.
     *
     * @return The set of attributes belonging to this rule.
     */
    public final java.util.Set<Attribute> attributeSet_Nominal() {
        final java.util.Set<Attribute> set = new java.util.TreeSet<Attribute>();

        for (int i = 0; i < m_body.size(); i++) {
            final Condition c = m_body.get(i);
            if (c.getAttr().isNominal())
                set.add(c.getAttr());
        }

        return set;
    }

    /**
     * return the set of Instances that are covered by the rule.
     *
     * @param data the set of instances
     * @return the set of instances taht are not covered.
     */
    public final ArrayList<Instance> coveredInstances(final Instances data) {
        //TODO ELM: this fucks a lot, since it changes the reference to the dataset. actually, it should make a copy or something. ok, I put, that instances are added directly. this is a problem
        final ArrayList<Instance> covd = new ArrayList<>(data.numInstances());
        final Enumeration<Instance> e = data.enumerateInstances();

        while (e.hasMoreElements()) {
            final Instance i = e.nextElement();

            if (covers(i))
                covd.add(i);
        }

        return covd;
    }

    /**
     * return the set of Instances that are covered by the rule.
     *
     * @param data the set of instances
     * @return the set of instances taht are not covered.
     */
    public final Instances coveredInstancesNonCopying(final Instances data) {
        //TODO ELM: this fucks a lot, since it changes the reference to the dataset. actually, it should make a copy or something. ok, I put, that instances are added directly. this is a problem
        final Instances covd = new Instances(data, data.numInstances());
        final Enumeration<Instance> e = data.enumerateInstances();

        while (e.hasMoreElements()) {
            final Instance i = e.nextElement();

            if (covers(i))
                covd.addDirectly(i);
        }

        covd.compactify();
        return covd;
    }

    /**
     * The normalized rule length checks, whether the rule's body can be optimized by substitution of conditions with other conditions that have inverted cmp. It computes the rule length by regarding the inverted rule representation.
     *
     * @return The normalized rule length.
     */
    public final int normalizedLength() throws Exception {
        final HashSet<Attribute> set = attributeSet();
        final Iterator<Attribute> it = set.iterator();
        int result = 0;

        while (it.hasNext()) {
            final Attribute att = it.next();
            final int sumOfValues = att.numValues();

            if (att.isNominal()) {
                final int fVals = numValues(att, false);
                final int tVals = numValues(att, true);
                result += Math.min(fVals, sumOfValues - fVals) + Math.min(tVals, sumOfValues - tVals);
            } else if (att.isNumeric())
                result += numValues(att, true) + numValues(att, false);
            else
                throw new Exception("Attribute " + att + " is unsupported !");
        }

        return result;
    }

    /**
     * The set of attributes that are to be tested by this rule.
     *
     * @return The set of attributes belonging to this rule.
     */
    public final HashSet<Attribute> attributeSet() {
        final HashSet<Attribute> set = new HashSet<Attribute>();

        for (int i = 0; i < m_body.size(); i++) {
            final Condition c = m_body.get(i);
            set.add(c.getAttr());
        }

        return set;
    }

    /**
     * Determines the amount of different attributes that are to be tested when applying this rule. So only the mentioned attributes will be counted.
     *
     * @return Amount of referred attributes.
     */
    public final int referredAttributes() {
        return attributeSet().size();
    }

    private int numValues(final Attribute att, final boolean cmp) {
        final HashSet<Double> valSet = new HashSet<Double>();

        for (int i = 0; i < m_body.size(); i++) {
            final Condition c = m_body.get(i);

            if (c.getAttr().equals(att) && c.cmp() == cmp) {
                final double value = c.getValue();
                valSet.add(new Double(value));
            }
        }

        return valSet.size();
    }

    public final double getTieBreaker() {
        if (m_tie == -1)
            return m_tie = m_rand.nextDouble();
        else
            return m_tie;
    }

}