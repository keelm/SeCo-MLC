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
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import de.tu_darmstadt.ke.seco.utils.Utils;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.Enumeration;

/**
 * CandidateRule is a subclass of Rule for candidate rules. A candidate rule contains additional information like - a slot for storing the result of the rule - the history of the rule (a pointer back to the predecessor)
 * <pruningDepth>
 * Parts of it is based on code for JRip and for Prism.
 *
 * @author Xin Xu
 * @author Eibe Frank
 * @author Ian H. Witten
 * @author Knowledge Engineering Group
 * @version $Revision: 277 $
 */
public class SingleHeadRule extends Rule {

	/**
	 * The head of the rule, a condition on the class attribute
	 */
	private Condition m_head;

	/* Constructor */
	public SingleHeadRule(Heuristic heuristic) {
		super(heuristic);
		this.m_head = null;
	}

	public SingleHeadRule(Heuristic heuristic, Condition head) {
		super(heuristic);
		this.m_head = head;
	}

	/**
	 * Creates a CandidateRule from the given instance
	 *
	 * @param example instance that will be transformed into a CandidateRule (must have access to a data set)
	 * @throws Exception
	 */
	public SingleHeadRule(final Heuristic heuristic, final Instance example) throws Exception {
		this(heuristic);

		final Instances dataset = (Instances) example.dataset(); // TODO by m.zopf: This is a design problem. One should not go from the interface instance to the collection, because information about the concrete instances in the collection is lost and we need a cast

		// Set head
		final Attribute headatt = dataset.classAttribute();
		if (headatt.isNominal())
			setHead(new NominalCondition(headatt, example.value(headatt)));
		else if (headatt.isNumeric())
			setHead(new NumericCondition(headatt, example.value(headatt)));
		else
			throw new Exception("only numeric and nominal attributes supported !");

		// Set body
		final Enumeration<Attribute> atts = dataset.enumerateAttributesWithoutClass();

		while (atts.hasMoreElements()) {

			final Attribute att = atts.nextElement();

			Condition cond;

			if (att.isNominal())
				cond = new NominalCondition(att, example.value(att));
			else if (att.isNumeric())
				cond = new NumericCondition(att, example.value(att));
			else
				throw new Exception("only numeric and nominal attributes supported !");

			addCondition(cond);

		}
	}

	@Override
	public Object copy() {
		Object copy = null;

		try {
			copy = super.clone();
		} catch (final CloneNotSupportedException e) {
			// should never happen
			e.printStackTrace();
			System.err.println(e.getMessage());
		}

		final SingleHeadRule r = (SingleHeadRule) copy;
		r.m_head = this.m_head;
		r.m_body = new ArrayList<Condition>(this.m_body);
		r.m_stats = (TwoClassConfusionMatrix) this.m_stats.clone();

		r.m_val = m_val;
		r.resetTieBreaker();

		return copy;
	}

	/**
	 * print out a candidate rule with coverage statistics and the heuristic value
	 * <pruningDepth>
	 * `@return a printable representation of the rule
	 */
	@Override
	public String toString() {
		String rule = null;
		if (m_head == null)
			rule = "No rule built yet.";
		else {
			rule = m_head.toString();

			if (m_body.size() > 0) {
				rule += " :- " + m_body.get(0).toString();
				for (int i = 1; i < m_body.size(); i++)
					rule += ", " + m_body.get(i).toString();
			}
		}

		rule += ". " + getStats();

		return rule + " Value: " + Utils.doubleToString(m_val, 3);
	}

	/**
	 * finds out whether this rule is semantically equivalent to another rule
	 *
	 * @param o the rule this rule is to be compared with
	 * @return true if and only if the two rules have the same head (if both heads are null, it counts as the same head) and their body contains the same conditions (the order of the conditions within the body does not matter, and neither do duplicate conditions)
	 */
	@Override
	public boolean equals(final Object o) {

		if (o instanceof SingleHeadRule) {
			final SingleHeadRule compRule = (SingleHeadRule) o;

			// If the rules are equal, the heads can either both be null, or be
			// equal
			final boolean headsNull = m_head == null && compRule.getHead() == null;
			final boolean headsEqual = !(m_head == null) && m_head.equals(compRule.getHead());

			final boolean sameHead = headsNull || headsEqual;

			// If all conditions in this rule's body also exist in the other
			// rules body and vice versa, their bodies are equal.
			final ArrayList<Condition> compBody = compRule.getBody();
			final boolean sameBody = compBody.containsAll(m_body) && m_body.containsAll(compBody);

			// If both heads and bodies of the rules are the same, the rules are
			// equal.
			return sameHead && sameBody;
		}

		return false;

	}

	/**
	 * set the head of the rule to a new class condition
	 */
	public void setHead(final Condition h) {
		m_head = h;
	}

	/**
	 * @return the class value predicted by the rule
	 */
	public double getPredictedValue() {
		return m_head.getValue();
	}

	/**
	 * classify the passed Instance, i.e. return the class value if the instance is covered by the rule, or the missing value if it is not covered by the rule.
	 *
	 * @param inst the instance
	 * @return the class of the instance or a missing value
	 */
	public double classifyInstance(final Instance inst) {
		// throws Exception {
		if (covers(inst))
			return m_head.getValue();
		else
			// if the rule don't math
			return weka.core.Utils.missingValue();
	}

	/**
	 * @return the head of the rule, a condition on the class attribute
	 */
	public Condition getHead() {
		return m_head;
	}

}