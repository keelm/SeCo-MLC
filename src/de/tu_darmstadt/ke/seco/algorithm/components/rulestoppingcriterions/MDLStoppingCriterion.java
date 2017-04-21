/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * MDLStoppingCriterion.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by David Schuld
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.core.Instance;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;
import de.tu_darmstadt.ke.seco.stats.RuleStats;

/**
 * A rule stopping criterion based on the minimum description length, as it is used in the JRip learner.
 *
 * @author Knowledge Engineering Group
 *
 */
public class MDLStoppingCriterion extends RuleStoppingCriterion implements Serializable {

	/**
	 * This is an implementation of the stopping criterion used in JRip.
	 *
	 * @author David Schuld
	 */

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;

	/** The limit of description length surplus in ruleset generation */
	private static double MAX_DL_SURPLUS = 64.0;

	private RuleStats rstats = null;

	private double m_Total = Double.NEGATIVE_INFINITY;

	private double m_classVal;

	private boolean firstCall = true;

	private Double[] orderedClasses, unorderedClasses;

	private double expFPRate, defDL;

	private double all = 0;

	private double lastClassValue = -1; // invalid class at the beginning, so (lastClassValue != classValue) is always true at the beginning

	@Override
	public boolean checkForRuleStop(final SingleHeadRuleSet theory, final SingleHeadRule rule, Instances examples, final Instances covered, final double classValue, final SeCoAlgorithm seCoAlgorithm) throws Exception {

		Instances orderedInstances = null;
		// on first call
		if (firstCall == true) {
			firstCall = false;
			m_Total = RuleStats.numAllConditions(examples);
			orderedInstances = examples.orderClasses();
			final int disVals = orderedInstances.numDistinctValues(orderedInstances.classAttribute());
			orderedClasses = new Double[disVals];
			unorderedClasses = new Double[disVals];
			final List<Double> s = new ArrayList<Double>();
			for (int i = 0; i < disVals; i++)
				s.add((double) orderedInstances.countInstances(i));

			unorderedClasses = s.toArray(unorderedClasses);
			// sort
			Collections.sort(s);
			orderedClasses = s.toArray(orderedClasses);
			for (final Double d : orderedClasses)
				all += d;

		}

		// check here if a new classVal was set --> this means that the
		// following
		// variables have to be recomputed (see
		// JRip.rulesetForOneClass()/buildClassifier())
		// -expFPRate
		// -defDL
		if (lastClassValue != classValue) {
			lastClassValue = classValue;

			rstats = new RuleStats();
			rstats.setNumAllConds(m_Total);
			rstats.setData(examples);

			final int y = (int) m_classVal;
			expFPRate = unorderedClasses[y] / all;

			double classYWeights = 0, totalWeights = 0;
			for (int j = 0; j < examples.numInstances(); j++) {
				final Instance datum = examples.instance(j);
				totalWeights += datum.weight();
				if ((int) datum.classValue() == y)
					classYWeights += datum.weight();
			}

			defDL = RuleStats.dataDL(expFPRate, 0.0, totalWeights, 0.0, classYWeights);
			all -= unorderedClasses[y];

		}

		double dl = defDL, minDL = defDL;
		double[] rst;
		boolean stop;

		rstats.addAndUpdate(rule);
		final int last = rstats.getRuleset().size() - 1; // Index of last rule
		dl += rstats.relativeDL(last, expFPRate, true);

		if (Double.isNaN(dl) || Double.isInfinite(dl))
			throw new Exception("Should never happen: dl in " + "building stage NaN or infinite!");

		if (dl < minDL)
			minDL = dl; // The best dl so far

		rst = rstats.getSimpleStats(last);

		stop = checkStop(rst, minDL, dl);

		examples = rstats.getFiltered(theory.numRules())[1];

		seCoAlgorithm.setInstances(examples);

		return stop;
	}

	/**
	 * Check whether the stopping criterion meets
	 *
	 * @param rst
	 *            the statistic of the ruleset
	 * @param minDL
	 *            the min description length so far
	 * @param dl
	 *            the current description length of the ruleset
	 * @return true if stop criterion meets, false otherwise
	 */
	public boolean checkStop(final double[] rst, final double minDL, final double dl) {

		return (dl > minDL + MAX_DL_SURPLUS) || (!RuleStats.gr(rst[2], 0.0)) || ((rst[4] / rst[0]) >= 0.5);
	}

	@Override
	public void setProperty(final String name, final String value) {
		if (name == "reset" && value == "true") { // TODO: "reset" is not a property. it is used for resetting some variables and not to configure the object
			lastClassValue = -1;
			firstCall = true;
		}
	}

	public RuleStats getRuleStats() {
		return rstats;
	}
}
