/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * DefaultRuleAndCoverageRuleStop.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Matthias Thiel Created on 17.11.2004, 19:14 Modified by Frederik Janssen
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions;

import java.io.Serializable;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Laplace;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import de.tu_darmstadt.ke.seco.utils.Logger;

/**
 * The DefaultRuleAndCoverageRuleStop checks, whether a given rule is better than the default rule according to a preset rule evaluator and a preset heuristic and whether the rule covers more positives than negatives.
 *
 * @author Frederik Janssen
 */
public class DefaultRuleAndCoverageRuleStop extends RuleStoppingCriterion implements Serializable {

	/**
	 * default serial UID
	 */
	private static final long serialVersionUID = 1L;

	private final Heuristic heuristic = new Laplace();

	/**
	 * Creates a new instance of HeuristicRuleStop Initializes with a DefaultRuleEvaluator using Laplace heuristic.
	 */
	public DefaultRuleAndCoverageRuleStop() {

	}

	/**
	 * Compares the given rule with the default rule using the preset rule evaluator.
	 *
	 * @param theory
	 *            The current theory.
	 * @param rule
	 *            The CandidateRule that is to be compared.
	 * @param examples
	 *            The training set.
	 * @return true, if it should stop, false otherwise.
	 */
	@Override
	public boolean checkForRuleStop(final SingleHeadRuleSet theory, final SingleHeadRule rule, final Instances examples, final Instances covered, final double classValue, final SeCoAlgorithm seCoAlgorithm) throws Exception {
		Logger.info("DefaultRuleEvaluator used with " + rule.getHeuristic().toString());
		final TwoClassConfusionMatrix stats = rule.getStats();
		SingleHeadRule defaultRule;
		defaultRule = new SingleHeadRule(rule.getHeuristic(), new NominalCondition(examples.classAttribute(), classValue));

		defaultRule.evaluateRule(examples, classValue, rule.getHeuristic());
		rule.evaluateRule(examples, classValue, heuristic);
		if (rule.length() == 0 || stats.getNumberOfTruePositives() <= stats.getNumberOfFalsePositives())
			return true; // rule stop, if there are no conditions or the rule
		// covers
		// more negatives than positives
		return SingleHeadRule.getBetterRule(rule, defaultRule) != rule;
	}
}
