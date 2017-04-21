/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * CoverageRuleStop.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 17.11.2004, 19:14 Modified by Frederik Janssen
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions;

import java.io.Serializable;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The CoverageRuleStop checks, whether a given rule covers more positive than negative examples.
 *
 * @author Knowledge Engineering Group
 */
public class CoverageRuleStop extends RuleStoppingCriterion implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Creates a new instance of CoverageRuleStop
	 */
	public CoverageRuleStop() {
	}

	/**
	 * Tests whether the Candidate SingleHeadRule covers more positive than negative examples.
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
		final TwoClassConfusionMatrix stats = rule.getStats();
		if (stats.getNumberOfTruePositives() <= stats.getNumberOfFalsePositives())
			return true;
		else
			return false;
	}
}
