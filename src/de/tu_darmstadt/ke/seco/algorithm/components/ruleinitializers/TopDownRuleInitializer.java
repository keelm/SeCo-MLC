/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * TopDownRuleInitializer.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Matthias Thiel Created on 17.11.2004, 17:42
 */

package de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers;

import java.io.Serializable;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.utils.Logger;

/**
 * This initializer will create a first rule that classifies everything as the current target class.
 *
 * @author Knowledge Engineering Group
 */
public class TopDownRuleInitializer extends RuleInitializer implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;

	/** Creates a new instance of DefaultRuleInitializer */
	public TopDownRuleInitializer() {
		Logger.info("DefaultRuleInitializer used");
	}

	/**
	 * Generates the first CandidateRule that classifies all examples with the target class.
	 *
	 * @param examples
	 *            Training set.
	 * @return The CandidateRule.
	 */
	@Override
	public SingleHeadRule[] initializeRule(Heuristic heuristic, final Instances examples, final double classValue) throws Exception {
		final Condition head = new NominalCondition(examples.classAttribute(), classValue);
		final SingleHeadRule r = new SingleHeadRule(heuristic, head);
		final SingleHeadRule[] result = new SingleHeadRule[1];
		result[0] = r;
		return result;
	}
}
