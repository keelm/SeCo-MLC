/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * SelectAllCandidatesSelector.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Matthias Thiel Created on 17.11.2004, 16:19
 */

package de.tu_darmstadt.ke.seco.algorithm.components.candidateselectors;

import java.io.Serializable;
import java.util.TreeSet;

import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;
import de.tu_darmstadt.ke.seco.utils.Logger;

/**
 * The default candidate selector for the most common cases.
 *
 * @author Knowledge Engineering Group
 */
public class SelectAllCandidatesSelector extends CandidateSelector implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;

	/** Creates a new instance of DefaultSelector */
	public SelectAllCandidatesSelector() {
		Logger.info("DefaultSelector used");
	}

	/**
	 * Selects all candidates.
	 *
	 * @param rules
	 *            The rule set.
	 * @param examples
	 *            The training set.
	 */
	@Override
	public SingleHeadRuleSet selectCandidates(final TreeSet<SingleHeadRule> rules, final Instances examples) throws Exception {
		// just select all given rules
		final SingleHeadRuleSet set = new SingleHeadRuleSet();
		final Object[] ruleArray = rules.toArray();
		for (final Object element : ruleArray)
			set.addRule((SingleHeadRule) element);
		return set;
	}
}
