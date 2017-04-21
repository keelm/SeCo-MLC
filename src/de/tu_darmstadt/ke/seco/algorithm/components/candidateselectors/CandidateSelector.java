/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * ICandidateSelector.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Matthias Thiel Created on 30.10.2004, 11:31
 */

package de.tu_darmstadt.ke.seco.algorithm.components.candidateselectors;

import java.util.TreeSet;

import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;

/**
 * An interface for implementations that select candidate rules for a separate and conquer algorithm.
 *
 * @author Knowledge Enginnering Group
 */
public abstract class CandidateSelector extends SeCoComponent {

	private static final long serialVersionUID = 2085218122602364434L;

	/**
	 * This will determine the CandidateRules for the next iteration of the algorithm.
	 *
	 * @param rules
	 *            A sorted set of rules.
	 * @param examples
	 *            The training set of examples.
	 * @return a set of CandidateRules
	 * @throws Exception
	 */
	public abstract SingleHeadRuleSet selectCandidates(TreeSet<SingleHeadRule> rules, Instances examples) throws Exception;
}
