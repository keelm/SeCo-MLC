/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * IRuleInitializer.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 30.10.2004, 11:06
 */

package de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers;

import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;

/**
 * Interface for a rule initializer that is called at the beginning of the Find-Best-SingleHeadRule method.
 *
 * @author Knowledge Engineering Group
 */
public abstract class RuleInitializer extends SeCoComponent {

	private static final long serialVersionUID = -5476770621058427933L;

	/**
	 * This will determine an initial rule.
	 *
	 * @param examples
	 *            The training set.
	 * @return the first CandidateRule
	 * @throws Exception
	 */
	public abstract SingleHeadRule[] initializeRule(Heuristic heuristic, Instances examples, double classValue) throws Exception;

}
