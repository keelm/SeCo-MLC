/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * IStoppingCriterion.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 30.10.2004, 12:31
 */

package de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions;

import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;
import weka.core.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;

/**
 * A stopping criterion may prevent further processing of a refined rule.
 * 
 * @author Knowledge Engineering Group
 */
public abstract class StoppingCriterion extends SeCoComponent {

	private static final long serialVersionUID = 3308139913497980541L;

	/**
	 * A stopping criterion may prevent further processing of a refined rule.
	 * 
	 * @param refinement
	 *            The current refined CandidateRule.
	 * @param examples
	 *            The training set.
	 * @return true, if the criterion is fulfilled, otherwise false
	 * @throws Exception
	 */
	public abstract boolean checkForStop(SingleHeadRule refinement, Instances examples) throws Exception;
}
