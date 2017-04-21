/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * NoNegativesCoveredStop.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by David Schuld Created on 29. Oktober 2004, 17:30
 */

package de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions;

import java.io.Serializable;

import weka.core.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;

/**
 * Stopping Criterion for JRip and others. Stops adding conditions to a rule when there are no more negative examples covered.
 * 
 * @author Knowledge Engineering Group
 * 
 */
public class NoNegativesCoveredStop extends StoppingCriterion implements Serializable {

	private static final long serialVersionUID = 1L;

	@Override
	public boolean checkForStop(final SingleHeadRule refinement, final Instances examples) throws Exception {
		return refinement.getStats().getNumberOfFalsePositives() == 0;
	}
}
