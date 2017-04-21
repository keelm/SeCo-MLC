/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * FoilGain.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by David Schuld
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

public abstract class GainHeuristic extends Heuristic {

	private static final long serialVersionUID = 222666004640432247L;

	/**
	 * return false because inheritances are GainHeuristics that require that the last rule of a refinement process is returned (see method returnBestRefinement in IRuleRefiner)
	 */
	@Override
	public boolean isValueHeuristic() {
		return false;
	}

}
