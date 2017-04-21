/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * AbstractSeco.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Marc Ruppert Created on 29.06.2006
 */
package de.tu_darmstadt.ke.seco.algorithm.components.weightmodels;

import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import weka.core.Instance;

/**
 * @author Knowledge Engineering Group
 * 
 *         This is a Interface for the possibility to choose how to weight your examples. A Default one with non weighting is created to -> DefaultWeight
 * 
 *         Allmost sure it will result in a endless loop if the RuleStop doesn't consider the covering!
 */
public abstract class WeightModel extends SeCoComponent {

	private static final long serialVersionUID = -6550133418463777729L;

	/**
	 * The changeWeight method changes the weight of the examples covered by rule.
	 * 
	 * @param examples
	 *            The Examples where the weight have to be changed.
	 * @param covered
	 *            This is a container for counting how many rules cover a example.
	 * @param rule
	 *            The rule, that initiate the "covering" and so the new weights.
	 * @return an array of the weighted examples (0) and the cover counter (1)
	 * @throws Exception
	 */
	public abstract Instances[] changeWeights(Instances examples, Instances covered, SingleHeadRule rule) throws Exception;

	/**
	 * The calcNewWeight method calculates the weight of the delivered example
	 * 
	 * @param example
	 *            The example where the weight have to be updated.
	 * @param cover
	 *            The container-example with the count of covered rules.
	 * @return the weight of the example
	 * 
	 * @throws Exception
	 */
	public abstract double calcNewWeight(Instance example, Instance cover) throws Exception;
}
