/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * NoOpWeight.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Marc Ruppert Created on 29.06.2006
 */

package de.tu_darmstadt.ke.seco.algorithm.components.weightmodels;

import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.utils.Logger;
import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import weka.core.Instance;

import java.io.Serializable;
import java.lang.reflect.Field;

/**
 * This is the DefaultWeight implementation of a Weighter for the weighting of examples. Bye default it doesn't change the weight of the examples.
 * 
 * @author Knowledge Engineering Group
 */
public class NoOpWeight extends WeightModel implements Serializable {

	/**
     * 
     */
	private static final long serialVersionUID = 1L;

	// TODO by m.zopf: are m_multWeightfactor and m_addWeightfactor used by every weight model? then place it in WeightModel, otherwise remove it here.
	/**
	 * A factor for multiplicative increase/decrease of weight
	 */
	protected double m_multWeightfactor = 1.0;

	/**
	 * A factor for additve increase/decrease of weight
	 */
	protected double m_addWeightfactor = 0.5;

	public NoOpWeight() {
		Logger.info("DefaultWeight used");
	}

	/**
	 * The changeWeight from the DefaultWeight change the weight of the examples covered by rule. No changes yet, cause its the default weight. If you want to implement weighting, do a seperate implementation of the IWeightModell, that does the weighting.
	 * 
	 * @param examples
	 *            The Examples where the weight have to be changed
	 * @param rule
	 *            The rule, that initiate the "covering" and so the new weights
	 * 
	 * @return The weighted examples
	 * @throws Exception
	 */
	@Override
	public Instances[] changeWeights(final Instances examples, final Instances covered, final SingleHeadRule rule) throws Exception {
		// Nothing done here, so nothing to worry about
		// double newweight = calcNewWeight(examples.instance(counter),
		// covered.instance(counter));
		final Instances[] container = new Instances[2];
		container[0] = examples;
		container[1] = covered;
		return container;
	}

	/**
	 * The calcNewWeight method calculates the weight of the delivered example
	 * 
	 * @param example
	 *            The example where the weight have to be updated
	 * @param cover
	 *            The container-example with the count of covered rules
	 * 
	 * @return The weight of the Example
	 * 
	 * @throws Exception
	 * @author Marc Ruppert
	 */
	@Override
	public double calcNewWeight(final Instance example, final Instance cover) throws Exception {
		// No weighting so nothing to do,
		// method is never called :-)
		return example.weight();
	}

	/**
	 * This will be used for setting properties.
	 * 
	 * @param name
	 *            The name of the property.
	 * @param value
	 *            The value of the property.
	 */
	@Override
	public void setProperty(final String name, final String value) {
		if (name.equalsIgnoreCase("MultiValue"))
			m_multWeightfactor = Double.parseDouble(value);
		else if (name.equalsIgnoreCase("AddValue"))
			m_addWeightfactor = Double.parseDouble(value);
	}

	/**
	 * @return Returns the m_addWeightfactor.
	 */
	public double getM_addWeightfactor() {
		return m_addWeightfactor;
	}

	/**
	 * @param weightfactor
	 *            The m_addWeightfactor to set.
	 */
	public void setM_addWeightfactor(final double weightfactor) {
		m_addWeightfactor = weightfactor;
	}

	/**
	 * @return Returns the m_multWeightfactor.
	 */
	public double getM_multWeightfactor() {
		return m_multWeightfactor;
	}

	/**
	 * @param weightfactor
	 *            The m_multWeightfactor to set.
	 */
	public void setM_multWeightfactor(final double weightfactor) {
		m_multWeightfactor = weightfactor;
	}

	@Override
	public String toString() {
		return (new ReflectionToStringBuilder(this, ToStringStyle.SHORT_PREFIX_STYLE) {
			@Override
			protected boolean accept(final Field f) {
				return false; // exclude all fields from the toString
			}
		}).toString().replaceAll("\\[\\]", "");
	}
}
