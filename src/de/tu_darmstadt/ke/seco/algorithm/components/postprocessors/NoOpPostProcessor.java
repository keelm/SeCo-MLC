/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * NoOpPostProcessor.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Frederik Janssen Created on 05.07.2010
 */

package de.tu_darmstadt.ke.seco.algorithm.components.postprocessors;

import java.io.Serializable;
import java.lang.reflect.Field;

import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;

/**
 * The DefaultPostProcessor class is used to initialize a PostProcessor that does nothing, i.e., if no post processing is desired.
 *
 * @author Knowledge Engineering Group
 */

public class NoOpPostProcessor extends PostProcessor implements Serializable {

	// default constructor needed by the factory
	public NoOpPostProcessor() {
		super(null);
	}

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * essentially this method just returns the theory; it is not processed in any way
	 *
	 * @param theory
	 *            the theory to be processed
	 * @param newExamples
	 *            the examples the theory was learned from
	 * @return the processed theory
	 */
	@Override
	public SingleHeadRuleSet postProcessTheory(final SingleHeadRuleSet theory, final Instances newExamples, final double classValue) throws Exception {
		return theory;
	}

	// TODO by m.zopf: why is this empty?
	public void clone(final SeCoAlgorithm parent) {
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
