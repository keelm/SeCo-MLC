/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * IPostProcessor.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 29.10.2004, 17:19
 */

package de.tu_darmstadt.ke.seco.algorithm.components.postprocessors;

import java.lang.reflect.Field;

import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;
import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;
import de.tu_darmstadt.ke.seco.models.Instances;

/**
 * The interface for an implementation that will perform postprocessing on the learned theory.
 *
 * @author Knowledge Engineering Group
 */
public abstract class PostProcessor extends SeCoComponent {

	private static final long serialVersionUID = 5694870047893740016L;
	// TODO by m.zopf: This should not be in the PostProcessor because only PostProcessorRipper needs this. So move it to PostProcessorRipper.
	protected final SeCoAlgorithm seCoAlgorithm;

	// default constructor needed by the factory
	public PostProcessor() {
		seCoAlgorithm = new SeCoAlgorithm(null);
	}

	public PostProcessor(final SeCoAlgorithm seCoAlgorithm) {
		this.seCoAlgorithm = seCoAlgorithm;
	}

	/**
	 * Processes the given theory after learning.
	 *
	 * @param theory
	 *            The learned theory.
	 * @param newExamples
	 *            The examples on which the theory was learned.
	 * @return the postprocessed theory
	 * @throws Exception
	 */

	public abstract SingleHeadRuleSet postProcessTheory(SingleHeadRuleSet theory, Instances newExamples, double classValue) throws Exception;

	@Override
	public String toString() {
		return (new ReflectionToStringBuilder(this, ToStringStyle.SHORT_PREFIX_STYLE) {
			@Override
			protected boolean accept(final Field f) {
				return super.accept(f) && !f.getName().equals("seCoAlgorithm"); // exclude seCoAlgorithm from the toString because of the length
			}
		}).toString();
	}
}
