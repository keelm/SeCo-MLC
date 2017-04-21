/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * NoOpStop.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Frederik Janssen
 */

package de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions;

import java.io.Serializable;
import java.lang.reflect.Field;

import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import weka.core.Instances;
import de.tu_darmstadt.ke.seco.utils.Logger;

/**
 * 
 * @author Knowledge Engineering Group no special stop, it just returns false
 */
public class NoOpStop extends StoppingCriterion implements Serializable {

	/**
     * 
     */
	private static final long serialVersionUID = 1L;

	/** Creates a new instance of DefaultStop */
	public NoOpStop() {
		Logger.info("DefaultStop");
	}

	@Override
	public boolean checkForStop(final SingleHeadRule refinement, final Instances examples) throws Exception {
		return false;
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
