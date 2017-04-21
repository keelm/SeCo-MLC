/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * NominalCondition.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes Fürnkranz
 * 
 * Parts of the code adapted from JRip Copyright (C) 2001 Xin Xu, Eibe Frank Prism Copyright (C) 1999 Ian H. Witten
 */

package de.tu_darmstadt.ke.seco.models;

import weka.core.Instance;

// import weka.core.Instance;
// import de.tu_darmstadt.ke.seco.models.Attribute;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning on top of weka.
 *
 * NominalCondition implements nominal conditions for rules
 *
 * Parts of it is based on code for JRip and for Prism.
 *
 * @author Xin Xu
 * @author Eibe Frank
 * @author Ian H. Witten
 * @author Knowledge Engineering Group
 * @version $Revision: 104 $
 */

/**
 * Conditions with nominal values
 */
public class NominalCondition extends Condition implements Cloneable {

	private static final long serialVersionUID = 4266303550207026835L;

	/* Constructors */
	public NominalCondition(final Attribute a) {
		super(a);
	}

	public NominalCondition(final Attribute a, final double value) {
		super(a, value);
	}

	public NominalCondition(final Attribute a, final double value, final boolean cmp) {
		super(a, value, cmp);
	}

	/**
	 * check whether the instance is covered by this condition
	 *
	 * @param inst
	 *            the instance in question
	 * @return the boolean value indicating whether the instance is covered by this antecedent
	 */
	@Override
	public boolean covers(final Instance inst) {
		if (!inst.isMissing(m_att))
			/*
			 * return true if the instance value is == the stored value and the flag is true or if the instance value is != the stored value and the flag is false
			 */
			return ((inst.value(m_att) == m_val) == m_cmp);
		else
			return false;
	}

	@Override
	public String toString() {
		if (Double.isNaN(m_val))
			return (m_att.name() + (m_cmp ? " = " : " != ") + "?");
		return (m_att.name() + (m_cmp ? " = " : " != ") + m_att.value((int) m_val));
	}
}
