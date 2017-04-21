/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * NumericCondition.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Johannes Fürnkranz
 *
 * Parts of the code adapted from JRip Copyright (C) 2001 Xin Xu, Eibe Frank Prism Copyright (C) 1999 Ian H. Witten
 */

package de.tu_darmstadt.ke.seco.models;

import de.tu_darmstadt.ke.seco.models.Attribute;
import weka.core.Instance;
import de.tu_darmstadt.ke.seco.utils.Utils;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning on top of weka.
 *
 * NumericCondition implements numeric conditions for rules
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
 * Conditions with numeric values
 */
public class NumericCondition extends Condition implements Cloneable {

	private static final long serialVersionUID = 8841503581326918047L;

	public NumericCondition(Attribute a) {
		super(a);
	}

	public NumericCondition(Attribute a, double value) {
		super(a, value);
	}

	public NumericCondition(Attribute a, double value, boolean cmp) {
		super(a, value, cmp);
	}

	/**
	 * check Whether the instance is covered by this condition
	 *
	 * @param inst
	 *            the instance in question
	 * @return the boolean value indicating whether the instance is covered by this antecedent
	 */
	@Override
	public boolean covers(Instance inst) {
		if (!inst.isMissing(m_att)) {
			/*
			 * return true if the instance value is < the stored value and the flag is true or if the instance value is >= the stored value and the flag is false
			 */
			// !== NOTE changed implementation to make it more JRip-like
			// if((inst.value(m_att) <= m_val) && m_cmp) {
			// return true;
			// } else if((inst.value(m_att) >= m_val) && !m_cmp) {
			// return true;
			// } else {
			// return false;
			// }

			if (m_cmp == true)
				return inst.value(m_att) <= m_val;
			else
				return inst.value(m_att) >= m_val;

		}
		else
			return false;
	}

	@Override
	public String toString() {
		return (m_att.name() + (m_cmp ? " <= " : " >= ") + Utils.doubleToString(m_val, 8));
	}
}
