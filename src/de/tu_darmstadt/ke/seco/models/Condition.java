/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * Condition.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 * 
 * Parts of the code adapted from JRip Copyright (C) 2001 Xin Xu, Eibe Frank Prism Copyright (C) 1999 Ian H. Witten
 */

package de.tu_darmstadt.ke.seco.models;

import weka.core.Instance;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * This class implements conditions for rules.
 *
 * Parts of it is based on code for JRip and for Prism.
 *
 * @author Xin Xu
 * @author Eibe Frank
 * @author Ian H. Witten
 * @author Knowledge Engineering Group
 * @version $Revision: 141 $
 */
public abstract class Condition implements Serializable, Comparable<Condition> {

    private static final long serialVersionUID = 1147635219523348724L;

    /**
     * The attribute of the antecedent
     */
    protected Attribute m_att;

    /**
     * The attribute value of the antecedent. For numeric attribute, value is the value of the split point
     */
    protected double m_val;

    /**
     * A flag that indicates how the value should be used. The semantics of this flag depends on the subclass.
     */
    protected boolean m_cmp = true;

    /* Constructors */
    public Condition(Attribute a) {
        m_att = a;
    }

    public Condition(Attribute a, double value) {
        this(a);
        m_val = value;
    }

    public Condition(Attribute a, double value, boolean cmp) {
        this(a, value);
        m_cmp = cmp;
    }

    /* Get functions of this antecedent */
    public Attribute getAttr() {
        return m_att;
    }

    public double getValue() {
        return m_val;
    }

    public void setValue(double v) {
        m_val = v;
    }

    public boolean cmp() {
        return m_cmp;
    }

    public void setCmp(boolean c) {
        m_cmp = c;
    }

    /**
     * return the list of instances that are covered by the condition
     *
     * @param data the list of instances
     * @return a new list of covered instances
     */
    public Instances coveredInstances(Instances data) {
        /* Mostly borrowed from Prism.java */
        Instances covd = new Instances(data, data.numInstances());
        Enumeration<Instance> e = data.enumerateInstances();
        while (e.hasMoreElements()) {
            Instance i = e.nextElement();
            if (this.covers(i))
                covd.add(i);
        }
        covd.compactify();
        return covd;
    }

    public abstract boolean covers(Instance inst);

    @Override
    public abstract String toString();

    /**
     * implements Copyable
     *
     * @return a shallow copy of itself
     */
    public Object copy() {
        Object c = null;
        try {
            c = super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
            System.err.println(e.getMessage());
        }
        return c;
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = m_att != null ? (m_att.name() != null ? m_att.name().hashCode() : 0) : 0;
        temp = Double.doubleToLongBits(m_val);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (m_cmp ? 1 : 0);
        return result;
    }

    @Override
    public boolean equals(final Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Condition other = (Condition) obj;
        return hashCode() == other.hashCode();
    }

    @Override
    public int compareTo(Condition otherCond) {
        int boolIndex = this.m_att.index() - otherCond.m_att.index();

        if (boolIndex != 0)
            return boolIndex;
        else {
            double boolValueDiff = this.m_val - otherCond.m_val;
            if (boolValueDiff < 0)
                return -1;
            else if (boolValueDiff > 0)
                return 1;
            else if (this.m_cmp != otherCond.m_cmp)
                return this.m_cmp ? -1 : +1;
        }

        return 0;
    }

}
