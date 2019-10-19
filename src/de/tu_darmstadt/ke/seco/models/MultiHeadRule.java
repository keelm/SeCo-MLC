/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * CandidateRule.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Johannes Fürnkranz
 *
 * Parts of the code adapted from JRip Copyright (C) 2001 Xin Xu, Eibe Frank Prism Copyright (C) 1999 Ian H. Witten
 */

package de.tu_darmstadt.ke.seco.models;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import de.tu_darmstadt.ke.seco.utils.Utils;
import weka.core.Attribute;
import weka.core.Instance;

import java.io.Serializable;
import java.util.*;

public class MultiHeadRule extends Rule {

    public static class Head implements Iterable<Condition>, Cloneable, Serializable {

        private static final long serialVersionUID = -1993928248786099668L;

        private final Map<Integer, Condition> conditions;

        private Head(final Map<Integer, Condition> conditions) {
            this.conditions = conditions;
        }

        public Head() {
            this.conditions = new HashMap<>();
        }

        public Collection<Map.Entry<Integer, Condition>> entries() {
            return conditions.entrySet();
        }

        public Collection<Condition> getConditions() {
            return conditions.values();
        }

        public Collection<Integer> getLabelIndices() {
            return conditions.keySet();
        }

        public void clear() {
            conditions.clear();
        }

        public void addCondition(final Condition condition) {
            conditions.put(condition.getAttr().index(), condition);
        }

        public boolean containsCondition(final int attributeIndex) {
            return conditions.containsKey(attributeIndex);
        }

        public Condition getCondition(final int attributeIndex) {
            return conditions.get(attributeIndex);
        }

        public int size() {
            return conditions.size();
        }

        @Override
        public Iterator<Condition> iterator() {
            return getConditions().iterator();
        }

        @SuppressWarnings("CloneDoesntCallSuperClone")
        @Override
        public Head clone() {
            return new Head(new HashMap<>(conditions));
        }

        @Override
        public String toString() {
            return getConditions().toString();
        }

        @Override
        public int hashCode() {
            return conditions.hashCode();
        }

        @Override
        public boolean equals(final Object obj) {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            Head other = (Head) obj;
            return hashCode() == other.hashCode();
        }

    }

    private static final long serialVersionUID = -7921578297596174962L;

    private Head head;

    /* Constructor */
    public MultiHeadRule(final Heuristic heuristic) {
        super(heuristic);
        setHead(null);
    }
    
    /**
     * Sets the head of the rule.
     */
    public final void setHead(final Head head) {
        this.head = head;
    }

    /**
     * @return the head of the rule, multiple conditions on the label attributes
     */
    public Head getHead() {
        return head;
    }

    /**
     * @return the class values predicted by the rule
     */
    public double[] getPredictedValue() {
        double[] result = new double[head.size()];
        int index = 0;

        for (Condition condition : head) {
            result[index] = condition.getValue();
            index++;
        }

        return result;
    }

    /**
     * classify the passed Instance, i.e. return the class values if the instance is covered by the rule, or null if it
     * is not covered by the rule.
     *
     * @param inst the instance
     * @return the class of the instance or null
     */
    public double[] classifyInstance(final Instance inst) {
        if (covers(inst)) {
            return getPredictedValue();
        }

        return null;
    }

    @Override
    public Object copy() {
        Object copy = null;

        try {
            copy = super.clone();
        } catch (final CloneNotSupportedException e) {
            // should never happen
            e.printStackTrace();
            System.err.println(e.getMessage());
        }

        final MultiHeadRule r = (MultiHeadRule) copy;
        r.head = this.head != null ? this.head.clone() : null;
        r.m_body = new ArrayList<>(this.m_body);
        r.m_stats = (TwoClassConfusionMatrix) this.m_stats.clone();

        r.m_val = m_val;
        r.resetTieBreaker();

        return copy;
    }

    /**
     * print out a candidate rule with coverage statistics and the heuristic value <p> `@return a printable
     * representation of the rule
     */
    @Override
    public String toString() {
        String rule;
        if (head == null)
            rule = "No rule built yet.";
        else {
            rule = head.toString();

            if (m_body.size() > 0) {
                rule += " :- " + m_body.get(0).toString();
                for (int i = 1; i < m_body.size(); i++)
                    rule += ", " + m_body.get(i).toString();
            }
        }

        rule += ". " + getStats();
        return rule + " Value: " + Utils.doubleToString(m_val, 3);
    }

    /**
     * finds out whether this rule is semantically equivalent to another rule
     *
     * @param o the rule this rule is to be compared with
     * @return true if and only if the two rules have the same head (if both heads are null, it counts as the same head)
     * and their body contains the same conditions (the order of the conditions within the body does not matter, and
     * neither do duplicate conditions)
     */
    @Override
    public boolean equals(final Object o) {
        if (o instanceof MultiHeadRule) {
            final MultiHeadRule compRule = (MultiHeadRule) o;

            // If the rules are equal, the heads can either both be null, or be
            // equal
            final boolean headsNull = head == null && compRule.getHead() == null;
            final boolean headsEqual = !(head == null) && head.equals(compRule.getHead());

            final boolean sameHead = headsNull || headsEqual;

            // If all conditions in this rule's body also exist in the other
            // rules body and vice versa, their bodies are equal.
            final ArrayList<Condition> compBody = compRule.getBody();
            final boolean sameBody = compBody.containsAll(m_body) && m_body.containsAll(compBody);

            // If both heads and bodies of the rules are the same, the rules are
            // equal.
            return sameHead && sameBody;
        }

        return false;
    }
    
    /**
     * count the positive predicted labels of the rule
     * 
     * @param 
     * @return n number of one's in the head of the rule
     */
    
    public int getCardinality() {
    	int cardinality = 0;
    	for (Condition label : head) {
    		if (label.m_val == 1) {
    			cardinality++;
    		}
    	}
    	return cardinality;
    }
    

}