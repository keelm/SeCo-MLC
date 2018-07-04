/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * TopDownRefiner.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by David Schuld
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.utils.Logger;
import de.tu_darmstadt.ke.seco.utils.NumericComparator;
import weka.core.Instance;
import weka.core.UnassignedClassException;
import weka.core.UnassignedDatasetException;

import java.util.*;

/**
 * Generic refiner class that refines rule in a top-down manner, i.e. it specialises a general rule until certain stopping criteria are met, used in learning algorithms such as JRip or Bexa.
 *
 * @author Knowledge Engineering Group
 */
public class TopDownRefiner extends RuleRefiner {

    private static final long serialVersionUID = 7986005475771734126L;

    public enum NominalCompareMode {

        EQUALITY("equal"), INEQUALITY("unequal"), BOTH("both");

        private String stringRepresentation;

        private NominalCompareMode(String stringRepresentation) {
            this.stringRepresentation = stringRepresentation;
        }

        public static NominalCompareMode parse(final String nominalCompareMode) throws IllegalArgumentException, NullPointerException {
            if (nominalCompareMode == null)
                throw new NullPointerException();
            else if (nominalCompareMode.equalsIgnoreCase(EQUALITY.stringRepresentation))
                return EQUALITY;
            else if (nominalCompareMode.equalsIgnoreCase(INEQUALITY.stringRepresentation))
                return INEQUALITY;
            else if (nominalCompareMode.equalsIgnoreCase(BOTH.stringRepresentation))
                return BOTH;
            else
                throw new IllegalArgumentException();
        }

        @Override
        public String toString() {
            return stringRepresentation;
        }
    }

    /**
     * The kind of nominal conditions that should be created. Influences the cmp setting of Class Condition and NominalCondition
     * <pruningDepth>
     * Divided into two booleans:
     * <pruningDepth>
     * m_nominalCmpModeUseEquality decides whether Equality (==) should be used or not, then Inequality is used (!=)
     * <pruningDepth>
     * m_nominalCmpModeUseBoth decides whether both Equality (==) and Inequality (!=) should be used or only one of them
     */

    @ConfigurableProperty
    private NominalCompareMode nominalCompareMode = NominalCompareMode.EQUALITY;

    @Override
    public SingleHeadRuleSet refineRule(final SingleHeadRule c, final Instances examples, final double classValue) throws Exception {

        if (heuristic == null)
            heuristic = c.getHeuristic();
        clearRefinements();
        int g = 0;
        final Set<Condition> usable = createUsableConditions(c, examples, classValue);
        filterUsableConditions(c, usable, examples, classValue);
        final Iterator<Condition> it = usable.iterator();
        HashSet<Instance> coveredInstancesByBaseRule = new HashSet<>();

        for (Instance instance : examples)
            if (c.covers(instance))
                coveredInstancesByBaseRule.add(instance);

        while (it.hasNext()) {
            final Condition cond = it.next();

            final SingleHeadRule newRule = (SingleHeadRule) c.specialize(cond);
            // if (beamwidth != null)
            evaluateRule(newRule, cond, examples, coveredInstancesByBaseRule, classValue);
            // evaluateRule(newRule, examples, classValue);

            addToRefinements(newRule);

        }

        SingleHeadRuleSet refinements = getRefinementsAsRuleSet();
        clearRefinements();
        return refinements;
    }

    // TODO by m.zopf: this method is already implemented in SingleHeadRule and should be reused
    protected void evaluateRule(SingleHeadRule r, final Condition conditionAddedToBaseRule, final Instances examples, HashSet<Instance> coveredInstancesByBaseRule, final double classValue) throws Exception {
        //TODO ELM: THIS WORKS?!?!?!?!?! i just replaced the function, but these hashes are not used anymore
        r.evaluateRuleForMultilabel(examples, classValue, heuristic);
//		double tp = 0;
//		double fp = 0;
//		double tn = 0;
//		double fn = 0;
//
//		for (int i = 0; i < examples.numInstances(); i++) {
//			final Instance inst = examples.instance(i);
//			final double w = inst.weight();
//			int counterCase = 0;
//
//			// TicksPerSecondCounter.globalTicksPerSecondCounter.tick();
//			if (!coveredInstancesByBaseRule.contains(inst) || !conditionAddedToBaseRule.covers(inst))
//				counterCase += 2;
//			if (inst.classValue() != classValue)
//				counterCase++;
//			switch (counterCase) {
//			case 0:
//				tp += w;
//				break;
//			case 1:
//				fp += w;
//				break;
//			case 2:
//				fn += w;
//				break;
//			case 3:
//				tn += w;
//				break;
//			}
//		}
//		final TwoClassConfusionMatrix tcs = new TwoClassConfusionMatrix(tp, fn, fp, tn);
//		r.setStats(tcs);
//		r.computeRuleValue(heuristic);

        // System.out.println("evaluated rule: " + r);S
    }

    private Set<Condition> createUsableConditions(final SingleHeadRule c, final Instances examples, final double classValue) throws Exception {

        final Set<Condition> usable = new TreeSet<Condition>();
        final Enumeration<Attribute> atts = examples.enumerateAttributesWithoutClass();
        final Set<Attribute> usedAtts = c.attributeSet_Nominal();
        Attribute att;

        while (atts.hasMoreElements()) {
            att = atts.nextElement();
            if (usedAtts.contains(att))
                continue;

            if (att.isNominal()) {
                final int numVal = att.numValues();
                for (int i = 0; i < numVal; i++) {

                    if (nominalCompareMode == NominalCompareMode.EQUALITY || nominalCompareMode == NominalCompareMode.BOTH)
                        usable.add(new NominalCondition(att, i, true));
                    if (nominalCompareMode == NominalCompareMode.INEQUALITY || nominalCompareMode == NominalCompareMode.BOTH)
                        usable.add(new NominalCondition(att, i, false));
                }
            } else if (att.isNumeric()) {

                final TreeSet<Instance> ts = new TreeSet<Instance>(new NumericComparator(att)); // TODO by m.zopf: Maybe here should an inverse comparator be used. Does the ordering of the set matter?
                final Enumeration<Instance> en = examples.enumerateInstances();
                while (en.hasMoreElements()) {
                    final Instance inst = en.nextElement();
                    if ((!inst.isMissing(att)) && (inst.weight() == 1.0))
                        ts.add(inst);
                }
                final Iterator<Instance> it = ts.iterator();
                Instance lastInst = null;
                while (it.hasNext()) {
                    final Instance inst = it.next();
                    if (lastInst != null) {
                        final double cVal = inst.classValue();
                        final double lastVal = lastInst.classValue();
                        if ((cVal == classValue && lastVal != classValue) || (cVal != classValue && lastVal == classValue)) {
                            final double v1 = inst.value(att);
                            final double v2 = lastInst.value(att);
                            final double val = v2 + (v1 - v2) / 2;
                            final Condition cond1 = new NumericCondition(att, val, false);
                            final Condition cond2 = new NumericCondition(att, val, true);
                            usable.add(cond1);
                            usable.add(cond2);
                        }
                    }
                    lastInst = inst;
                }
            } else
                throw new Exception("only numeric and nominal attributes supported !");

        }

        return usable;

    }

    public void filterUsableConditions(final SingleHeadRule c, final Set<Condition> usable, final Instances examples, final double classValue) {
        final ArrayList<Instance> covered = c.coveredInstances(examples);
        final Iterator<Condition> it = usable.iterator();
        while (it.hasNext()) {
            final Condition cond = it.next();
            final Enumeration<Instance> en = Collections.enumeration(covered);
            boolean uncoverNegative = false;
            boolean remove = false;
            boolean coversPositive = false;
            //TODO ELM: what is done here?? do I have to change it since I changed how to count true positives?
            while (en.hasMoreElements() && !(coversPositive && uncoverNegative)) {
                final Instance inst = en.nextElement();

                try {
                    if (inst.classValue() == classValue) { // if inst is in X_p
                        // prevent conditions that uncover positive examples
                        // if ( !cond.covers(inst) ) remove = true;
                        if (!coversPositive && cond.covers(inst))
                            coversPositive = true;
                    } else if (!uncoverNegative && !cond.covers(inst))
                        uncoverNegative = true;
                } catch (final UnassignedClassException ex) {
                    System.err.println(ex.getMessage());
                    ex.printStackTrace();
                    System.exit(-1);
                } catch (final UnassignedDatasetException ex) {
                    System.err.println(ex.getMessage());
                    ex.printStackTrace();
                    System.exit(-1);
                }
            }

            if (!uncoverNegative)
                remove = true;
            if (!coversPositive)
                remove = true;
            if (remove)
                it.remove();
        }

        // irredundancy restriction !!!
    }

    @Override
    public void setProperty(final String name, final String value) {
        super.setProperty(name, value);

        if (name.equalsIgnoreCase("nominal.cmpmode"))
            try {
                nominalCompareMode = NominalCompareMode.parse(value);
            } catch (final Exception e) {
                Logger.warn("Illegal value for 'nominal.cmpmode'. Using default '" + NominalCompareMode.EQUALITY + "'.");
                nominalCompareMode = NominalCompareMode.EQUALITY;
            }
    }

    @Override
    public String getPropertiesString() {
        String propertiesString = super.getPropertiesString();

        if (propertiesString.length() > 0)
            propertiesString = propertiesString + " ";

        return propertiesString + "nominal.cmpmode=\"" + nominalCompareMode.toString() + "\"";
    }
}
