package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.learners.core.SeCoClassifier;
import de.tu_darmstadt.ke.seco.learners.core.SeCoClassifierFactory;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Utils;

public class MultilabelSecoClassifier extends SeCoClassifier implements MultiLabelLearner {

    protected int[] m_labelindices;
    protected SeCoAlgorithm m_secoAlgo;
    protected MultilabelSecoClassifier internalClassifier;


    public MultilabelSecoClassifier(int labelindices[], SeCoAlgorithm secoAlgo) {
        m_labelindices = labelindices;
        m_secoAlgo = secoAlgo;
    }

    public MultilabelSecoClassifier() {

    }

    public MultilabelSecoClassifier(String id) {
        super(id);
    }

    @Override
    public boolean isUpdatable() {
        return false;
    }

    @Override
    public void build(MultiLabelInstances instances) throws Exception {
        internalClassifier = SeCoClassifierFactory
                .buildSeCoClassifierMultilabel(m_secoAlgo, Instances.toSeCoInstances(instances.getDataSet()),
                        m_labelindices, classifyMethod);
        m_theory = internalClassifier.m_theory;
    }

    @Override
    public MultiLabelLearner makeCopy() throws Exception {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        Instance inst = (Instance) instance.copy();
        //System.out.println("######################################" + classifyMethod + "######################");
        //ensures that no label is set before. could also be not a so great idea...
        for (int i = 0; i < m_labelindices.length; i++)
            inst.setValue(m_labelindices[i], Utils.missingValue());
        if (SeCoAlgorithm.DEBUG_STEP_BY_STEP) {
            //System.out.println("######instance to classify:\n" + inst);
        }
        for (int i = 0; i < m_theory.numRules(); i++) {
            Rule rule = m_theory.getRule(i);
            //// CHECK ORDER OF THEORY
            // System.out.println(m_theory);
            ////
            if (rule instanceof SingleHeadRule) {
                SingleHeadRule singleHeadRule = (SingleHeadRule) rule;

                if ("magicSkipHead".equals(singleHeadRule.getHead().getAttr().name())) {
                    continue; //should not happen, I add id anyways
                }
                boolean predicted = false;
                if (singleHeadRule.covers(inst)) {
                    if (SeCoAlgorithm.DEBUG_STEP_BY_STEP)
                        System.out.println("rule fires: " + rule);
                    Condition head = singleHeadRule.getHead();
                    if (Utils.isMissingValue(inst.value(head.getAttr().index()))) {
                        inst.setValue(head.getAttr(),
                                head.getValue()); //value will always be 1.0 in the beginning setting
                        predicted = true;
                        if (SeCoAlgorithm.DEBUG_STEP_BY_STEP)
                            System.out.println("label is set since not set before, new instance:\n " + inst);
                    }
                }
                if (i < m_theory.numRules() - 1) {
                    SingleHeadRule nextRule = (SingleHeadRule) m_theory.getRule(i + 1);
                    if ("magicSkipHead".equals(nextRule.getHead().getAttr().name())) {
                        i++; //skip this next rule, do everything we have to do here
                        if (predicted) {
                            if (SeCoAlgorithm.DEBUG_STEP_BY_STEP)
                                System.out.println("stopping rule found and applies, stop here: ");
                            break; //then, end here!!
                        }
                        //otherwise continue in the rule list
                    }
                }
            } else {
                MultiHeadRule multiHeadRule = (MultiHeadRule) rule;

                if (multiHeadRule.getHead().size() > 0 &&
                        "magicSkipHead".equals(multiHeadRule.getHead().iterator().next().getAttr().name())) {
                    continue; // Should not happen
                }
                boolean predicted = false;
                if (multiHeadRule.covers(inst)) {
                    if (SeCoAlgorithm.DEBUG_STEP_BY_STEP) {
                        System.out.println("rule fires: " + rule);
                    }
                    Head head = multiHeadRule.getHead();

                    for (Condition condition : head) {
                        if (Utils.isMissingValue(inst.value(condition.getAttr().index()))) {
                            inst.setValue(condition.getAttr().index(), condition.getValue());
                            predicted = true;
                            if (SeCoAlgorithm.DEBUG_STEP_BY_STEP) {
                                System.out.println("label is set since not set before, new instance:\n " + inst);
                            }
                        }
                    }
                    
                    if (classifyMethod.equals("DecisionList")) {
                    	break;
                    }
                }
                /* no stopping criterion is used in this implementation
                if (i < m_theory.numRules() - 1) {
                    MultiHeadRule nextRule = (MultiHeadRule) m_theory.getRule(i + 1);
                    if (multiHeadRule.getHead().size() > 0 &&
                            "magicSkipHead".equals(nextRule.getHead().iterator().next().getAttr().name())) {
                        i++; // skip this next rule, do everything we have to do here
                        if (predicted) {
                            if (SeCoAlgorithm.DEBUG_STEP_BY_STEP)
                                System.out.println("stopping rule found and applied, stop here: ");
                            break; // then, end here!!
                        }
                        // otherwise continue in the rule list
                    }
                }
                */
            }
        }
        if (SeCoAlgorithm.DEBUG_STEP_BY_STEP)
            System.out.println("output:\n" + inst);
        boolean[] bipartition = new boolean[m_labelindices.length];
        for (int i = 0; i < m_labelindices.length; i++) {
            double pred = inst.value(m_labelindices[i]);

            if (pred == 1.0) {
                bipartition[i] = true;
            } else bipartition[i] = false;
        }

        return new MultiLabelOutput(bipartition);
    }

    @Override
    public void setDebug(boolean debug) {
        // TODO Auto-generated method stub
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("\nuncoveredInstsRate...: " + m_secoAlgo.getUncoveredInstancesPercentage());
        stringBuilder.append("\npredictZerosInHeads..: " + m_secoAlgo.isPredictZero());
        stringBuilder
                .append("\nuseSkippingRules.....: " + (m_secoAlgo.getSkipThresholdPercentage() < 0 ? "false" : "true"));
        stringBuilder.append("\nskipThresholdPerc....: " + m_secoAlgo.getSkipThresholdPercentage());
        stringBuilder.append("\naddFullyCoveredInsts.: " + m_secoAlgo.readdAllCovered);
        stringBuilder.append("\nuseMultilabelHeads...: " + m_secoAlgo.areMultilabelHeadsUsed());
        stringBuilder.append("\nbeamWidth............: " + m_secoAlgo.getBeamWidth());
        stringBuilder.append("\nevaluationStrategy...: " + m_secoAlgo.getEvaluationStrategy());
        stringBuilder.append("\naveragingStrategy....: " + m_secoAlgo.getAveragingStrategy());
        return super.toString().replace("\nSingleHeadRuleSet..............:",
                stringBuilder.toString() + "\nSingleHeadRuleSet..............:");
    }

}