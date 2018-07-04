package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.rules.JRip;

public class BR {

    public static void main(String[] args) throws Exception {

        String trainFile = "data/flags-train.arff";
        String testFile = "data/flags-test.arff";
        String xmlFile = "data/flags.xml";

        MultiLabelInstances trainInstances = new MultiLabelInstances(trainFile, xmlFile);
        MultiLabelInstances testInstances = new MultiLabelInstances(testFile, xmlFile);

        JRip baseLearner = new JRip();
        MultiLabelLearner multiLabelLearner = new BinaryRelevance(baseLearner);

        multiLabelLearner.build(trainInstances);
        System.out.println(multiLabelLearner);

        Evaluator evaluator = new Evaluator();
        Evaluation results = evaluator.evaluate(multiLabelLearner, testInstances, trainInstances);
        System.out.println(results);

    }

}
