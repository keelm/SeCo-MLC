package de.tu_darmstadt.ke.seco.learners.core;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.*;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.MultilabelSecoClassifier;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import de.tu_darmstadt.ke.seco.utils.Logger;
import de.tu_darmstadt.ke.seco.utils.StopWatch;
import weka.core.Instance;

public class SeCoClassifierFactory {

    public static SeCoClassifier buildSeCoClassifier(final SeCoAlgorithm seCoAlgorithm, final Instances trainingData) throws Exception {
        Logger.info("Building classifier with algorithm:\n" + seCoAlgorithm);

        final StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        final SeCoClassifier seCoClassifier = new SeCoClassifier();
        seCoClassifier.setSeCoAlgorithm(seCoAlgorithm);

        Instances orderedInstances = trainingData.orderClasses();

        final Attribute oldClassOrder = trainingData.classAttribute();
        final Attribute newClassOrder = orderedInstances.classAttribute();

        for (int i = 0; i < oldClassOrder.numValues(); i++) {
            final String oldVal = oldClassOrder.value(i);
            for (int j = 0; j < newClassOrder.numValues(); j++) {
                final String newVal = newClassOrder.value(j);
                if (newVal.equals(oldVal))
                    seCoClassifier.orderedClassMapping.put(new Double(j), new Double(i));
            }
        }

        setDefaultRule(((SingleHeadRuleSet) seCoClassifier.getTheory()), seCoClassifier.getHeuristic(), orderedInstances, trainingData.classAttribute());
        Logger.debug("buildClassifier(), class loop");

        // do some other init stuff
        // for optimization of refiner
        // TODO by m.zopf: if this should always be done, it should be placed in SeCoFactory
        seCoAlgorithm.getRuleRefiner().setProperty("beamwidth", seCoAlgorithm.getRuleFilter().getProperty("beamwidth"));

		/*
         * This has to run through all class values beginning with the class with the smallest number of examples onto (not included) the class with the most examples
		 */
        // iterate over distinct (all) class values
        // see Instances.distinctClassValues for more info on the arguments
        // for (final double classValue : orderedInstances.getDistinctClassValues(true, true)) {
        // // set the current class value
        // // setClassVal(classValue);
        // // build classifier for one class
        // final SingleHeadRuleSet myTheory = seCoClassifier.getSeCoAlgorithm().separateAndConquer(orderedInstances, classValue);
        // seCoClassifier.getTheory().getRules().addAll(myTheory.getRules());
        //
        // orderedInstances = seCoClassifier.getSeCoAlgorithm().getNewInstances();
        // }

        // in multilable prediction, predict always 'label is set' and not 'label is not set'
        final SingleHeadRuleSet myTheory = seCoClassifier.getSeCoAlgorithm().separateAndConquer(orderedInstances, 0.0);

        ((SingleHeadRuleSet) seCoClassifier.getTheory()).getRules().addAll(myTheory.getRules());

        orderedInstances = seCoClassifier.getSeCoAlgorithm().getNewInstances();

        // set the DefaultRule with the current stats
        setDefaultRule(((SingleHeadRuleSet) seCoClassifier.getTheory()), seCoClassifier.getHeuristic(), orderedInstances, trainingData.classAttribute());
        seCoAlgorithm.getRuleStoppingCriterion().setProperty("reset", "true");
        Logger.info("Complete theory:\n" + seCoClassifier.getTheory().toString());

        stopWatch.stop();
        seCoClassifier.setLearningTime(stopWatch.getElapsedTime());
        Logger.info("Building classifier finished. Time used: " + stopWatch.getElapsedTime() + "ms.");
        return seCoClassifier;
    }

    /**
     * This will create the default rule that will assign the most frequent class to any example. The default rule will be put into the theory. The examples must be ordered in ascending frequency of their classes before applying this method.
     *
     * @param theory   The rule set that will get the default rule.
     * @param examples The training set.
     * @param cl       original class attribute.
     */
    // TODO by m.zopf: setting the default rule to the most frequent class before doing anything is maybe not the best choice. may it be smarter to set the default rule to the most frequent unclassified class after learning?
    private static void setDefaultRule(final SingleHeadRuleSet theory, Heuristic heuristic, final Instances examples, final Attribute cl) throws Exception {
        // make the default rule with the last class

        final Instances data = examples;
        // determine the index of the most frequent class
        double mostFreqClassIndex = 0.0;
        for (double i = 1.0; i < data.numClasses(); i++)
            if (data.countInstances(i) > data.countInstances(mostFreqClassIndex))
                mostFreqClassIndex = i;

        final int defClass = (int) mostFreqClassIndex;
        final NominalCondition defHead = new NominalCondition(data.classAttribute(), defClass);
        final SingleHeadRule def = new SingleHeadRule(heuristic, defHead);
        def.evaluateRule(examples, mostFreqClassIndex, heuristic);
        theory.setDefaultRule(def);

        // compute stats:
        final TwoClassConfusionMatrix defStats = def.getStats();

        double totalcorrect = 0;
        double totalincorrect = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            final Instance inst = data.instance(i);
            if ((int) inst.classValue() == defClass)
                totalcorrect += inst.weight();
            else
                totalincorrect += inst.weight();
            // missing values are never covered
        }
        defStats.setNumberOfTruePositives(totalcorrect);
        defStats.setNumberOfFalsePositives(totalincorrect);

        Logger.debug("defaultRule: " + def);
    }

    // multilable:
    public static SeCoClassifier buildSeCoClassifierUnordered(final SeCoAlgorithm seCoAlgorithm, Instances instances) throws Exception {
        Logger.info("Building classifier with algorithm:\n" + seCoAlgorithm);

        final StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        final SeCoClassifier seCoClassifier = new SeCoClassifier();
        seCoClassifier.setSeCoAlgorithm(seCoAlgorithm);

        Instances orderedInstances = instances.orderClasses();

        final Attribute oldClassOrder = instances.classAttribute();
        final Attribute newClassOrder = orderedInstances.classAttribute();

        for (int i = 0; i < oldClassOrder.numValues(); i++) {
            final String oldVal = oldClassOrder.value(i);
            for (int j = 0; j < newClassOrder.numValues(); j++) {
                final String newVal = newClassOrder.value(j);
                if (newVal.equals(oldVal))
                    seCoClassifier.orderedClassMapping.put(new Double(j), new Double(i));
            }
        }

        // set default rule to ?
        final Instances data = orderedInstances;
        // determine the index of the most frequent class
        final NominalCondition defHead = new NominalCondition(data.classAttribute(), Double.NaN);
        final SingleHeadRule def = new SingleHeadRule(seCoClassifier.getHeuristic(), defHead);
        ((SingleHeadRuleSet) seCoClassifier.getTheory()).setDefaultRule(def);

        Logger.debug("defaultRule: " + def);
        Logger.debug("buildClassifier(), class loop");

        // do some other init stuff
        // for optimization of refiner
        // TODO by m.zopf: if this should always be done, it should be placed in SeCoFactory
        seCoAlgorithm.getRuleRefiner().setProperty("beamwidth", seCoAlgorithm.getRuleFilter().getProperty("beamwidth"));

		/*
         * This has to run through all class values beginning with the class with the smallest number of examples onto (not included) the class with the most examples
		 */
        // iterate over distinct (all) class values
        // see Instances.distinctClassValues for more info on the arguments
        // for (final double classValue : orderedInstances.getDistinctClassValues(true, true)) {
        // // set the current class value
        // // setClassVal(classValue);
        // // build classifier for one class
        // final SingleHeadRuleSet myTheory = seCoClassifier.getSeCoAlgorithm().separateAndConquer(orderedInstances, classValue);
        // seCoClassifier.getTheory().getRules().addAll(myTheory.getRules());
        //
        // orderedInstances = seCoClassifier.getSeCoAlgorithm().getNewInstances();
        // }

        // in multilable prediction, predict always 'label is set' and not 'label is not set'
        SingleHeadRuleSet learnedTheory = seCoClassifier.getSeCoAlgorithm().separateAndConquerUnordered(orderedInstances);

        ((SingleHeadRuleSet) seCoClassifier.getTheory()).getRules().addAll(learnedTheory.getRules());

        orderedInstances = seCoClassifier.getSeCoAlgorithm().getNewInstances();

        // set the DefaultRule with the current stats
        // setDefaultRule(seCoClassifier.getTheory(), seCoClassifier.getHeuristic(), orderedInstances, instances.classAttribute());
        seCoAlgorithm.getRuleStoppingCriterion().setProperty("reset", "true");
        Logger.info("Complete theory:\n" + seCoClassifier.getTheory().toString());

        stopWatch.stop();
        seCoClassifier.setLearningTime(stopWatch.getElapsedTime());
        Logger.info("Building classifier finished. Time used: " + stopWatch.getElapsedTime() + "ms.");
        return seCoClassifier;
    }

    // multilabel:
    public static SeCoClassifier buildSeCoClassifierMultiClass(final SeCoAlgorithm seCoAlgorithm, Instances instances) throws Exception {
        Logger.info("Building classifier with algorithm:\n" + seCoAlgorithm);

        final StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        final SeCoClassifier seCoClassifier = new SeCoClassifier();
        seCoClassifier.setSeCoAlgorithm(seCoAlgorithm);

        Instances orderedInstances = instances.orderClasses();

        final Attribute oldClassOrder = instances.classAttribute();
        final Attribute newClassOrder = orderedInstances.classAttribute();

        for (int i = 0; i < oldClassOrder.numValues(); i++) {
            final String oldVal = oldClassOrder.value(i);
            for (int j = 0; j < newClassOrder.numValues(); j++) {
                final String newVal = newClassOrder.value(j);
                if (newVal.equals(oldVal))
                    seCoClassifier.orderedClassMapping.put(new Double(j), new Double(i));
            }
        }

        // set default rule to ?
        final Instances data = orderedInstances;
        // determine the index of the most frequent class
        final NominalCondition defHead = new NominalCondition(data.classAttribute(), Double.NaN);
        final SingleHeadRule def = new SingleHeadRule(seCoClassifier.getHeuristic(), defHead);
        ((SingleHeadRuleSet) seCoClassifier.getTheory()).setDefaultRule(def);

        Logger.debug("defaultRule: " + def);
        Logger.debug("buildClassifier(), class loop");

        // do some other init stuff
        // for optimization of refiner
        // TODO by m.zopf: if this should always be done, it should be placed in SeCoFactory
        seCoAlgorithm.getRuleRefiner().setProperty("beamwidth", seCoAlgorithm.getRuleFilter().getProperty("beamwidth"));

			/*
             * This has to run through all class values beginning with the class with the smallest number of examples onto (not included) the class with the most examples
			 */
        // iterate over distinct (all) class values
        // see Instances.distinctClassValues for more info on the arguments
        // for (final double classValue : orderedInstances.getDistinctClassValues(true, true)) {
        // // set the current class value
        // // setClassVal(classValue);
        // // build classifier for one class
        // final SingleHeadRuleSet myTheory = seCoClassifier.getSeCoAlgorithm().separateAndConquer(orderedInstances, classValue);
        // seCoClassifier.getTheory().getRules().addAll(myTheory.getRules());
        //
        // orderedInstances = seCoClassifier.getSeCoAlgorithm().getNewInstances();
        // }

        // in multilabel prediction, predict always 'label is set' and not 'label is not set'
        SingleHeadRuleSet learnedTheory = seCoClassifier.getSeCoAlgorithm().separateAndConquerMultiClass(orderedInstances);

        ((SingleHeadRuleSet) seCoClassifier.getTheory()).getRules().addAll(learnedTheory.getRules());

        orderedInstances = seCoClassifier.getSeCoAlgorithm().getNewInstances();

        // set the DefaultRule with the current stats
        // setDefaultRule(seCoClassifier.getTheory(), seCoClassifier.getHeuristic(), orderedInstances, instances.classAttribute());
        seCoAlgorithm.getRuleStoppingCriterion().setProperty("reset", "true");
        Logger.info("Complete theory:\n" + seCoClassifier.getTheory().toString());

        stopWatch.stop();
        seCoClassifier.setLearningTime(stopWatch.getElapsedTime());
        Logger.info("Building classifier finished. Time used: " + stopWatch.getElapsedTime() + "ms.");
        return seCoClassifier;
    }

    // multilabel:
    public static MultilabelSecoClassifier buildSeCoClassifierMultilabel(final SeCoAlgorithm seCoAlgorithm, Instances instances, int[] labelIndices, String classifyMethod) throws Exception {
        Logger.info("Building classifier with algorithm:\n" + seCoAlgorithm);

        final StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        final MultilabelSecoClassifier seCoClassifier = new MultilabelSecoClassifier(labelIndices, seCoAlgorithm);
        seCoClassifier.setSeCoAlgorithm(seCoAlgorithm);

        // TODO by m.zopf: if this should always be done, it should be placed in SeCoFactory
        seCoAlgorithm.getRuleRefiner().setProperty("beamwidth", seCoAlgorithm.getRuleFilter().getProperty("beamwidth"));

        seCoClassifier.m_theory = seCoClassifier.getSeCoAlgorithm().separateAndConquerMultilabel(instances, labelIndices);

        seCoClassifier.setClassifyMethod(classifyMethod);
        
        seCoAlgorithm.getRuleStoppingCriterion().setProperty("reset", "true");
        Logger.info("Complete theory:\n" + seCoClassifier.getTheory().toString());

        stopWatch.stop();
        seCoClassifier.setLearningTime(stopWatch.getElapsedTime());
        Logger.info("Building classifier finished. Time used: " + stopWatch.getElapsedTime() + "ms.");
        return seCoClassifier;
    }
}
