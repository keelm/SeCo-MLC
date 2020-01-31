package de.tu_darmstadt.ke.seco;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Precision;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Recall;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.SubsetAccuracy;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRuleSet;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.Weka379AdapterMultilabel;
import mulan.evaluation.measure.Measure;

public class Results {	
	
	public void printResults(double parameter, Weka379AdapterMultilabel multilabelLearner, List<Measure> eval_results, double coverage, String optimizationMethod) throws IOException {
        
		// Specify csv file name and save path, does not automatically change according to parameters!
		String filename = "emotions.csv";
        FileWriter csvWriter = new FileWriter("defaultsubset/" + filename, true);
        
        // Compute results
        Recall r = new Recall();
        Precision p = new Precision();
        
        double averageOptimizedR = 0;
        double averageEvaluatedR = 0;
        double averageP = 0;
        double currOptimizedR = 0;
        double currEvaluatedR = 0;
        double currP = 0;
        double averageSA = 0;
        double examCov = coverage;
        
        MultiHeadRuleSet theory = (MultiHeadRuleSet) multilabelLearner.getSeCoClassifier().getTheory();
        Iterator<MultiHeadRule> i = theory.iterator();
        
        int numRules = theory.size();
        double cardinality = 0;
        
        while (i.hasNext()) {
        	MultiHeadRule rule = i.next();       
        	currOptimizedR = r.evaluateConfusionMatrix(rule.getRecallStats());
        	currEvaluatedR = r.evaluateConfusionMatrix(rule.getRecallEvalStats());
        	currP = p.evaluateConfusionMatrix(rule.getStats());
        	averageOptimizedR += currOptimizedR;
        	averageEvaluatedR += currEvaluatedR;
        	averageP += currP;
        	cardinality += rule.getCardinality();
        }       
        
        averageOptimizedR = averageOptimizedR / theory.size();
        averageEvaluatedR = averageEvaluatedR / theory.size();
        averageP = averageP / theory.size();
        averageSA = averageSA / theory.size();
        examCov = examCov / theory.size();
        cardinality = cardinality / theory.size();
        
        csvWriter.append("" + parameter);
        csvWriter.append(",");
        csvWriter.append("" + averageOptimizedR);
        csvWriter.append(",");
        csvWriter.append("" + averageP);
        csvWriter.append(",");
        csvWriter.append("" + examCov);
        for (Measure measure : eval_results) {
        	csvWriter.append("," + measure.getValue());
        }
        csvWriter.append(",");
        csvWriter.append("" + numRules);
        csvWriter.append(",");
        csvWriter.append("" + cardinality);
        csvWriter.append(",");
        csvWriter.append("" + averageEvaluatedR);
        csvWriter.append("\n");
        
        csvWriter.flush();
        csvWriter.close();
	}
}
