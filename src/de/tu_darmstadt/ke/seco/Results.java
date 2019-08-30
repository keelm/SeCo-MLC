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

	
	
	public void printResults(double parameter, Weka379AdapterMultilabel multilabelLearner, List<Measure> eval_results) throws IOException {
		// get theory and produce statistics
        String filename = "remain_weather_measures.csv";
        //FileWriter csvWriter = new FileWriter("C:\\Users\\Pascal\\Documents\\Studium\\BachelorOfScienceInformatik\\Bachelorarbeit\\Ergebnisse\\Weather\\" + filename, true);
        FileWriter csvWriter = new FileWriter("remain/weather/" + filename, true);
        
        Recall r = new Recall();
        Precision p = new Precision();
        //SubsetAccuracy sa = new SubsetAccuracy();
        //Covered Labels
        //Covered Examples
        
        double averageR = 0;
        double averageP = 0;
        double currR = 0;
        double currP = 0;
        double averageSA = 0;
        double currSA = 0;
        double examCov = 0;
        MultiHeadRuleSet theory = (MultiHeadRuleSet) multilabelLearner.getSeCoClassifier().getTheory();
        Iterator<MultiHeadRule> i = theory.iterator();
        
        while (i.hasNext()) {
        	MultiHeadRule rule = i.next();        	
        	currR = r.evaluateConfusionMatrix(rule.getStats());
        	currP = p.evaluateConfusionMatrix(rule.getStats());
        	averageR += currR;
        	averageP += currP;
        	
        	//currSA = sa.evaluateConfusionMatrix(rule.getStats());
        	averageSA += currSA;
        	
        	// example coverage =  for current example-set
        	examCov += rule.getStats().getNumberOfPredictedPositive();
        }
        averageR = averageR / theory.size();
        averageP = averageP / theory.size();
        averageSA = averageSA / theory.size();
        examCov = examCov / theory.size();
        
        csvWriter.append("" + parameter);
        csvWriter.append(",");
        csvWriter.append("" + averageR);
        csvWriter.append(",");
        csvWriter.append("" + averageP);
        //csvWriter.append(",");
        //csvWriter.append("" + averageSA);
        //csvWriter.append(",");
        //csvWriter.append("" + examCov);
        for (Measure measure : eval_results) {
        	csvWriter.append("," + measure.toString());
        }
        csvWriter.append("\n");
        
        csvWriter.flush();
        csvWriter.close();
        ////
	}
}
