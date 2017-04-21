package de.tu_darmstadt.ke.seco.models;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.core.UnassignedDatasetException;
import weka.filters.Filter;
import weka.filters.supervised.attribute.ClassOrder;
import de.tu_darmstadt.ke.seco.utils.Logger;

public class InstancesUtils {

	public static int countInstances(Instances instances, final double classValue) throws UnassignedClassException, UnassignedDatasetException {
		int count = 0;

		final Enumeration<Instance> en = instances.enumerateInstances();
		while (en.hasMoreElements()) {
			final Instance inst = en.nextElement();

			if (inst.classValue() == classValue) {
				count++;
			}
		}

		return count;
	}

	public static Instances orderClasses(Instances instances) throws Exception {
		ClassOrder classOrder = new ClassOrder();
		classOrder.setInputFormat(instances);
		return Filter.useFilter(instances, classOrder);
	}

	public static void setWeightsTo0(Instances instances) {
		for (final Instance instance : instances) {
			instance.setWeight(0);
		}
	}

	public static Collection<Double> getDistinctClassValues(Instances instances, final boolean ascending, final boolean skipLast) throws UnassignedClassException, UnassignedDatasetException {

		final Set<Double> valuesSet = new HashSet<Double>();

		final Enumeration<Instance> en = instances.enumerateInstances();
		while (en.hasMoreElements()) {
			final Instance inst = en.nextElement();
			valuesSet.add(inst.classValue());
		}
		TreeSet<Double> orderedSet;

		orderedSet = new TreeSet<Double>();
		orderedSet.addAll(valuesSet);

		// remove the highest value
		if (skipLast) {
			orderedSet.pollLast();
		}

		Logger.debug("Created distinct class values:");
		if (ascending) {
			Logger.debug(orderedSet.toString());
		}
		else {
			Logger.debug(orderedSet.descendingSet().toString());
		}

		if (ascending) {
			return orderedSet;
		}
		else {
			return orderedSet.descendingSet();
		}
	}

	public static boolean containsPositive(Instances instances, final double classValue) {
		// final java.util.Enumeration<Instance> en = examples.enumerateInstances();

		for (final weka.core.Instance instance : instances) {
			try {
				if ((instance.classValue() == classValue)) {
					return true;
				}
			}
			catch (final UnassignedClassException ex) {
				System.err.println(ex.getMessage());
				ex.printStackTrace();
				System.exit(-1);
			}
			catch (final UnassignedDatasetException ex) {
				System.err.println(ex.getMessage());
				ex.printStackTrace();
				System.exit(-1);
			}
		}
		Logger.debug("containsPositive() returns false");
		return false;
	}

	public static ArrayList<Instances> splitInstancesStratified(Instances instances, double splitpoint) {
		return splitInstancesStratified(instances, splitpoint, new Random(0));
	}

	public static ArrayList<Instances> splitInstancesStratified(Instances instances, double splitpoint, Random random) {
		ArrayList<Instances> split = new ArrayList<>();
		split.add(new Instances(instances, 0));
		split.add(new Instances(instances, 0));

		HashMap<Double, ArrayList<Instance>> instancesByClasses = new HashMap<>(instances.numClasses());

		for (Instance instance : instances) {
			if (!instancesByClasses.containsKey(instance.classValue())) {
				instancesByClasses.put(instance.classValue(), new ArrayList<Instance>());
			}

			instancesByClasses.get(instance.classValue()).add(instance);
		}

		for (ArrayList<Instance> instancesByClass : instancesByClasses.values()) {
			long numberOfInstancesToPutInFirstSlice = Math.round(instancesByClass.size() * splitpoint);

			while (numberOfInstancesToPutInFirstSlice > 0) {
				split.get(0).add(instancesByClass.remove(random.nextInt(instancesByClass.size())));
				numberOfInstancesToPutInFirstSlice--;
			}

			split.get(1).addAll(instancesByClass);
		}

		return split;
	}
}
