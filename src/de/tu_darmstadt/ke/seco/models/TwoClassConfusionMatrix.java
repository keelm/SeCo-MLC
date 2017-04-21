/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * TwoClassStats.java Copyright (C) 2002 University of Waikato Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Johannes Fürnkranz
 */

/*
 * Changes by Johannes Fürnkranz: - Added empty Constructor - Added functions for incrementing counts - Added total pos, neg, predicted pos, predicted neg, corr, incorr - Added error rate and accuracy - Added copy constructor
 */

package de.tu_darmstadt.ke.seco.models;

import java.io.Serializable;

/**
 * Encapsulates performance functions for two-class problems.
 *
 * @author Len Trigg (len@reeltwo.com)
 * @author Knowledge Engineering Group
 * @version $Revision: 356 $
 */
public class TwoClassConfusionMatrix implements Serializable {

	private static final long serialVersionUID = -7388399187651374331L;

	/** Pos predicted as pos */
	private double numberOfTruePositives = 0;

	/** Pos predicted as neg */
	private double numberOfFalseNegatives = 0;

	/** Neg predicted as pos */
	private double numberOfFalsePositives = 0;

	/** Neg predicted as neg */
	private double numberOfTrueNegatives = 0;

	/**
	 * Initializes a TwoClassStats with all 0s.
	 */
	public TwoClassConfusionMatrix() {
	}

	/**
	 * Creates the TwoClassStats with the given initial performance values.
	 *
	 * @param numberOfTruePositives
	 *            the number of correctly classified positives
	 * @param numberOfFalsePositives
	 *            the number of incorrectly classified negatives
	 * @param numberOfTrueNegatives
	 *            the number of correctly classified negatives
	 * @param numberOfFalseNegatives
	 *            the number of incorrectly classified positives
	 */
	public TwoClassConfusionMatrix(final double numberOfTruePositives, final double numberOfFalseNegatives, final double numberOfFalsePositives, final double numberOfTrueNegatives) {
		this.numberOfTruePositives = numberOfTruePositives;
		this.numberOfFalseNegatives = numberOfFalseNegatives;
		this.numberOfFalsePositives = numberOfFalsePositives;
		this.numberOfTrueNegatives = numberOfTrueNegatives;
	}

	@Override
	public Object clone() {
		return new TwoClassConfusionMatrix(numberOfTruePositives, numberOfFalseNegatives, numberOfFalsePositives, numberOfTrueNegatives);
	}

	/** Increments the number of true positives */
	public void addTruePositives(final double numberOfTruePositivesToAdd) {
		numberOfTruePositives += numberOfTruePositivesToAdd;
	}

	/** Increments the number of false negatives */
	public void addFalseNegatives(final double numberOfFalseNegativesToAdd) {
		numberOfFalseNegatives += numberOfFalseNegativesToAdd;
	}

	/** Increments the number of false positives */
	public void addFalsePositives(final double numberOfFalsePositivesToAdd) {
		numberOfFalsePositives += numberOfFalsePositivesToAdd;
	}

	/** Increments the number of true negatives */
	public void addTrueNegatives(final double numberOfTrueNegativesToAdd) {
		numberOfTrueNegatives += numberOfTrueNegativesToAdd;
	}

	/** Gets the number of positive instances predicted as positive */
	public double getNumberOfTruePositives() {
		return numberOfTruePositives;
	}

	/** Gets the number of positive instances predicted as negative */
	public double getNumberOfFalseNegatives() {
		return numberOfFalseNegatives;
	}

	/** Gets the number of negative instances predicted as positive */
	public double getNumberOfFalsePositives() {
		return numberOfFalsePositives;
	}

	/** Gets the number of negative instances predicted as negative */
	public double getNumberOfTrueNegatives() {
		return numberOfTrueNegatives;
	}

	/** Gets the number of correctly predicted instances */
	public double getNumberOfCorrectlyClassified() {
		return numberOfTruePositives + numberOfTrueNegatives;
	}

	/** Gets the number of incorrectly predicted instances */
	public double getNumberOfIncorrectClassified() {
		return numberOfFalsePositives + numberOfFalseNegatives;
	}

	/** Gets the total number of positive instances */
	public double getNumberOfPositives() {
		return numberOfTruePositives + numberOfFalseNegatives;
	}

	/** Gets the total number of negative instances */
	public double getNumberOfNegatives() {
		return numberOfTrueNegatives + numberOfFalsePositives;
	}

	/** Gets the total number of examples predicted positive */
	public double getNumberOfPredictedPositive() {
		return numberOfTruePositives + numberOfFalsePositives;
	}

	/** Gets the total number of examples predicted negative */
	public double getNumberOfPredictedNegative() {
		return numberOfTrueNegatives + numberOfFalseNegatives;
	}

	/** Gets the total number of examples */
	public double getNumberOfExamples() {
		return numberOfTruePositives + numberOfFalsePositives + numberOfFalseNegatives + numberOfTrueNegatives;
	}

	/**
	 * Returns a string containing the various performance measures for the current object
	 */
	@Override
	public String toString() {
		return "[[" + getNumberOfTruePositives() + " " + getNumberOfFalsePositives() + "][" + getNumberOfFalseNegatives() + " " + getNumberOfTrueNegatives() + "]]";
	}

	public void setNumberOfTruePositives(final double numberOfTruePositives) {
		this.numberOfTruePositives = numberOfTruePositives;
	}

	public void setNumberOfFalseNegatives(final double numberOfFalseNegatives) {
		this.numberOfFalseNegatives = numberOfFalseNegatives;
	}

	public void setNumberOfFalsePositives(final double numberOfFalsePositives) {
		this.numberOfFalsePositives = numberOfFalsePositives;
	}

	public void setNumberOfTrueNegatives(final double numberOfTrueNegatives) {
		this.numberOfTrueNegatives = numberOfTrueNegatives;
	}
}
