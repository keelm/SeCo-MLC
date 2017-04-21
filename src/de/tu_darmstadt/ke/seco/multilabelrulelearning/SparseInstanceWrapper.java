package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import weka.core.Instance;
import weka.core.SparseInstance;
import weka.core.Utils;

public class SparseInstanceWrapper extends SparseInstance {

	
	
	private Instance wrappedInstance;
	
	public Instance getWrappedInstance() {
		return wrappedInstance;
	}


	public SparseInstanceWrapper(Instance instance, int[] labelIndices) {
		super(instance); //doesnt copy, get sure
	    if(instance instanceof SparseInstance){
		int vals=m_AttValues.length;  
			double[] tempValues = new double[vals];
			int[] tempIndices = new int[vals];
	      System.arraycopy(m_AttValues, 0, tempValues, 0, vals);
	      System.arraycopy(m_Indices, 0, tempIndices, 0, vals);
	      m_AttValues=tempValues;
	      m_Indices=tempIndices;
	    }
		wrappedInstance=instance;
		this.m_Dataset=instance.dataset();
		
		double[] tempatts = this.toDoubleArray();
		//if of type wrapper, than hold the changed label features
		if(!(instance instanceof SparseInstanceWrapper)){
			for (int i = 0; i < labelIndices.length; i++) {
				int labelIndex = labelIndices[i];
				tempatts[labelIndex]=Utils.missingValue(); //set missing
				//very very innefficient
	//			setMissing(labelIndex);
			}
		}
		SparseInstanceWrapper tempSparseInstance = new SparseInstanceWrapper(1.0,tempatts);
		this.m_AttValues=tempSparseInstance.m_AttValues;
		this.m_Indices=tempSparseInstance.m_Indices;
		this.m_NumAttributes=tempSparseInstance.m_NumAttributes;
		
		
		// TODO Auto-generated constructor stub
	}

	  /**
	   * Produces a true copy of this instance, i.e. from the external view, and return a SparseInstance object. 
	   * 
	   * @return the shallow copy
	   */
	  @Override
	  public/* @pure@ */Object copy() {
	    SparseInstance result = new SparseInstance(this.weight(),this.toDoubleArray());
	    result.setDataset(m_Dataset);
	    return result;
	  }

	  /**
	   * Returns the values of each attribute as an array of doubles. Sets the classValue for the current view, so different results are obtained for different classIndeces set.
	   * 
	   * @return an array containing all the instance attribute values
	   */
	  public double[] toDoubleArray() {
		    double[] newValues = super.toDoubleArray();
		    try {
		    	if(m_Dataset!=null && classIndex()>=0)
		    		newValues[classIndex()]=classValue();	
			} catch (Exception e) {
			}
		    return newValues;
	  }


	
	private SparseInstanceWrapper(double d, double[] tempatts) {
		super(d,tempatts);
	}


	public double classValue(){
		//do not return missing, but the true value, but only for the target label
		return wrappedInstance.value(classIndex());
		
	}
	
}
