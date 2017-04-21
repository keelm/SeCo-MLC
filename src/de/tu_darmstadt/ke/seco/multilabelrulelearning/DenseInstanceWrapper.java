package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Utils;

public class DenseInstanceWrapper extends DenseInstance {

	
	
	private Instance wrappedInstance;

	public Instance getWrappedInstance() {
		return wrappedInstance;
	}


	public DenseInstanceWrapper(Instance instance, int[] labelIndices) {
		super(instance); //this does NOT make a deep copy!!!!
		if(instance instanceof DenseInstance)
			m_AttValues=instance.toDoubleArray();
		wrappedInstance=instance;
		//if of type wrapper, than hold the changed label features
		if(!(instance instanceof DenseInstanceWrapper)){
			for (int i = 0; i < labelIndices.length; i++) {
				int labelIndex = labelIndices[i];
				m_AttValues[labelIndex]=Utils.missingValue(); //set missing
			}
		}
	}

	  /**
	   * Produces a true copy of this instance, i.e. from the external view, and return a DenseInstance object. 
	   * 
	   * @return the shallow copy
	   */
	  @Override
	  public/* @pure@ */Object copy() {
	    DenseInstance result = new DenseInstance(this.weight(),this.toDoubleArray());
	    result.setDataset(m_Dataset);
	    return result;
	  }

	  /**
	   * Returns the values of each attribute as an array of doubles. Sets the classValue for the current view, so different results are obtained for different classIndeces set.
	   * 
	   * @return an array containing all the instance attribute values
	   */
	  public double[] toDoubleArray() {
		    double[] newValues = new double[m_AttValues.length];
		    System.arraycopy(m_AttValues, 0, newValues, 0, m_AttValues.length);
		    try {
		    	if(m_Dataset!=null && classIndex()>=0)
		    		newValues[classIndex()]=classValue();	
			} catch (Exception e) {
			}
		    return newValues;
	  }

	
	public double classValue(){
		//do not return missing, but the true value, but only for the target label
		//but take the index of the wrapped instances object
		return wrappedInstance.value(classIndex());
		
	}
	
	  @Override
	  public String toStringNoWeight(int afterDecimalPoint) {
	    StringBuffer text = new StringBuffer();

	    for (int i = 0; i < m_AttValues.length; i++) {
	      if (i > 0)
	        text.append(",");
//	      if(value(i)==wrappedInstance.value(i)|wrappedInstance.value(i)==0)
	      if(value(i)==wrappedInstance.value(i))
	    	  text.append(toString(i, afterDecimalPoint));
	      else{
	    	  text.append(toString(i, afterDecimalPoint)+"|"+wrappedInstance.toString(i, afterDecimalPoint));
	      }
	    }

	    return text.toString();
	  }
	
}
