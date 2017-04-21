package de.tu_darmstadt.ke.seco.algorithm;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JTextField;

public class PropertyBox extends Box {

	private static final long serialVersionUID = -7246793875335910067L;

	private String propertyName;

	private JTextField textField;

	public PropertyBox(String propertyName) {
		super(BoxLayout.X_AXIS);
		this.propertyName = propertyName;
		JLabel lbl = new JLabel(propertyName + ": ");
		add(lbl);

		textField = new JTextField();
		add(textField);
	}

	public String getPropertyValue() {
		return textField.getText();
	}

	public String getPropertyName() {
		return propertyName;
	}
}
