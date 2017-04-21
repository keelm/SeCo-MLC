package de.tu_darmstadt.ke.seco.algorithm;

import java.awt.Component;
import java.awt.EventQueue;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashSet;
import java.util.Set;

import javax.swing.Box;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JTextField;
import javax.swing.ListCellRenderer;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.reflections.Reflections;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;

public class SeCoAlgorithmBuilderGUI extends JFrame {
    private static final long serialVersionUID = -3189629157357944069L;
    private JComboBox<Class<? extends SeCoComponent>> comboBoxCandidateSelector = new JComboBox<>();
    private JComboBox<Class<? extends SeCoComponent>> comboBoxHeuristic = new JComboBox<>();
    private JComboBox<Class<? extends SeCoComponent>> comboBoxPostProcessor = new JComboBox<>();
    private JComboBox<Class<? extends SeCoComponent>> comboBoxRuleFilter = new JComboBox<>();
    private JComboBox<Class<? extends SeCoComponent>> comboBoxRuleInitializer = new JComboBox<>();
    private JComboBox<Class<? extends SeCoComponent>> comboBoxRuleRefiner = new JComboBox<>();
    private JComboBox<Class<? extends SeCoComponent>> comboBoxRuleStoppingCriterion = new JComboBox<>();
    private JComboBox<Class<? extends SeCoComponent>> comboBoxStoppingCriterion = new JComboBox<>();

    private Box horizontalBoxCandidateSelector = Box.createHorizontalBox();
    private Box horizontalBoxHeuristic = Box.createHorizontalBox();
    private Box horizontalBoxPostProcessor = Box.createHorizontalBox();
    private Box horizontalBoxRuleFilter = Box.createHorizontalBox();
    private Box horizontalBoxRuleInitializer = Box.createHorizontalBox();
    private Box horizontalBoxRuleRefiner = Box.createHorizontalBox();
    private Box horizontalBoxRuleStoppingCriterion = Box.createHorizontalBox();
    private Box horizontalBoxStoppingCriterion = Box.createHorizontalBox();

    /**
     * Launch the application.
     */
    public static void main(String[] args) {
        EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                try {
                    SeCoAlgorithmBuilderGUI window = new SeCoAlgorithmBuilderGUI();
                    window.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    class SeCoComponentClassRenderer implements ListCellRenderer<Class<? extends SeCoComponent>> {
        @Override
        public Component getListCellRendererComponent(JList<? extends Class<? extends SeCoComponent>> list, Class<? extends SeCoComponent> value, int index, boolean isSelected, boolean cellHasFocus) {
            final JLabel label = new JLabel();

            if (value != null) {
                Class<? extends SeCoComponent> item = value;
                label.setText(item.getSimpleName());
            }

            return label;
        }
    }

    /**
     * Create the application.
     */
    public SeCoAlgorithmBuilderGUI() {
        initialize();
    }

    /**
     * Initialize the contents of the
     */
    private void initialize() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException e) {
            e.printStackTrace();
        }

        setBounds(100, 100, 813, 402);
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);

        JMenu mnFile = new JMenu("File");
        menuBar.add(mnFile);

        JMenuItem mntmNew = new JMenuItem("New");
        mnFile.add(mntmNew);

        JMenuItem mntmLoad = new JMenuItem("Load...");
        mnFile.add(mntmLoad);

        JMenuItem mntmSaveAs = new JMenuItem("SaveAs...");
        mnFile.add(mntmSaveAs);
        getContentPane().setLayout(null);

        JLabel lblCandidateSelector = new JLabel("CandidateSelector:");
        lblCandidateSelector.setBounds(10, 71, 110, 14);
        getContentPane().add(lblCandidateSelector);

        JLabel lblHeuristic = new JLabel("Heuristic:");
        lblHeuristic.setBounds(10, 96, 110, 14);
        getContentPane().add(lblHeuristic);

        JLabel lblPostProcessor = new JLabel("PostProcessor:");
        lblPostProcessor.setBounds(10, 121, 110, 14);
        getContentPane().add(lblPostProcessor);

        JLabel lblRuleFilter = new JLabel("RuleFilter:");
        lblRuleFilter.setBounds(10, 146, 110, 14);
        getContentPane().add(lblRuleFilter);

        JLabel lblRuleInitializer = new JLabel("RuleInitializer:");
        lblRuleInitializer.setBounds(10, 171, 110, 14);
        getContentPane().add(lblRuleInitializer);

        JLabel lblRulerefiner = new JLabel("RuleRefiner:");
        lblRulerefiner.setBounds(10, 196, 110, 14);
        getContentPane().add(lblRulerefiner);

        JLabel lblRulestoppingcriterion = new JLabel("RuleStoppingCriterion:");
        lblRulestoppingcriterion.setBounds(10, 221, 110, 14);
        getContentPane().add(lblRulestoppingcriterion);

        JLabel lblStoppingcriterion = new JLabel("StoppingCriterion:");
        lblStoppingcriterion.setBounds(10, 246, 110, 14);
        getContentPane().add(lblStoppingcriterion);

        JLabel lblSeco = new JLabel("SeCo:");
        lblSeco.setBounds(10, 11, 110, 14);
        getContentPane().add(lblSeco);

        comboBoxCandidateSelector.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED)
                    loadConfigurableProperties(comboBoxCandidateSelector, horizontalBoxCandidateSelector);
            }
        });
        comboBoxCandidateSelector.setBounds(130, 68, 200, 20);
        comboBoxCandidateSelector.setRenderer(new SeCoComponentClassRenderer());
        getContentPane().add(comboBoxCandidateSelector);

        comboBoxHeuristic.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED)
                    loadConfigurableProperties(comboBoxHeuristic, horizontalBoxHeuristic);
            }
        });
        comboBoxHeuristic.setBounds(130, 93, 200, 20);
        comboBoxHeuristic.setRenderer(new SeCoComponentClassRenderer());
        getContentPane().add(comboBoxHeuristic);

        comboBoxPostProcessor.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED)
                    loadConfigurableProperties(comboBoxPostProcessor, horizontalBoxPostProcessor);
            }
        });
        comboBoxPostProcessor.setBounds(130, 118, 200, 20);
        comboBoxPostProcessor.setRenderer(new SeCoComponentClassRenderer());
        getContentPane().add(comboBoxPostProcessor);

        comboBoxRuleFilter.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED)
                    loadConfigurableProperties(comboBoxRuleFilter, horizontalBoxRuleFilter);
            }
        });
        comboBoxRuleFilter.setBounds(130, 143, 200, 20);
        comboBoxRuleFilter.setRenderer(new SeCoComponentClassRenderer());
        getContentPane().add(comboBoxRuleFilter);

        comboBoxRuleInitializer.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED)
                    loadConfigurableProperties(comboBoxRuleInitializer, horizontalBoxRuleInitializer);
            }
        });
        comboBoxRuleInitializer.setBounds(130, 168, 200, 20);
        comboBoxRuleInitializer.setRenderer(new SeCoComponentClassRenderer());
        getContentPane().add(comboBoxRuleInitializer);

        comboBoxRuleRefiner.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED)
                    loadConfigurableProperties(comboBoxRuleRefiner, horizontalBoxRuleRefiner);
            }
        });
        comboBoxRuleRefiner.setBounds(130, 193, 200, 20);
        comboBoxRuleRefiner.setRenderer(new SeCoComponentClassRenderer());
        getContentPane().add(comboBoxRuleRefiner);

        comboBoxRuleStoppingCriterion.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED)
                    loadConfigurableProperties(comboBoxRuleStoppingCriterion, horizontalBoxRuleStoppingCriterion);
            }
        });
        comboBoxRuleStoppingCriterion.setBounds(130, 218, 200, 20);
        comboBoxRuleStoppingCriterion.setRenderer(new SeCoComponentClassRenderer());
        getContentPane().add(comboBoxRuleStoppingCriterion);

        comboBoxStoppingCriterion.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                if (e.getStateChange() == ItemEvent.SELECTED)
                    loadConfigurableProperties(comboBoxStoppingCriterion, horizontalBoxStoppingCriterion);
            }
        });
        comboBoxStoppingCriterion.setBounds(130, 243, 200, 20);
        comboBoxStoppingCriterion.setRenderer(new SeCoComponentClassRenderer());
        getContentPane().add(comboBoxStoppingCriterion);

        horizontalBoxCandidateSelector.setBounds(340, 68, 407, 20);
        getContentPane().add(horizontalBoxCandidateSelector);

        horizontalBoxHeuristic.setBounds(340, 93, 407, 20);
        getContentPane().add(horizontalBoxHeuristic);

        horizontalBoxPostProcessor.setBounds(340, 118, 407, 20);
        getContentPane().add(horizontalBoxPostProcessor);

        horizontalBoxRuleFilter.setBounds(340, 143, 407, 20);
        getContentPane().add(horizontalBoxRuleFilter);

        horizontalBoxRuleInitializer.setBounds(340, 168, 407, 20);
        getContentPane().add(horizontalBoxRuleInitializer);

        horizontalBoxRuleRefiner.setBounds(340, 193, 407, 20);
        getContentPane().add(horizontalBoxRuleRefiner);

        horizontalBoxRuleStoppingCriterion.setBounds(340, 218, 407, 20);
        getContentPane().add(horizontalBoxRuleStoppingCriterion);

        horizontalBoxStoppingCriterion.setBounds(340, 243, 407, 20);
        getContentPane().add(horizontalBoxStoppingCriterion);

        loadComboboxContents();
    }

    @SuppressWarnings("unchecked")
    private void loadComboboxContents() { // TODO: select algorithm default values as default in combo box
        comboBoxCandidateSelector.setModel(new DefaultComboBoxModel<Class<? extends SeCoComponent>>(getAllComponentsInPackage(SeCoAlgorithmFactory.CANDIDATE_SELECTORS_PATH).toArray(new Class[0])));
        comboBoxHeuristic.setModel(new DefaultComboBoxModel<Class<? extends SeCoComponent>>(getAllComponentsInPackage(SeCoAlgorithmFactory.HEURISTICS_PATH).toArray(new Class[0])));
        comboBoxPostProcessor.setModel(new DefaultComboBoxModel<Class<? extends SeCoComponent>>(getAllComponentsInPackage(SeCoAlgorithmFactory.POST_PROCESSORS_PATH).toArray(new Class[0])));
        comboBoxRuleFilter.setModel(new DefaultComboBoxModel<Class<? extends SeCoComponent>>(getAllComponentsInPackage(SeCoAlgorithmFactory.RULE_FILTERS_PATH).toArray(new Class[0])));
        comboBoxRuleInitializer.setModel(new DefaultComboBoxModel<Class<? extends SeCoComponent>>(getAllComponentsInPackage(SeCoAlgorithmFactory.RULE_INITIALIZERS_PATH).toArray(new Class[0])));
        comboBoxRuleRefiner.setModel(new DefaultComboBoxModel<Class<? extends SeCoComponent>>(getAllComponentsInPackage(SeCoAlgorithmFactory.RULE_REFINERS_PATH).toArray(new Class[0])));
        comboBoxRuleStoppingCriterion.setModel(new DefaultComboBoxModel<Class<? extends SeCoComponent>>(getAllComponentsInPackage(SeCoAlgorithmFactory.RULE_STOPPING_CRITERIONS_PATH).toArray(new Class[0])));
        comboBoxStoppingCriterion.setModel(new DefaultComboBoxModel<Class<? extends SeCoComponent>>(getAllComponentsInPackage(SeCoAlgorithmFactory.STOPPING_CRITERIONS_PATH).toArray(new Class[0])));
    }

    private Set<Class<? extends SeCoComponent>> getAllComponentsInPackage(String packageToSearchIn) { // TODO: sort return value alphabetically
        Reflections reflections = new Reflections(packageToSearchIn); // TODO: should be implemented without using additional library
        Set<Class<? extends SeCoComponent>> seCoComponentClasses = reflections.getSubTypesOf(SeCoComponent.class);

        Set<Class<? extends SeCoComponent>> seCoComponentClassesWithoutAbstractClasses = new HashSet<>();
        for (Class<? extends SeCoComponent> seCoComponentClass : seCoComponentClasses)
            if (!Modifier.isAbstract(seCoComponentClass.getModifiers()))
                seCoComponentClassesWithoutAbstractClasses.add(seCoComponentClass);

        return seCoComponentClassesWithoutAbstractClasses;
    }

    private void loadConfigurableProperties(JComboBox<Class<? extends SeCoComponent>> comboBox, Box box) {
        box.removeAll();
        @SuppressWarnings("unchecked")
        Class<? extends SeCoComponent> selectedComponentClass = (Class<? extends SeCoComponent>) comboBox.getSelectedItem();

        for (Field field : selectedComponentClass.getDeclaredFields())
            if (field.isAnnotationPresent(ConfigurableProperty.class)) {
                JLabel lbl = new JLabel(field.getName() + ": ");
                box.add(lbl);

                JTextField textField = new JTextField();
                textField.setBounds(94, 24, 86, 20);
                box.add(textField);

            }

        box.validate();
        box.repaint();
    }
}
