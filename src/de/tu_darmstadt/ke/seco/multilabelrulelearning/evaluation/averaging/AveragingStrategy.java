package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Condition;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.DenseInstanceWrapper;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.SparseInstanceWrapper;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import weka.core.Instance;
import weka.core.Utils;

import java.util.Collection;
import java.util.Iterator;

public abstract class AveragingStrategy {

    public static final String MICRO_AVERAGING = "micro-averaging";

    public static final String LABEL_BASED_AVERAGING = "label-based-averaging";

    public static final String EXAMPLE_BASED_AVERAGING = "example-based-averaging";

    public static final String MACRO_AVERAGING = "macro-averaging";
    
    private String optimizationHeuristic = "adapted";
    
    private String evaluationHeuristic = "clever";
    
    public void setHeuristics(String optimizationHeuristic, String evaluationHeuristic) {
    	this.optimizationHeuristic = optimizationHeuristic;
    	this.evaluationHeuristic = evaluationHeuristic;
    }

    private double getLabelValue(final Instance instance, final int labelIndex) {
        Instance wrappedInstance =
                instance instanceof DenseInstanceWrapper ? ((DenseInstanceWrapper) instance).getWrappedInstance() :
                        ((SparseInstanceWrapper) instance).getWrappedInstance();
        return wrappedInstance.value(labelIndex);
    }

    final Iterable<Integer> instancesIterable(final Instances instances) {
        return () -> new Iterator<Integer>() {

            private int i = 0;

            @Override
            public boolean hasNext() {
                return i < instances.numInstances();
            }

            @Override
            public Integer next() {
                return i++;
            }

        };
    }

    final boolean areAllLabelsAlreadyPredicted(final Instance instance, final Head head) {
        for (Condition labelAttribute : head) {
            if (Utils.isMissingValue(instance.value(labelAttribute.getAttr().index()))) {
                return false;
            }
        }

        return true;
    }

    final void aggregate(final boolean covers, final Head head, final Instance instance, final int labelIndex,
                         final TwoClassConfusionMatrix confusionMatrix, final TwoClassConfusionMatrix stats, final TwoClassConfusionMatrix recall,
                         final TwoClassConfusionMatrix recallEval) {
        
    	/*
    	 * first: count entries for Precision in F-Measure
    	 * second: count entries for Recall in F-Measure (only for micro-averaging)
    	 */    	
    	
    	double labelValue = getLabelValue(instance, labelIndex);
        Condition labelAttribute = head.getCondition(labelIndex);
        
        if (covers) {
			if (labelAttribute != null) {
                if (labelAttribute.getValue() != labelValue) {
                	confusionMatrix.addFalsePositives(instance.weight());

                    if (stats != null) {
                        stats.addFalsePositives(instance.weight());
                    }
                } else {
                	confusionMatrix.addTruePositives(instance.weight());

                    if (stats != null) {
                        stats.addTruePositives(instance.weight());
                    }
                }
            } else {
                if (labelValue == 1) {
                	confusionMatrix.addFalsePositives(instance.weight());

                    if (stats != null) {
                        stats.addFalsePositives(instance.weight());
                    }
                } else {
                	confusionMatrix.addTruePositives(instance.weight());

                    if (stats != null) {
                        stats.addTruePositives(instance.weight());
                    }
                }
            }
        } else {
        	if (labelValue == 1) {
        		confusionMatrix.addFalseNegatives(instance.weight());

                if (stats != null) {
                    stats.addFalseNegatives(instance.weight());
                }
            } else {
            	confusionMatrix.addTrueNegatives(instance.weight());

                if (stats != null) {
                    stats.addTrueNegatives(instance.weight());
                }
            }
        }
        
        
        switch (optimizationHeuristic) {
        	case "classic":
        		if (covers) {
        			if (labelAttribute != null) {
            			if (labelValue == 1) {
            				if (labelAttribute.getValue() == 1) {
            					recall.addTruePositives(instance.weight());
            				} else {
            					recall.addFalseNegatives(instance.weight());
            				}
            			} else {
            				if (labelAttribute.getValue() == 1) {
            					recall.addFalsePositives(instance.weight());
            				} else {
            					recall.addTrueNegatives(instance.weight());
            				}
            			}
            		} else {
            			if (labelValue == 1) {
            				recall.addFalseNegatives(instance.weight());
            			} else {
            				recall.addTrueNegatives(instance.weight());
            			}
            		}
        		} else {
        			if (labelValue == 1) {
        				recall.addFalseNegatives(instance.weight());
        			} else {
        				recall.addTrueNegatives(instance.weight());
        			}
        		}
        		break;
        		
        		
        	case "adapted":
        		if (covers) {
        			if (labelAttribute != null) {
                        if (labelAttribute.getValue() != labelValue) {
                        	recall.addFalsePositives(instance.weight());
                        } else {
                        	recall.addTruePositives(instance.weight());
                        }
                    } else {
                        if (labelValue == 1) {
                        	recall.addFalsePositives(instance.weight());
                        } else {
                        	recall.addTruePositives(instance.weight());
                        }
                    }
                } else {
                	if (labelValue == 1) {
                		recall.addFalseNegatives(instance.weight());
                    } else {
                    	recall.addTrueNegatives(instance.weight());
                    }
                }
        		break;
        		
        		
        	case "clever":
        		if (covers) {
        			if (labelAttribute != null) {
                        if (labelAttribute.getValue() != labelValue) {
                        	recall.addFalsePositives(instance.weight());
                        } else {
                        	recall.addTruePositives(instance.weight());
                        }
                    } else {
                        if (labelValue == 1) {
                        	recall.addFalsePositives(instance.weight());
                        } else {
                        	recall.addTruePositives(instance.weight());
                        }
                    }
        		} else {
        			if (labelAttribute != null) {
            			if (labelAttribute.getValue() != labelValue) {
            				recall.addTrueNegatives(instance.weight());
            			} else {
            				recall.addFalseNegatives(instance.weight());
                        }
            		} else {
                        if (labelValue == 1) {
                        	recall.addTrueNegatives(instance.weight());
                        } else {
                        	recall.addFalseNegatives(instance.weight());
                        }
                    }
        		}
        		break;
        		
        		
        	case "covered":
        		if (covers) {
        			if (labelAttribute != null) {
                        if (labelAttribute.getValue() != labelValue) {
                        	recall.addFalsePositives(instance.weight());
                        } else {
                        	recall.addTruePositives(instance.weight());
                        }
                    } else {
                        if (labelValue == 1) {
                        	recall.addFalsePositives(instance.weight());
                        } else {
                        	recall.addTruePositives(instance.weight());
                        }
                    }
        		}
        		break;
        		
        		
        	default: throw new IllegalArgumentException("Given optimization heuristic is not specified");
        
        }    
        
        
        
        switch (evaluationHeuristic) {
    	case "classic":
    		if (covers) {
    			if (labelAttribute != null) {
        			if (labelValue == 1) {
        				if (labelAttribute.getValue() == 1) {
        					recallEval.addTruePositives(instance.weight());
        				} else {
        					recallEval.addFalseNegatives(instance.weight());
        				}
        			} else {
        				if (labelAttribute.getValue() == 1) {
        					recallEval.addFalsePositives(instance.weight());
        				} else {
        					recallEval.addTrueNegatives(instance.weight());
        				}
        			}
        		} else {
        			if (labelValue == 1) {
        				recallEval.addFalseNegatives(instance.weight());
        			} else {
        				recallEval.addTrueNegatives(instance.weight());
        			}
        		}
    		} else {
    			if (labelValue == 1) {
    				recallEval.addFalseNegatives(instance.weight());
    			} else {
    				recallEval.addTrueNegatives(instance.weight());
    			}
    		}
    		break;
    		
    		
    	case "adapted":
    		if (covers) {
    			if (labelAttribute != null) {
                    if (labelAttribute.getValue() != labelValue) {
                    	recallEval.addFalsePositives(instance.weight());
                    } else {
                    	recallEval.addTruePositives(instance.weight());
                    }
                } else {
                    if (labelValue == 1) {
                    	recallEval.addFalsePositives(instance.weight());
                    } else {
                    	recallEval.addTruePositives(instance.weight());
                    }
                }
            } else {
            	if (labelValue == 1) {
            		recallEval.addFalseNegatives(instance.weight());
                } else {
                	recallEval.addTrueNegatives(instance.weight());
                }
            }
    		break;
    		
    		
    	case "clever":
    		if (covers) {
    			if (labelAttribute != null) {
                    if (labelAttribute.getValue() != labelValue) {
                    	recallEval.addFalsePositives(instance.weight());
                    } else {
                    	recallEval.addTruePositives(instance.weight());
                    }
                } else {
                    if (labelValue == 1) {
                    	recallEval.addFalsePositives(instance.weight());
                    } else {
                    	recallEval.addTruePositives(instance.weight());
                    }
                }
    		} else {
    			if (labelAttribute != null) {
        			if (labelAttribute.getValue() != labelValue) {
        				recallEval.addTrueNegatives(instance.weight());
        			} else {
        				recallEval.addFalseNegatives(instance.weight());
                    }
        		} else {
                    if (labelValue == 1) {
                    	recallEval.addTrueNegatives(instance.weight());
                    } else {
                    	recallEval.addFalseNegatives(instance.weight());
                    }
                }
    		}
    		break;
    		
    	/*	
    	case "covered":
    		if (covers) {
    			if (labelAttribute != null) {
                    if (labelAttribute.getValue() != labelValue) {
                    	recallEval.addFalsePositives(instance.weight());

                        if (false && stats != null) {
                            stats.addFalsePositives(instance.weight());
                        }
                    } else {
                    	recallEval.addTruePositives(instance.weight());

                        if (false && stats != null) {
                            stats.addTruePositives(instance.weight());
                        }
                    }
                } else {
                    if (labelValue == 1) {
                    	recallEval.addFalsePositives(instance.weight());

                        if (false && stats != null) {
                            stats.addFalsePositives(instance.weight());
                        }
                    } else {
                    	recallEval.addTruePositives(instance.weight());

                        if (false && stats != null) {
                            stats.addTruePositives(instance.weight());
                        }
                    }
                }
    		}
    		break;
    		*/
    		
    	default: throw new IllegalArgumentException("Given evaluation heuristic is not specified");
    
    }        
        
        
    }

    public final MetaData evaluate(final Instances instances, final MultiHeadRule rule,
                                   final Heuristic heuristic,
                                   final Collection<Integer> relevantLabels, final MetaData metaData,
                                   final String optimizationHeuristic,
                                   final String evaluationHeuristic) {
        TwoClassConfusionMatrix stats = new TwoClassConfusionMatrix();
        TwoClassConfusionMatrix recall = new TwoClassConfusionMatrix();
        TwoClassConfusionMatrix recallEval = new TwoClassConfusionMatrix();
        setHeuristics(optimizationHeuristic, evaluationHeuristic);
        if (metaData != null) {
            stats.addTrueNegatives(metaData.stats.getNumberOfTrueNegatives());
            stats.addFalseNegatives(metaData.stats.getNumberOfFalseNegatives());
        }

        MetaData result = evaluate(instances, rule, heuristic, relevantLabels, metaData, stats, recall, recallEval);
        rule.setStats(stats);
        return result;
    }

    protected abstract MetaData evaluate(final Instances instances, final MultiHeadRule rule, final Heuristic heuristic,
                                         final Collection<Integer> relevantLabels, final MetaData metaData,
                                         final TwoClassConfusionMatrix stats, final TwoClassConfusionMatrix recall,
                                         final TwoClassConfusionMatrix recallEval);

    public static AveragingStrategy create(final String strategy) {
        if (strategy.equalsIgnoreCase(MICRO_AVERAGING)) {
            return new MicroAveraging();
        } else if (strategy.equalsIgnoreCase(LABEL_BASED_AVERAGING)) {
            return new LabelBasedAveraging();
        } else if (strategy.equalsIgnoreCase(EXAMPLE_BASED_AVERAGING)) {
            return new ExampleBasedAveraging();
        } else if (strategy.equalsIgnoreCase(MACRO_AVERAGING)) {
            return new MacroAveraging();
        }

        throw new IllegalArgumentException("Invalid averaging strategy: " + strategy);
    }

}