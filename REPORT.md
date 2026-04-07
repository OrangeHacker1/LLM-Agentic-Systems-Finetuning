# Methodology
This project was broken into 6 phases. The phases are as follows:
1. Data Construction
2. Stage 1 Training
3. Stage 2 Training
4. Judge Analysis
5. Forgetting Analysis
6. Albiration Study

Data construction involves creating 2 datasets. The first is an Alpaca style dataset from xxx. This then uses an HPC server. The fine tuning will be done with QLoRA. The second dataset will involveusing a 70B parameter model (xxx) to provide training data for stage 2.     
Stage 1 training will be used to take the initial model (xxx), and do a basic imitation training on the HPC server to create a new model that will have improved accuracy with its structure, should there be no issues.   
Stage 2 Training is similar to stage 1. It will train with dataset 2 and further build upon the first model. This will further refine the model to produce even more accurate results.   
**NOTE**: It is vital for this project that all three of the models, initial, Stage 1, and Stage 2, are saved / screenshot. They will be needed for later experiments.   
Judge Analysis is a measure of how well the models have performed in produceing the desired results with the following tasks (xxx). This will then use a judge model, 70B parameter (xxx). The judge will then compare the models against one another to test the varrying levels of correctness and structured output. This is one of the most vital points for this experiment, as improvements are necessary to see if the models are properly being changed or if revisions are needed to further fine tune the models.    
The next phase involves testing how much of an impact the retraining had on the models. This will be a vital part of the project, as it shows the trade offs of the additional training. This will combine with the Judge Analysis to determine if the method is overall worth it to produce a better structured output. Depending on the results, the best model (Initial, Stage 1, or Stage 2) will be found to be the best for this project / evaluation.   

In order to complete this project, we needed to use several dependencies and hyperparameters. The hyperparameters used for this project are as follows:   

(Parameters)



# Experiments

# Analysis

# Prompt Engineering

# Appendix: Full Prompts
