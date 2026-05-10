# Methodology
Generative AI was used to create the code for this project and help with debugging. It was also used to go over the results and help outline the results.        
This project was broken into 6 phases. The phases are as follows:
1. Data Construction
2. Stage 1 Training
3. Stage 2 Training
4. Judge Analysis
5. Forgetting Analysis
6. Albiration Study

Data construction involves creating 2 datasets. The first is an Alpaca style dataset from xxx. This then uses an HPC server. The fine tuning will be done with QLoRA. The second dataset will involveusing a 70B parameter model phi-3-mini-4k-instruct to provide training data for stage 2.     
Stage 1 training will be used to take the initial model phi-3-mini-4k-instruct, and do a basic imitation training on the HPC server to create a new model that will have improved accuracy with its structure, should there be no issues.   
Stage 2 Training is similar to stage 1. It will train with dataset 2 and further build upon the first model. This will further refine the model to produce even more accurate results.   
**NOTE**: It is vital for this project that all three of the models, initial, Stage 1, and Stage 2, are saved / screenshot. They will be needed for later experiments. The results for stage one are saved and used for reproducability.       
Judge Analysis is a measure of how well the models have performed in produceing the desired results with the following tasks from alpacha_eval.jsonl. This will then use a judge model, 70B parameter llama-3.3-70b-instruct-awq. The judge will then compare the models against one another to test the varrying levels of correctness and structured output. This is one of the most vital points for this experiment, as improvements are necessary to see if the models are properly being changed or if revisions are needed to further fine tune the models.       
The next phase involves testing how much of an impact the retraining had on the models. This will be a vital part of the project, as it shows the trade offs of the additional training. This will combine with the Judge Analysis to determine if the method is overall worth it to produce a better structured output. Depending on the results, the best model (Initial, Stage 1, or Stage 2) will be found to be the best for this project / evaluation. If the models have a lot of ties, then the model has maintained its ability to answer questions.     

In order to complete this project, we needed to use several dependencies and hyperparameters. The hyperparameters used for this project are as follows:   

(Parameters)
                        
            ###
            # Configuration for Teacher Dataset Generation
            ###
            
            generation:
              temperature: 0.2        # LOW = deterministic JSON
              max_tokens: 512         # JSON doesn’t need huge outputs
              retries: 3              # retry invalid JSON
            
            # Temperature: This temperature will help m,aintain consistency.
            # Retries: This will help make sure there are no invalid json files.
            
            dataset:
              path: data/alpaca_train.jsonl
              max_samples: 1000
              eval_split: 0.05     # 5% evaluation set
              train_size: 1000     # Teacher Model
              eval_size: 100       # Teacher Model
            
            tasks:
              types:
                - extraction
                - classification
                - schema
                - repair
                - tool
            
            logging:
              path: logs/data_generation/
            
            self_consistency:
              samples: 1              # Not needed here (keep 1)
            
            
            ###
            # NEW: Training Configuration (Stage 1)
            ###
            
            model:
              name: microsoft/phi-3-mini-4k-instruct
              max_length: 1024
            
            training:
              output_dir: ./outputs/stage1
              batch_size: 4
              eval_batch_size: 4
              gradient_accumulation_steps: 4
              epochs: 3
              learning_rate: 2e-5
              logging_steps: 25
              save_steps: 500
              eval_steps: 500
              bf16: true
              
            
            lora:
              r: 16
              alpha: 32
              dropout: 0.05
              target_modules:
                - q_proj
                - k_proj
                - v_proj
                - o_proj
            
            ###
            # NEW: Evaluation Configuration (Stage 4)
            ###
            evaluation:
              alpaca_eval_path: data/eval/alpaca_eval.jsonl
              json_eval_path: data/eval/json_eval.jsonl
              output_dir: results/phase4
              output_dir_new: results/phase4_NEW
              max_new_tokens: 256
              judge_temperature: 0
            
            ###
            # NEW: Training Configuration (Stage 2)
            ###
            stage2:
              output_dir: ./outputs/stage2
              learning_rates:
                - 2e-5     # baseline
                - 5e-5     # medium
                - 1e-4     # high
            
              batch_size: 4
              gradient_accumulation_steps: 4
              epochs: 2
            
            
            # Checkpoint Directories
            # stage2_path_final is for debugging.
              checkpoint:
                stage1_path: ./outputs/stage1_adapter
                stage2_path: ./outputs/stage2_final
                stage2_path_final: ./outputs/stage2_final

(Dependencies)     
The dependencies are in the requirements.txt file and setup_enev.sh.   

# Experiments

## Evaluation Metrics

Structured Output - Measures the ability to produce a valid json structure.    
General Instruction Following (Alpacha Dataset) - The ability to answer questions from the alpaca dataset.    
Forgetting Analysis - The impact the stage 2 training had on the model.    

## Charts

Initial Alpaca Accuracy and Json Structure Accuracy

| Metric | Value |
|------|------|
| JSON Valid Rate | 0.0 |
| JSON Teacher Pass Rate | 0.7 |
| Alpaca Score | 7.96 |

Initial Forgetting Analysis

| Metric | Value |
|------|------|
| Stage 1 Win Rate | 0.20 |
| Stage 2 Win Rate | 0.0325 |
| Tie Rate | 0.7675 |
| Forgetting Detected | Yes |


Second Model Accuracy (Cleaned data for JSON training)
| Model Checkpoint | Json Teacher Pass Rate | ROUGE-L | BERTScore | JSON Validity | Schema Compliance | Exact Match |
|------------------|-----------------------|-----------|----------|---------------|-------------------|-------------|
| Stage 0: Base | 0.567 | 0.215 | 0.866 | 0.359 | 0.359 | 0.0 |
| Stage 1: Alpaca | 0.563 | 0.216 | 0.863 | 0.129 | 0.128 | 0.0 |
| Stage 2: Teacher JSON | 0.563 | 0.216 | 0.863 | 0.129 | 0.128 | 0.0 |

Final Model Accuracy
| Model Checkpoint | Alpaca Judge Score | ROUGE-L | BERTScore | JSON Validity | Schema Compliance | Exact Match | Json Teacher Pass Rate |
|------------------|-----------------------|-----------|----------|---------------|-------------------|-------------|--------------|
| Stage 0: Base | 12.75 | 0.215 | 0.866 | 0.01 | 0.0 | 0.01 | 0.091 |
| Stage 1: Alpaca | 7.65 | 0.216 | 0.863 | 0.0 | 0.0 | 0.0 | 0.158 |
| Stage 2: Teacher JSON | 7.69 | 0.249 | 0.872 | 0.001 | 0.001 | 0.0075 | 0.213 |

Second Forgetting Analysis

| Metric | Value |
|------|------|
| Stage 1 Win Rate | 0.035 |
| Stage 2 Win Rate | 0.005 |
| Tie Rate | 0.96 |
| Forgetting Detected | Yes |

Final Forgetting Analysis
| Metric | Value |
|------|------|
| Stage 1 Win Rate | 0.468|
| Stage 2 Win Rate | 0.138 |
| Tie Rate | 0.395 |
| Forgetting Detected | Yes |


## Json Structure Evaluation

While doing this project, there were three main runs.   
Initially, the first major run revealed a flaw in the initial dataset. This led to a 0% for the jason validity for the jason teacher dataseet. Due to an error found early on, the first and base models were skipped.       
The second run was done with a more generous json validater, as shown above. This revealed that there was an issue loading the lora models. After fixing the lora model, it was revealed that there was
After going over the training data, I created a new dataset. This dataset was cleaned to ensure that the data sent would be valid sintax for json. After running the training for stage 3 again, the model started properly answering with json formatting, although the results were low. Although this is also shown in the base model. An error occurred in the new json load. This would require changing to the original check.     



## Alpacha Evaluation

After training the model to answer with a json format, the model maintained a general score around 8 out of 10 for all of the evaluation prompts. This showed that the model was able to retain its ability to answer the alpaca questions.   
According to ChatGPT, I had it look over the 2000+ json results, failures typically occured with computational tasks, like arithmatic.  
The alpach values remained stable throughout the three models.

## Forgetting Analysis

The model shows slight levels of forgetting in its intial view. This was proportional to the slight improvements made to the model (stage 2). This shows thsat there is a trade off in order to increase the abilities of another ability. The final forgetting test shows that the quality takes a major hit compared to the first model.   
This shows that the alpacha Judge score does not reflect a perfect match to accuracy, as the stage 2 model outperformed initially. The forgetting analysis implies forgetting, or a hit to quality, as the Judge model had a significant advantage.     


## Ablation Study

The accuracy for json structure was extremely low. In order to test if it was the fault of the training data, I desidded to test the learning rate of the model.    
This albination study will also be used to test the level of forgetting (the effects the learning has on the alpaca score). Another important goal was to test whether the models are being propperly loaded. If learning rate has no impact, then it can be assumed that the models are either not training properly or are not loading propperly.   

Second Ablation Study
| Learning Rate | Json Teacher Pass Rate | ROUGE-L | BERTScore | JSON Validity | Schema Compliance | Exact Match |
| Stage 2: 2e5 | 0.563 | 0.216 | 0.863 | 0.129 | 0.128 | 0.0 |  
| Stage 2: 5e5 | 0.563 | 0.216 | 0.863 | 0.129 | 0.128 | 0.0 |
| Stage 2: 1e4 | 0.563 | 0.216 | 0.863 | 0.129 | 0.128 | 0.0 |

Final Ablation Study
| Learning Rate | Alpaca Judge Win Rate | ROUGE-L | BERTScore | JSON Validity | Schema Compliance | Exact Match | Json Teacher Pass Rate |
| Stage 2: 2e5 | 7.625 | 0.253 | 0.873 | 0.001 | 0.001 | 0.0075 | 0.238 |
| Stage 2: 5e5 | 8.078 | 0.355 | 0.898 | 0.15 | 0.149 | 0.0275 | 0.467 |
| Stage 2: 1e4 | 8.143 | 0.361 | 0.899 | 0.002 | 0.025 | 0.025 | 0.438 |

# Analysis

Qualitative comparison of outputs across
checkpoints, failure case analysis, discussion of
forgetting vs retention, what the results imply about
sequential fine-tuning       

After completing this project, the results revealed several trends for the models. When it came to the alpacha average score, both models 1 and two maintained a high level of accuracy, around 8. This sugjests that the model retained most of its ability throughout the training. This is also explainable due to the amount of parameters being changed was not as high compared to the original model (8 billion). In terms of json accuracy, the json did see improvement, even without using the cleaning method used in the second run. However, the results were miniscule. This likely explains why the forgetting was not significant. The model did not have enough time or paraeters to train.    
Looking at the teacher pass rates also reveals a greater difference. While model 2 wasn't able to replicate perfect / working json well, it did understand the task well. According to this project, the second model performed about twice as well as the initial model according to the teacher LLM.     
In order to test what was causing this phenominon, I decided to do the same albanation again for the final one, especially since the code was already there and the deadline was aproaching. This albanation would show whether the model needed more time to learn the json patterns for improvement. If the models with higher training rates do start to improve, it is highly likely the the alpaca solving abilities will be impacted. This makes sense, as it is necessary to allocate and change weights to get different results, aka json formatting.     
The albination study was unsuccessful since there was a failure during training. After retraining the models, it was shown that increasing the learning rate was able to increase the level json accuracy. This shows that the project is working as intended is immproving the json formatting during stage 2 trainig / Phase 3. Although, overfitting seemse to occur woth a learning rate '1e4'.      
Another important observation is that the Alpacha_average_Score can exceed 10.0. LLMs can make mistakes. There should be a validation method to make sure the score does not exceed 10.0. This would be a great improvement to the project.     
   
Final Improvements:    

        Clean Training Data (Use a clean function to remove false training data.) - Completed
        Add Rules and Restrictions (Could help, but does not show if the model is working properly. It should default answer in json.) -Completed
        Add better prompt questions for generalization. -Failed due to similar reasons. An increase in temperature and some kind of randomness is needed. (An attempt with 90 to 100% could potentially work. However, the json formatting was good based on what I saw.)

# Prompt Engineering

While doing this project, I worked thorugh the prompts to create a generic question answering LLM mode. TO achieve this, the prompts could only use bare bone calls. Otherwise, this will result in seeing if Zero Shot is effective, as opposed to if the actual LLM has internally learned the patterns.    
In order to better improve the efficiency of the Judge, I had the judge go into specific details to determine which response was better for answering the question. The Judge prompt can be seen below. (This is the pairwise prompt.)  
A second judge script was added to see if the LLM schematic was actually working properly. This has a decent level of leway with responses. This was why, despite a zero percent pass rate, the teacher rated the stage 2 model highly.     
After going over the code, I found that the data was not generalizing well enough. The model was answering the same question over and over again. The structure improved, but this was not as good as it could have been. This is likely due to the increase in the amount of data increasing to 20,000 without increasing the amount of custom prompts. Instead, the teacher agent would create a question to match the type of question. This is currently a work in progress. Due to time constraints, this may not have been implimented yet.    
Slight improvements were also made over itterations to better judge the models.   
Another prompt was used to make better questions to answer the schema questions, but the data resulted in the same output. This shouldn't have occurred since the temperature was at .50, but the data ran into the same issue. Unfortunately, there was no time for retraining and making a more in depth dataset, therefore this prompt was excluded due to not being used.    

# Appendix: Full Prompts

Alpaca_Judge:
    
    You are an expert evaluator.
    
    Instruction:
    {instruction}
    
    Reference Answer:
    {reference}
    
    Model Answer:
    {prediction}
    
    Score the model answer from 1 to 10 based on:
    - correctness
    - completeness
    - clarity
    
    Scoring:
    10 = excellent
    7 = good
    5 = partially correct
    3 = poor
    1 = incorrect
    
    Return ONLY a number (1-10).

Forgetting Analysis:
    
    You are comparing two model outputs.
    
    Instruction:
    {instruction}
    
    Reference Answer:
    {reference}
    
    Model A (Checkpoint 1):
    {output_a}
    
    Model B (Checkpoint 2):
    {output_b}
    
    Which is better?
    
    Return ONLY:
    A or B or TIE

Judge Prompt: (Stored in a Python File)

        JUDGE_PROMPT = """
        You are an expert evaluator comparing two model responses.
        
        Evaluate Response A and Response B for the user prompt below.
        
        Be strict and unbiased. Ignore response order.
        
        User Prompt:
        {instruction}
        
        Optional Input:
        {input_text}
        
        Response A:
        {response_a}
        
        Response B:
        {response_b}
        
        Score each response from 1 to 5 on:
        1. instruction_following
        2. correctness
        3. clarity
        4. completeness
        5. structured_output_validity
        6. hallucination_risk
        
        Then choose a winner:
        - "A"
        - "B"
        - "Tie"
        
        Return ONLY valid JSON in this schema:
        
        {
          "response_a_scores": {
            "instruction_following": int,
            "correctness": int,
            "clarity": int,
            "completeness": int,
            "structured_output_validity": int,
            "hallucination_risk": int
          },
          "response_b_scores": {
            "instruction_following": int,
            "correctness": int,
            "clarity": int,
            "completeness": int,
            "structured_output_validity": int,
            "hallucination_risk": int
          },
          "winner": "A|B|Tie",
          "justification": "brief reason"
        }
        """

Pairwise Prompt for Teacher: (This is used for measuring forgetting.)    
        
        PAIRWISE_PROMPT = '''You are an impartial evaluator.
        Task Prompt:\n{prompt}\n\nResponse A:\n{resp_a}\n\nResponse B:\n{resp_b}\n\nScore each response from 1-5 for:
        - instruction_following
        - correctness
        - clarity
        - completeness
        - structured_output_validity
        - hallucination_risk
        
        Return ONLY JSON:
        {{
          "response_a_scores": {{...}},
          "response_b_scores": {{...}},
          "winner": "A|B|Tie",
          "justification": "short reason"
        }}
        '''

Teacher Prompts:
        
        # prompts/teacher_prompts.py
        
        TEACHER_PROMPTS = {
        
            "extraction": """
        Extract structured information as JSON.
        
        Rules:
        - Output ONLY valid JSON
        - No explanations
        - Use double quotes
        
        Schema:
        {
          "name": string,
          "location": string,
          "date": string
        }
        
        Text:
        {input}
        """,
        
            "classification": """
        Classify the following text.
        
        Return ONLY JSON:
        {
          "label": "supported" | "refuted" | "neutral"
        }
        
        Text:
        {input}
        """,
        
            "schema": """
        Convert the text into structured JSON.
        
        Output ONLY JSON with meaningful keys.
        
        Text:
        {input}
        """,
        
            "repair": """
        Fix the following broken JSON.
        
        Return ONLY valid JSON.
        
        Input:
        {input}
        """,
        
            "tool": """
        Generate function arguments in JSON.
        
        Function:
        get_weather(city: string, date: string)
        
        Input:
        {input}
        
        Return ONLY JSON.
        """
        }

Custom Prompts: ( Used for making questions. )
        
        # prompts/custom_prompts.py
        
        CUSTOM_PROMPTS = {
        
            "extraction": [
                # EASY (10)
                "Extract name, location, and date from: John visited Paris on July 5th.",
                "Extract entities from: Alice went to Tokyo on March 3rd.",
                "Extract details from: Bob traveled to NYC on Jan 1.",
                "Find name/location/date: Sarah visited London in June.",
                "Extract structured info: Mike went to Berlin on Feb 2.",
                "Extract details: Emma visited Rome in April.",
                "Extract entities: Tom traveled to Madrid on May 6.",
                "Extract info: Lisa visited Dubai in August.",
                "Extract details: Jack went to Toronto in September.",
                "Extract info: Anna visited Sydney in October.",
        
                # MEDIUM (20)
                "From text extract name/location/date: John and Mary visited Paris on July 5th.",
                "Extract structured entities from paragraph: Alice traveled across Tokyo and Kyoto in March.",
                "Extract key fields: Bob moved from LA to NYC in 2020.",
                "Extract entities from: Sarah visited multiple cities including London and Paris.",
                "Extract name/location/date from mixed text with noise: ### Mike Berlin Feb 2 ###",
                "Extract structured info from paragraph with multiple dates.",
                "Extract entities even if order is scrambled.",
                "Extract fields even if date format varies.",
                "Extract info from paragraph with extra irrelevant text.",
                "Extract structured data from informal sentence.",
        
                # HARD (20)
                "Extract structured JSON from messy paragraph with multiple names and dates.",
                "Extract entities from long paragraph with distractions.",
                "Extract primary subject name/location/date ignoring irrelevant data.",
                "Extract structured fields from noisy OCR-like text.",
                "Extract entities from ambiguous sentence.",
                "Extract info when multiple candidates exist.",
                "Extract main event details from paragraph.",
                "Extract structured data from mixed language text.",
                "Extract key info from narrative paragraph.",
                "Extract entities from multi-sentence paragraph.",
            ],
        
            "classification": [
                # EASY
                "Classify: The evidence supports the claim.",
                "Classify: This is false.",
                "Classify: Not enough information.",
                "Label as supported/refuted/neutral: Evidence confirms hypothesis.",
                "Classify: The claim is wrong.",
        
                # MEDIUM
                "Classify with reasoning: The data partially supports the claim.",
                "Determine label: Mixed evidence exists.",
                "Classify ambiguous statement.",
                "Classify nuanced claim.",
                "Classify uncertain statement.",
        
                # HARD
                "Classify paragraph with conflicting evidence.",
                "Classify long argument.",
                "Determine label from complex reasoning.",
                "Classify scientific claim.",
                "Classify multi-sentence argument.",
            ],
        
            "schema": [
                "Generate JSON for: Product iPhone costs $999 by Apple.",
                "Convert to schema: Tesla Model S costs $80,000.",
                "Create JSON object for: Book titled X by author Y.",
                "Generate structured data for company description.",
                "Convert paragraph into structured JSON.",
                "Generate JSON for nested schema.",
                "Create structured object with multiple attributes.",
                "Generate JSON from semi-structured text.",
                "Convert product description into schema.",
                "Generate structured JSON from messy description.",
            ],
        
            "repair": [
                "Fix JSON: {name: John, location: Paris}",
                "Repair: {'city': 'Austin', 'temp': 75,}",
                "Fix malformed JSON with missing quotes.",
                "Correct JSON with trailing commas.",
                "Repair nested broken JSON.",
                "Fix JSON with wrong types.",
                "Repair JSON missing brackets.",
                "Fix JSON with duplicated keys.",
                "Correct JSON formatting errors.",
                "Repair invalid JSON string.",
            ],
        
            "tool": [
                "Generate arguments: Weather in Austin tomorrow.",
                "Call function: get_weather for NYC Monday.",
                "Create JSON for weather API.",
                "Generate tool arguments from query.",
                "Produce JSON for function call.",
                "Generate API call arguments.",
                "Convert question to tool JSON.",
                "Create structured function parameters.",
                "Generate JSON for API input.",
                "Produce tool call JSON.",
            ]
        }
