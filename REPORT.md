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
    
      checkpoint:
        stage1_path: ./outputs/stage1

(Dependencies)     
The dependencies are in the requirements.txt file and setup_enev.sh.   

# Experiments

## Evaluation Metrics

Structured Output - Measures the ability to produce a valid json structure.    
General Instruction Following (Alpacha Dataset) - The ability to answer questions from the alpaca dataset.    
Forgetting Analysis - The impact the stage 2 training had on the model.    

## Charts

Alpaca Accuracy and Json Structure Accuracy

| Metric | Value |
|------|------|
| JSON Valid Rate | 0.0 |
| JSON Teacher Pass Rate | 0.7 |
| Alpaca Score | 7.96 |

Forgetting Analysis

| Metric | Value |
|------|------|
| Stage 1 Win Rate | 0.20 |
| Stage 2 Win Rate | 0.0325 |
| Tie Rate | 0.7675 |
| Forgetting Detected | No |


Final Model Accuracy (Cleaned data for JSON training)
| Model Checkpoint | Alpaca Judge Win Rate | ROUGE-L / BERTScore | JSON Validity | Schema Compliance | Exact Match |
|------------------|-----------------------|---------------------|---------------|-------------------|-------------|
| Stage 0: Base | xxx | xxx | xxx | xxx | xxx |
| Stage 1: Alpaca | xxx | xxx | xxx | xxx | xxx |
| Stage 2: Teacher JSON | xxx | 7.958 | 0.0 | 0.7 | xxx |

## Json Structure Evaluation

This is where the results seem a bit strange. According to the teacher model, the stage two model does answer ccorrectly. However, there seems to be an issue creating the valid json format. This can be seen due to the 0% json valid rate. This needs to be looked into. I am planning to run an albanation study to see if training rate has any effect on this. The json validator does seem to be working, but I would need to edit the code to have the LLM give all its results to see what it is doing wrong. (My guess is that it might be defining the problem or using think blocks before the json.)    

## Alpacha Evaluation

After training the model to answer with a json format, the model maintained a general score around 8 out of 10 for all of the evaluation prompts. This showed that the model was able to retain its ability to answer the alpaca questions.   
According to ChatGPT, I had it look over the 2000+ json results, failures typically occured with computational tasks, like arithmatic.    

## Forgetting Analysis

Based on the results given, stage two takes a slight hit to its accuracy. This shows that there was a decent bit of damage done to the mode. This shows that the stage 2 training is updating the model and causing issues. The initial model performed better, as expected.

## Ablation Study

Errors occurred during saving. This resullted in needing to redo the Ablation study.      
    
ablation LR: 2e05      

| Metric | Value |
|------|------|
| JSON Valid Rate | 0.0 |
| JSON Teacher Pass Rate | 0.72 |
| Alpaca Score | 8.03 |

ablation LR: 5e05      

| Metric | Value |
|------|------|
| JSON Valid Rate | 0.0 |
| JSON Teacher Pass Rate | 0.72 |
| Alpaca Score | 8.03 |

ablation LR: .00001     

| Metric | Value |
|------|------|
| JSON Valid Rate | 0.0 |
| JSON Teacher Pass Rate | 0.72 |
| Alpaca Score | 8.03 |

# Analysis

Qualitative comparison of outputs across
checkpoints, failure case analysis, discussion of
forgetting vs retention, what the results imply about
sequential fine-tuning

Looking at the results, it can be implied that the model had issues learning how to give json only responses. The hardcoded json would fail 100% of the time. This implies there was failure in the training data. In order to improve the results, it is necessary to change the training data. Occording to the LLM I trained on, most of the results are actually acurate. This lead me to questioning how far off the trained model actually was. This was the focus of the albiration study. By testing the learning rate, I could determine whether there was just a biit more of training needed vs the training data being unreliable.    
Looking over the training data, I found that there were some issues. The structure was somewhat inconsistent. The structure would change based onn the task being done. If I were to constrain or unify the training data, it would be possible to increase the accuracy. There were weak formatting issues. The trainer model also add extra stuff. One solution for better results is to limit the amount of tasks. Another issue was the fix json tasks. This could cause confusion.     
After analysis, there were several things that would need to be tested and further debugged. First, the model for stage 2 should have rules added. By adding rules, it is possible to improve the results. This might scew results, but the teacher model said that the idea and scemantics were good.    
Looking at explicit examples, using ChatGPT to scan the file, there were examples that were using single quotes, examples that did not add quotes, and examples with extra text in the output, like: "Here is your json: {STUFF}". In order to increase the results, it is necessary to clean the data and retrain the models This will show if cleaned data is better.    
Final Improvements:

        Clean Training Data (Use a clean function to remove false training data.)
        Add Rules and Restrictions (Could help, but does not show if the model is working properly. It should default answer in json.)
        Narrow Training Data (This will help it better match the structure and produce json files.)

# Prompt Engineering

While doing this project, I worked thorugh the prompts to create a generic question answering LLM mode. TO achieve this, the prompts could only use bare bone calls. Otherwise, this will result in seeing if Zero Shot is effective, as opposed to if the actual LLM has internally learned the patterns.    
In order to better improve the efficiency of the Judge, I had the judge go into specific details to determine which response was better for answering the question. The Judge prompt can be seen below. (This is the pairwise prompt.)  
A second judge script was added to see if the LLM schematic was actually working properly. This has a decent level of leway with responses. This was why, despite a zero percent pass rate, the teacher rated the stage 2 model highly.     

# Appendix: Full Prompts

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
