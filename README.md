# LLM-Agentic-Systems-Finetuning
The goal is to see fine tuning a small LLM on Alpaca-style instruction data and continuing to fine tune a JSON instruction dataset with a stronger LLM teacher model can improve structure output reliability without losing its ability to answer questions.     

# SETUP
This project is broken up into two sections. One portion of then code is designed to build the datasets for evaluation and training. For this project, I will include the json files used for reproducability.    
The primary code for this project was run on an HPC server, the UTSA ARC server. This is where the code was loaded and run. Anaconda was enabled before starting the project. The project's code does not activate the python enviornment.      

You will also need to add an '.env' file.   

        TEACHER_BASE_URL=http://10.246.100.230/v1 
        #your_api_url_here
        TEACHER_MODEL="llama-3.3-70b-instruct-awq" 
        #gpt-4o
        TEACHER_API_KEY=<REDACTED>

# How to Run
This project has 5 main bash files. It is important to note that you will need to update the bash files to match your working directory, your abc123. The bash code usage is as follows:

This is for setting up the enviornment.    

    bash setup_env.sh

  Train the initial model.   
    
    bash run_phase2.sh

This is for fine tuning and updating the model to better fit json format.

    bash run_phase3.sh
    
Run the evaluation section of the project. It will test evaluation for forgetting and the accuracy of answering questions.   

    bash run_phase4.sh

After going through the training, the stage 2 model was not able to give propper json format. However, the model understood what it was supposed to do. In order to see if training could help improve the structure, I tested training rates.   

    bash run_abation.sh

