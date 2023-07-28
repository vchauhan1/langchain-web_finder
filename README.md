## Langchain Web retriver/finder
The Langchain Web Retriever/Finder is a tool designed to retrieve and find information from the web using a locally hosted model on an Apple M1 laptop. If you prefer to use the OpenAI or a compatible API, you can do so by exporting the necessary environment variables as shown below:


### Installation
To run the program, first, create a virtual environment and activate it using Conda:
```
conda create --name web_finder python=3.10
conda activate web_finder
```
### OpenAI API USE
This version of program is using locally hosted model on Apple M1 laptop, If you want to use OpenAI or compatible API you can export them like below. 

```
export GOOGLE_API_KEY=xxx
export GOOGLE_CSE_ID=xxx
export OPENAI_API_KEY=xxx
```
### Local Model Setup
If you choose to use the locally hosted model, follow these additional steps:
Place you model file in models directory, Example:
``` 
models/llama-7b.ggmlv3.q4_0.bin 
```

Export the required environment variables for the internet search model:
```
export GOOGLE_API_KEY=xxx
export GOOGLE_CSE_ID=xxx
```

### Running the Application
```
streamlit run web_finder.py
```