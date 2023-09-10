# LLM-GPT-model-
This model goes deep into data handling, math, and transformers behind large language models.
The project encompasses 'pre-training' extensively and several key components:

1. Data Extraction: I've gathered and prepared the data necessary to train my language model, a crucial step in achieving accurate and relevant responses.

2. Transformer Architecture (Decoders): The heart of my model lies in the Transformer architecture, specifically focusing on the decoder section. This design choice enables the model to generate contextually appropriate responses.

3. Base Chatbox (Untrained): As a starting point, I've developed a basic chatbot using the untrained model. This lays the foundation for further fine-tuning and improvement.

4. GPU Utilization in Training: A module displays how GPU the training process, allowing for efficient parallel processing and faster convergence.

5. PyTorch Function Visualization: A module created to visualize PyTorch functions, providing valuable insights into the underlying operations during training and inference.

6. Scalability and Parallelism: By incorporating parallel processing, your model is capable of handling large datasets and can potentially be scaled for more complex tasks.

7. Potential for Fine-tuning: With a base chatbot in place, there is room for further fine-tuning and refinement based on specific use cases or domains.

8. Strong Emphasis on PyTorch Expertise: The project showcases a deep understanding of PyTorch, as evidenced by the successful creation and training of a custom GPT model.

9. Foundation for NLP Applications: This project lays the groundwork for various natural language processing (NLP) applications, demonstrating a solid foundation in both theory and practical implementation.

10. Bigram Modeling: A Bigram model is a simple but effective approach to language modeling. It's based on the assumption that the probability of a word in a sentence is influenced primarily by the preceding word.
                     A Bigram model is a type of probabilistic model used for predicting the next word in a sequence of words. It considers pairs of adjacent words, known as bigrams.
                     Unlike more complex models like Transformers, which consider long-range dependencies, Bigram models focus on local context.
                     Bigram models are computationally efficient and can be trained on relatively small datasets. This makes them a popular choice for tasks like text prediction and spell checking.
                     Bigram models find applications in various NLP tasks such as language modeling, text prediction, spell checking, and part-of-speech tagging.

Command Line: After importing all neccessary libraries/modules,
              enable cuda, the virtual enviournment by 'activate' script (CUDA Toolkit, is a development platform for GPU (graphics processing unit) computing. CUDA provides libraries and tools for programming NVIDIA GPUs for general-purpose computing.)
              With 'jupyter notebook' opened choose the custom kernel created using 'python -m -ipykernel -install --user --name=cuda --display-name "kernelname"' to ensure the virtual enviournment is within the notebook and is interactable.
              For data extraction: 'python data-extract.py'
              Chatbot: 'python chatbot.py -batch_size 32' 32 or whatever size of batch size we want
