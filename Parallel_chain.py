from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3.2-Exp",
    task = "text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-R1",
    task = "text-generation"
)
model1 = ChatHuggingFace(llm=llm1)

model2= ChatHuggingFace(llm=llm2)

prompt1= PromptTemplate(
    template='Generate short and simple notes from the following  text \n {text}',
    input_variables=['text']
)

prompt2= PromptTemplate(
    template='Generate 5 short question anser from the following text\n {text}',
    input_variables=['text']
)

prompt3= PromptTemplate(
    template='merge the provided notes anf quize into the following into a singe document \n {notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

paralal_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
})

merge_chain= prompt3 | model1 | parser

chain= paralal_chain | merge_chain
text = """---BEGIN DOCUMENT---
Title: Generative AI — A Practical Overview
Author: ChatGPT (GPT-5 Thinking mini)
Date: 2025-10-04

1. Introduction
Generative Artificial Intelligence (Generative AI) refers to a class of machine learning methods that create new data samples which resemble a training dataset. These systems can produce text, images, audio, video, 3D models, code, and other modalities. Unlike discriminative models (which classify or predict), generative models model the underlying data distribution and sample from it.

2. Key Concepts and Definitions
- Data distribution: The probability distribution that generated the observed data.
- Latent space: A lower-dimensional representation where generative models map inputs and sample new outputs.
- Conditional generation: Producing outputs given some input condition (e.g., text-to-image, image captioning).
- Unconditional generation: Producing outputs without a specific condition (e.g., sampling from a trained image generator).

3. Main Types of Generative Models
- Generative Adversarial Networks (GANs): Two networks (generator and discriminator) trained adversarially. Strengths: high-quality images. Weaknesses: training instability, mode collapse.
- Variational Autoencoders (VAEs): Encoder-decoder architecture that learns a probabilistic latent space. Strengths: principled probabilistic framework, smooth latent interpolations. Weaknesses: sometimes blurrier outputs than GANs.
- Autoregressive Models: Models that factorize joint distribution into product of conditionals (e.g., GPT-family for text, PixelRNN for images). Strengths: strong likelihood modeling, great for text. Weaknesses: slow sampling for high-dimension outputs.
- Diffusion Models: Iteratively denoise from pure noise to data using learned reverse process. Strengths: high-fidelity samples, stable training. Weaknesses: can require many sampling steps (though sampling speedups exist).
- Flow-based Models: Learn invertible transformations with exact likelihoods (e.g., RealNVP, Glow). Strengths: exact likelihood and fast sampling. Weaknesses: architectural constraints and sometimes lower sample quality.
- Energy-Based Models and Score Matching: Another class, closely related to diffusion methods and probabilistic modeling.

4. Typical Applications
- Text generation: chatbots, content creation, summarization, code generation.
- Image generati
"""
result=chain.invoke({'text':text})
print(result)


