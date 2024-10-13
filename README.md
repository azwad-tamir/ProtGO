# ProtGO
## A Transformer based fusion model for accurately predicting Gene Ontology (GO) terms from full-scale Protein Sequences

Recent developments in next-generation sequencing technology have led to the creation of extensive, open-source protein databases consisting of hundreds of millions of sequences. To render these sequences applicable
in biomedical applications, they must be meticulously annotated by wet lab testing or extracting them from
existing literature. Over the last few years, researchers have developed numerous automatic annotation
systems, particularly deep learning models based on machine learning and artificial intelligence, to address
this issue. In this work, we propose a transformer-based fusion model capable of predicting Gene Ontology (GO)
terms from full-scale protein sequences, achieving state-of-the-art accuracy compared to other contemporary
machine learning annotation systems. The approach performs particularly well on clustered split datasets,
which comprise training and testing samples originating from distinct distributions that are structurally
diverse. This demonstrates that the model is able to understand both short and long-term dependencies
within the enzyme’s structure and can precisely identify the motifs associated with the various GO terms.
Furthermore, the technique is lightweight and less computationally expensive compared to the benchmark
methods, while at the same time not unaffected by sequence length, rendering it appropriate for diverse
applications with varying sequence lengths.

Dataset link: https://drive.google.com/file/d/1bZD67DqXv9LkYo0HCCEXW4USjgjgqBAY/view?usp=sharing

## Running Instructions:

step1: Clone the repository in the local machine

step2: Download the models from the following directory and paste it into the project root:
https://drive.google.com/file/d/1ObwqMIGE6A-gjr3lOTjaAWDhP0kbsJjL/view?usp=sharing

step3: Install pytorch by running the following in the terminal:

>> pip install torch==1.10.2+cpu torchvision==0.11.3+cpu --extra-index-url https://download.pytorch.org/whl/cpu

step4: Install other necessary libraries by running the following in the terminal:

>> pip install -r requirements.txt

step5: Run the training scripts to train the model and run evaluations.

