 # L'Oreal Hackathon
 
 ## Skin Condition Classification Hackathon

Build a sustainable and efficient machine learning model to classify skin conditions from beauty product descriptions. Our goal is to use the power of Large Language Models (LLMs) to label data and train a compact, high-performing model suitable for production environments. By prioritizing sustainability, we aim to minimize model size and resource consumption while achieving top-notch accuracy. Assisted L'Oreal France in creating innovative, eco-friendly solutions for skin condition classification!

 ## What I Do:
- Label Data with LLMs: Use advanced LLMs (e.g., OpenAI, LLaMA) to generate accurate labels for 6,241 unlabeled beauty product descriptions, leveraging a seed dataset of 200 labeled examples (covering skin conditions like oily, dry, normal, etc.).

- Train a Model: Develop a machine learning model to classify skin conditions based on the labeled dataset.

- Optimize for Sustainability: Design a lightweight model to reduce resource usage and CO2 emissions, measured using CodeCarbon.

- Evaluate Performance: Compete with others by optimizing for accuracy or F1-score while keeping the model efficient.

- Dataset : Labeled Data: 200 examples with:

- Unlabeled Data: 6,241 English descriptions (7–822 words, ~115 words on average)


In our implementation, we:
- Used AWS LLMs with a majority voting system (7 iterations) to create reliable labels.

- Built a custom, lightweight model with bidirectionality to capture subtle text patterns (97% of 24,000 tokens appeared <80 times).
 
- Applied prompt engineering to fine-tune inputs, boosting accuracy.

- Achieved a weighted F1-score of 0.87 while minimizing emissions and training time.


 ## Code Description

The provided Python script trains a multi-label classification model using text data from Final_Claude.xlsx. It:

- Loads the dataset, separating text (text_raw) from label columns.

- Converts text to TF-IDF features.

- Splits data into 80/20 train-test sets.

- Trains a Random Forest classifier within a MultiOutputClassifier.

- Tracks CO2 emissions with CodeCarbon.

- Evaluates performance with accuracy, macro precision, macro F1-score, and per-label metrics.
