# 📚 Machine Learning & Deep Learning Notes — কারিকুলাম

> সম্পূর্ণ বাংলায় লেখা Machine Learning ও Deep Learning-এর বিস্তারিত নোট।  
> প্রতিটি নোটে রয়েছে: Intuition, Math, Code, Diagrams, Real-world use cases।

---

## 📖 প্রতিটি নোটে কী কী আছে?

প্রতিটি নোট নিচের structure অনুসরণ করে লেখা:

1. 🎯 **সহজ ভাষায় পরিচিতি** — Intuition ও বাস্তব উদাহরণ
2. 📖 **বিস্তারিত ব্যাখ্যা** — ধাপে ধাপে deep explanation
3. 📐 **Math / Theory** — সূত্র ও manual calculation
4. 💻 **Code Example** — Runnable Python code with comments
5. 🎨 **Visual / Diagram** — ASCII art ও flowchart
6. ✅ **Real-world Use Cases** — বাস্তব প্রয়োগ
7. ⚖️ **Pros & Cons** — সুবিধা ও অসুবিধা
8. ⚠️ **Common Mistakes** — সাধারণ ভুল ও সতর্কতা
9. 🔗 **Related Topics** — পূর্বশর্ত ও পরবর্তী বিষয়
10. 🧠 **Memory Tricks** — মনে রাখার কৌশল

---

## 🗺️ Learning Syllabus & Tracker

> **টিপস:** প্রতিদিন এখান থেকে সিরিয়ালি ২-৩ টি টপিক প্রম্পট করে এআই-কে দিন। এআই নোট দিলে সেটি সেভ করে রাখুন এবং নিচের চেকলিস্টে (`[x]`) টিক চিহ্ন দিন। 

### 🟢 পর্ব ১: মেশিন লার্নিং (Machine Learning Core)
*এই টপিকগুলো সাধারণত খুব বেশি বড় নয়, তাই একটি করে দিলে এআই সহজেই ভালো নোট বানাতে পারবে।*

- [ ] **১. Introduction to Machine Learning:** Definition, AI vs ML vs DL differences, and Real-world use cases.
- [ ] **২. Types of Machine Learning:** Supervised, Unsupervised, and Reinforcement Learning (Intuition with examples).
- [x] **৩. Data Preprocessing Techniques:** Handling missing values, Encoding categorical data, and Feature Scaling (Standardization vs Normalization). 👉 [নোট পড়ুন](./Data_Preprocessing/Data_Preprocessing.md)
- [x] **৪. Linear Regression:** Core intuition, Best fit line, and Cost Function (Mean Squared Error). 👉 [নোট পড়ুন](./Linear_Regression.md)
- [x] **৫. Gradient Descent Algorithm:** Working mechanism, Learning Rate intuition, and updating weights. 👉 [নোট পড়ুন](./Gradient_Descent/Gradient_Descent.md)
- [x] **৬. Logistic Regression:** Why not linear regression for classification? Sigmoid Function intuition and Decision Boundary. 👉 [নোট পড়ুন](./Logistic_Regression/Logistic_Regression.md)
- [ ] **৭. Decision Trees:** Intuition behind splitting data, Entropy, and Information Gain.
- [ ] **৮. Ensemble Learning & Random Forest:** What is Bagging? How Random Forest combines multiple decision trees.
- [ ] **৯. Support Vector Machine (SVM):** Concept of Hyperplane, Margin, and Introduction to Kernel Trick.
- [ ] **১০. K-Means Clustering:** Unsupervised learning intuition, Centroid initialization, and finding optimal 'K' using Elbow Method.
- [x] **১১. Model Evaluation Metrics for Classification:** Confusion Matrix, Precision, Recall, and F1-Score. 👉 [নোট পড়ুন](./Model_Evaluation_Metrics/Model_Evaluation_Metrics.md)

---

### 🟡 পর্ব ২: ডিপ লার্নিং (Deep Learning Foundations - ANN)
*ANN (Artificial Neural Network) বোঝার জন্য এটিকে ৩টি ভাগে ভাগ করা হয়েছে।*

- [x] **১২. Introduction to ANN:** Biological Neuron vs Artificial Perceptron, Input layer, Hidden layers, Output layer, and Forward Propagation. 👉 [নোট পড়ুন](./ANN_Introduction/ANN_Introduction.md)
- [x] **১৩. Activation Functions in Neural Networks:** Why do we need non-linearity? Detailed intuition of Sigmoid, ReLU, Tanh, and Softmax functions. 👉 [নোট পড়ুন](./Activation_Functions/Activation_Functions.md)
- [x] **১৪. Backpropagation and Optimizers:** How neural networks learn (update weights), Chain Rule intuition, and overview of Optimizers (Stochastic Gradient Descent and Adam). 👉 [নোট পড়ুন](./Backpropagation_Optimizers/Backpropagation_Optimizers.md)
- [x] **১৪.৫ How to Improve Performance of Deep Learning Neural Network:** Data Preprocessing, Batch Normalization, Dropout, L2 Regularization, Weight Initialization (He/Xavier), Transfer Learning, Learning Rate Scheduling, Early Stopping, Data Augmentation, Hyperparameter Tuning — সব কিছু একসাথে। 👉 [নোট পড়ুন](./DL_Performance_Improvement/DL_Performance_Improvement.md)

---

### 🔵 পর্ব ৩: ইমেজ প্রসেসিং (Convolutional Neural Networks - CNN)
*CNN অনেক বড় বিষয়, তাই এটিকেও ৩টি ভাগে ভাগ করা হয়েছে।*

- [x] **১৫. Introduction to CNN & Image Processing:** Why standard ANN fails on images? Concept of Pixels, Channels (RGB), and Image Tensors. 👉 [নোট পড়ুন](./CNN_Introduction/CNN_Introduction.md)
- [x] **১৬. The Convolution Operation:** How Filters/Kernels work, Stride, Padding, and Feature Map generation. 👉 [নোট পড়ুন](./Convolution_Operation/Convolution_Operation.md)
- [x] **১৭. Pooling Layers & Fully Connected Network:** Max Pooling vs Average Pooling intuition, Flattening, and how the final class is predicted. 👉 [নোট পড়ুন](./Pooling_FC_Network/Pooling_FC_Network.md)
- [x] **১৮. Transfer Learning in Keras (CNN):** Pretrained Models (VGG16, ResNet50, MobileNetV2) ব্যবহার করে Feature Extraction ও Fine-Tuning-এর সম্পূর্ণ গাইড। Frozen layers, Catastrophic Forgetting এবং ImageNet weights-এর ব্যাখ্যা। 👉 [নোট পড়ুন](./Transfer_Learning_CNN/Transfer_Learning_CNN.md)

---

### 🟣 পর্ব ৪: টেক্সট এবং সিকোয়েন্স ডেটা (Recurrent Neural Networks - RNN)
*RNN এবং এর বিভিন্ন ভ্যারিয়েন্টগুলো ভালোভাবে বোঝার জন্য ৩টি ভাগে ভাগ করা হয়েছে।*

- [x] **১৮. Introduction to Sequence Data and Standard RNN:** Why CNN/ANN cannot handle sequence data (like text or time)? Concept of Hidden State and Unrolling RNN. 👉 [নোট পড়ুন](./RNN_Introduction/RNN_Introduction.md)
- [x] **১৯. The Vanishing Gradient Problem in Deep Networks:** What is the Vanishing Gradient problem and why does standard RNN fail to remember long sequences? 👉 [নোট পড়ুন](./Vanishing_Gradient_RNN/Vanishing_Gradient_RNN.md)
- [x] **২০. LSTM (Long Short-Term Memory) & GRU:** How LSTM solves the vanishing gradient using Gates (Forget, Input, and Output gates). Brief comparison with GRU. 👉 [নোট পড়ুন](./LSTM_GRU/LSTM_GRU.md)

---

### 🔴 পর্ব ৫: ট্রান্সফরমার এবং আধুনিক এলএলএম (Transformers & LLMs)
*যেহেতু ট্রান্সফরমার আর্কিটেকচার সবচেয়ে জটিল (ChatGPT এর মূল ভিত্তি), তাই একে সুনির্দিষ্ট ৫টি ধাপে ভাগ করা হয়েছে।*

- [x] **২১. Sequence-to-Sequence (Seq2Seq) Models & The Bottleneck Problem:** Traditional Encoder-Decoder architectures and why context vectors struggle with long sentences. 👉 [নোট পড়ুন](./Seq2Seq_Bottleneck/Seq2Seq_Bottleneck.md)
- [x] **২২. The Attention Mechanism (Basics):** What is Attention? How it helps the decoder focus on relevant parts of the input sequence instead of a single context vector. 👉 [নোট পড়ুন](./Attention_Mechanism/Attention_Mechanism.md)
- [x] **২৩. Self-Attention in Transformers:** Query, Key, and Value (Q, K, V) vectors intuition. How words in a sentence relate to each other. 👉 [নোট পড়ুন](./Self_Attention_QKV/Self_Attention_QKV.md)
- [x] **২৪. Multi-Head Attention & Positional Encoding:** Why single attention isn't enough? How Positional Encoding feeds sequence order into models that don't have recurrence. 👉 [নোট পড়ুন](./Multi_Head_Attention_PE/Multi_Head_Attention_PE.md)
- [x] **২৫. The Full Transformer Architecture:** Everything put together (Encoder block, Decoder block, Add & Norm, and Feed Forward networks). 👉 [নোট পড়ুন](./Full_Transformer_Architecture/Full_Transformer_Architecture.md)

---

### 🟠 পর্ব ৬: জেনারেটিভ AI ও RAG সিস্টেম (Generative AI & RAG)

- [x] **২৬. CRAG (Corrective Retrieval-Augmented Generation):** LangGraph দিয়ে তৈরি স্মার্ট RAG সিস্টেম যেটি retrieved document evaluate করে, প্রয়োজনে Web Search করে সঠিক উত্তর দেয়। 👉 [নোট পড়ুন](./CRAG/CRAG.md)

---

### ⚙️ পর্ব ৭: অপ্টিমাইজার ও ট্রেনিং কৌশল (Optimizers & Training Techniques)

- [x] **২৭. Optimizers (SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW):** Optimizer কী এবং কেন দরকার? সব প্রধান Optimizer-এর বিস্তারিত Math, Code, এবং Real-world use cases। 👉 [নোট পড়ুন](./Optimizers/Optimizers.md)

---

*🔄 নিয়মিত আপডেট হচ্ছে | 📅 শেষ আপডেট: ২০২৬-০৪-১১ (Full Transformer Architecture নোট যোগ হয়েছে — Transformer সিরিজ সম্পূর্ণ! 🎉)*
