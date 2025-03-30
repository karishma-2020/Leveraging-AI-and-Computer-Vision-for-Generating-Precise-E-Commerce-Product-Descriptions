# Leveraging-AI-and-Computer-Vision-for-Generating-Precise-E-Commerce-Product-Descriptions


# **Model Development**  

## **1. Overview**  
The model consists of two main components:  
- **Encoder:** Convolutional Neural Network (CNN)  
- **Decoder:** Recurrent Neural Network (RNN)  

The **encoder** extracts high-level features from product images, while the **decoder** combines the extracted visual context with word embeddings of true captions to iteratively predict the next sequence of words.  

---

## **2. Encoder**  
The **EncoderCNN** extracts meaningful visual features using a **pre-trained ResNet50 model**. The extracted features are reduced to an **embedding size of 256** for further processing.  

### **Key Features of Encoder:**  
- **Feature Extraction:** Uses ResNet50 to extract high-level visual features.  
- **Fully Connected Layer Modification:** Replaces the original fully connected layer with a **linear layer**, maintaining input size while modifying the output to **dense feature embeddings**.  
- **Activation Function:** Uses **ReLU** for non-linearity.  
- **Regularization:** Implements **Dropout (50%)** to prevent overfitting.  

### **Input & Output:**  
- **Input:** A batch of images as tensors.  
- **Output:** A tensor of shape **(batch_size, embed_size)** representing encoded image features.  

---

## **3. Decoder**  
The **Decoder (RNN)** generates captions by processing the image features extracted by the encoder. It utilizes **LSTM (Long Short-Term Memory)** to generate meaningful textual descriptions.  

### **Key Features of Decoder:**  
- **Word Embeddings:** Converts input word indices (captions) into dense embeddings of size **embed_size**.  
- **LSTM Parameters:**  
  - **embed_size:** Word embedding size.  
  - **hidden_size:** Dimensionality of the LSTM’s hidden state.  
  - **num_layers:** Number of stacked LSTM layers for learning complex patterns.  
- **Linear Layer:** Maps LSTM’s hidden state to the vocabulary space.  
- **Regularization:** Dropout is applied to the embedding layer.  
- **Optimizer:** **Adam** optimizer is used for training.  

### **Decoder Process:**  
1. **Embedding Captions:** Converts input captions into embeddings.  
2. **Concatenating Features:** Appends image features from the encoder at the start of the caption sequence.  
3. **LSTM Processing:** Feeds the concatenated sequence into the LSTM network.  
4. **Linear Projection:** The LSTM’s output is passed through a linear layer to generate vocabulary-sized logits.  
5. **Output:** A tensor of shape **(sequence_length + 1, batch_size, vocab_size)** containing predicted word distributions.  

---

## **4. Results**  
To enhance model accuracy and evaluation metrics, we experimented with different feature extraction methods, including:  
- **VGG16**  
- **VGG19**  
- **ResNet** (used with LSTM for caption generation)  

### **Evaluation Metrics:**  
- **BERT Score:** Measures semantic similarity between predicted and reference captions.  
- **BLEU Score:** Evaluates **n-gram overlap** between predicted and reference captions.  
  - A **BLEU score of 0.3+** is considered decent and indicates high-quality captions.  


