## Setup

```
conda create -n ie python=3.10
pip install -r requirements.txt
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install soundfile # if you don't have an audio backend already
```

If you have trouble with setting up the environment, please get in touch with the course staff.

## Usage

### Training

```
cd IE_Proj2_2026
python train.py
```

The outline of train.py is as follows:

* Extracts normalized MFCC features for each utterance;
* From data/trn_alignments.json, read acoustic frame-to-letter alignments;
* The model takes normalized MFCC features and predicts a letter for each frame;
* Compute cross entropy loss on the model's predictions for each frame against the acoustic frame-to-letter alignments, and update the model's parameters.

You do not have to edit train.py in any way (apart fron hyper-parameters like learning rate). If you do, please carefully document and explain your changes in your writeup. Do not change the interface to train.py: we will attempt to run train.py without any parameters on Gradescope.

Note: train.py enforces a 20-minute time limit on training. **Please do not change the time limit** as doing so could make your submission timeout on Gradescope. 

### Inference

```
python infer.py
```

infer.py reads best_model.pt and writes the predictions from data/clsp.devwav to output.txt. You can change the checkpointing logic by editing the training loop in train.py.

## To-do Items

### Improve the Model

You may noticed that your neural acoustic model in modules/model.py is in sorry shape, consisting only of a single linear layer mapping MFCC features to the number of letters in our vocabulary. 

* English is not a phonetic language, therefore letters in each word may not have a natural mapping to the exact acoustic frame that they align to. Instead, it might be necessary to take numerous consecutive acoustic frames into account in order to make a successful prediction. What kind of model architecture can take advantage of this property of the English language?
* How can you allow the model to learn non-linear relationships between input features and output labels?
* Our dataset is very small. What could you do to reduce overfitting?

Note that if you change the number of parameters in your model, you should update the batch size/learning rate accordingly in order to maintain the training variance across batches. Explain this in your writeup.

If there are other models that you tried, feel free to include them in your submission with alternative filenames (modules/model_1.py, etc.) and mention them in your writeup.

### Implement a Decoder

You may have improved your model, but you model's accuracy is still 0. You may also be pondering, if how neural model predicts a letter for each acoustic frame, how do we use it to predict a word for each utterance consisting of many frames?

There are a lot more frames than there are letters in the corresponding transcript. We can model that by constructing an HMM for each word, such that it contains one state for each letter in the word, as well as arcs that would allow us either to stay in that state (repeat the previous letter for this acoustic frame), or transition to the next state (move on to the next letter for this acoustic frame). For simplicity's sake, you don't need to implement this HMM explicitly or worry about its transition probabilities: we only need it conceptually.

How do we find the emission probability of our letter HMM, which emits acoustic features given the state (letter)? Now it is time to bring in the predictions y (or more precisely, log probabilities P(y | x(t))) that our neural acoustic model produced for acoustic features x(t) of frame t. In your writeup, explain how you can relate P(y | x(t)) with the emission probability of the letter HMM, and implement your idea in utils/decode.py. Hint: the unigram count of each letter (in number of aligned acoustic frames) can be found in data/trn_token_counts.json.

Once you have utils/decode.py, you can then compute a likelihood for each word in our vocabulary. Try and compare the marginal alpha probability with the Viterbi gamma probability.

Note: avoid looping over the words in your vocabulary (resulting in a time, vocabulary double loop). 

## Submission

Please make sure the following files are contained within your submission:

```
train.py
infer.py
best_model.pt
output.txt
utils/features.py
utils/decode.py
modules/dataset.py
modules/model.py
```
As well as any additional dependencies that you added. You do not have to include your data files in your submission.

## Evaluation Criteria

You will be graded on the following:

* Presence of required files.
* Your dev F1 score as computed from output.txt.
* Your dev and test F1 score as generated from best_model.pt and infer.py.
* The dev and test F1 score from best_model.pt trained using your train.py within a time limit of 20 minutes.
* Your writeup.

The test set is hidden from you. Please do not try to find it online.

**Please avoid using any pre-trained models in your submission.**
