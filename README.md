# NLP-Sequence-Labeling-CRFsuite

## Description:
Assign dialogue acts to sequences of utterances in conversations from a corpus. Optimize the tags assigned to the sequence as a whole rather than treating each tag decision separately.Use conditional random fields toolkit, CRFsuite.

The raw data for each utterance in the conversation consists of the speaker name, the tokens and their part of speech tags. Given a labeled set of data, we first create a baseline set of features as specified below, and measure the accuracy of the CRFsuite model created using those features. Then we experiment with additional features in an attempt to improve performance. Compare accuracy from both. Programs assign dialogue act tags to unlabeled data.

### Data:
The Switchboard (SWBD) corpus was collected from volunteers and consists of two person
telephone conversations about predetermined topics such as child care. SWBD DAMSL refers to a set of dialogue act annotations made to this data. 

- act_tag - the dialogue act associated with this utterance. Note, this will be blank for the unlabeled test data we use to test your code.
- speaker - the speaker of the utterance (A or B).
- pos - a whitespace-separated list where each item is a token, "/", and a part of speech tag (e.g., "What/WP are/VBP your/PRP$ favorite/JJ programs/NNS ?/."). When the utterance has no words(e.g., the transcriber is describing some kind of noise), the pos column may be blank, consist solely of "./.", have a pos but no token, or have an invented token such as MUMBLEx. You can view the text column to see the original transcription.
- text - The transcript of the utterance with some cleanup but mostly unprocessed and untokenized. This column may or may not be a useful source of features when the utterance solely consists of some kind of noise.

### CRFsuite
Install pycrfsuite (https://pypi.python.org/pypi/python-crfsuite), a Python interface
to CRFsuite (http://www.chokkan.org/software/crfsuite/). As discussed in the CRFsuite tutorial, and the pycrfsuite tutorial, you add training data to the Trainer object using the append method which takes two arguments (feature_vector_list,label_list) and loads the training data for a single sequence. In our case, each sequence corresponds to a dialogue, and the feature_vector_list is a list of feature vectors (one for each utterance in the dialogue). The label_list corresponds to the dialogue acts for those utterances. Each feature vector is a list of individual features which are binary. The presence of a feature indicates that it is true for this item. Absence indicates that the feature would be false. Here are the features for a training example using features for whether a
particular token is present or not in an utterance.

['TOKEN_i', 'TOKEN_certainly', 'TOKEN_do', 'TOKEN_.']

After loading the training data, we set the CRFsuite training parameters. The following
parameters are taken from the pycrfsuite tutorial. 

 trainer.set_params({
 'c1': 1.0, # coefficient for L1 penalty
 'c2': 1e-3, # coefficient for L2 penalty
 'max_iterations': 50, # stop earlier
 # include transitions that are possible, but not observed
 'feature.possible_transitions': True
 })

The last step is to train the model using the train method which takes a single argument, the name of the file in which to save the model. The two programs both train models and use them to tag data. Thus, you can give the model whatever name you like because you will be creating a Tagger object and using its open method to read the model. The tag method of the Tagger object processes a single sequence at a time (i.e., one dialogue) represented as a list of feature value vectors (i.e., one per utterance) in the same format
used by the Trainer object. The tagger will output a list of labels (i.e., dialogue acts): one per utterance. 

### Baseline Features:
• a feature for whether or not the speaker has changed in comparison with the previous
utterance.
• a feature marking the first utterance of the dialogue.
• a feature for every token in the utterance (see the description of CRFsuite for an
example).
• a feature for every part of speech tag in the utterance (e.g., POS_PRP POS_RB, POS_VBP POS_.).

>python3 baseline_tagger.py INPUTDIR TESTDIR OUTPUTFILE

### Advanced Features:
- Uppercase: check if feature isupper(), and add a feature if true
- Title: check if feature istitle(), and add a feature if true
- Digit: check if feature isdigit(), and add a feature if true
- Check (Uppercase, Title, Digit) features for previous utterance. IF it exists then add to feature list. IF previous doesn't exist, add tag 'BOS'.
- Check (Uppercase, Title, Digit) features for next utterance. IF it exists then add to feature list. IF next doesn't exist, add tag 'EOS'.

>python3 advanced_tagger.py INPUTDIR TESTDIR OUTPUTFILE

#### Baseline Accuracy: 72.26%
#### Advanced Accuracy: 73.35%
