import csv
import glob
import sys
import os
import helper_file as tool
import pycrfsuite
import timeit

# ---------------------------------------------------
input_dir = sys.argv[-3]
output_dir = sys.argv[-2]
output_filenm = sys.argv[-1]

# ---------------------------------------------------


class Process:

    def __init__(self):

        self.prv_uter = ""
        self.first_uter = False

        self.xtrain = []
        self.y_pred = []
        self.X_test = []
        self.y_test = []

    # ---------------------------------------

    def read(self, path):

        data = []

        file_names = sorted(glob.glob(os.path.join(path, "*.csv")))

        for dialogs in file_names:
            data.append("#####")
            data.append(tool.get_utterances_from_filename(dialogs))

        return data

    # ---------------------------------------

    def make_feature(self, file):

        feature_list, label_list = [], []

        for utterance in file:

            features = []

            label_list.append(utterance[0])

            if not self.first_uter and utterance[1] != self.prv_uter and self.prv_uter != "":
                features.append('SPEAKER_CHANGED')

            self.prv_uter = utterance[1]

            if self.first_uter:
                features.append('FIRST_UTTERANCE')
                self.first_uter = False

            if utterance[2]:
                for idx, postag in enumerate(utterance[2]):
                    features.append('TOKEN_' + postag[0])
                    features.append('POS_' + postag[1])

                    adv = self.adv_features(idx, postag, utterance[2])
                    features.extend(adv)

            else:
                features.append('NO_WORD')

            feature_list.append(features)

        return feature_list, label_list

    # -------------------------------------------

    def adv_features(self, idx, postag, utterance):

        adv = []

        if postag[0].isupper():
            adv.append('IS_UPPER')
        if postag[0].istitle():
            adv.append('IS_TITLE')
        if postag[0].isdigit():
            adv.append('IS_DIGIT')

        if idx > 0:
            postag_prv = utterance[idx - 1]
            adv.append('TOKEN_PRV_' + postag_prv[0])
            adv.append('POS_PRV_' + postag_prv[1])

            if postag_prv[0].isupper():
                adv.append('IS_UPPER_PRV')
            if postag_prv[0].istitle():
                adv.append('IS_TITLE_PRV')
            if postag_prv[0].isdigit():
                adv.append('IS_DIGIT_PRV')
        else:
            adv.append('BOS')

        if idx < len(utterance) - 1:
            postag_nxt = utterance[idx + 1]
            adv.append('TOKEN_NXT_' + postag_nxt[0])
            adv.append('POS_NXT_' + postag_nxt[1])

            if postag_nxt[0].isupper():
                adv.append('IS_UPPER_NXT')
            if postag_nxt[0].istitle():
                adv.append('IS_TITLE_NXT')
            if postag_nxt[0].isdigit():
                adv.append('IS_DIGIT_NXT')
        else:
            adv.append('EOS')

        return adv

    # -------------------------------------------

    def train_dialogs(self):

        # Read Data
        global input_dir
        self.xtrain = self.read(input_dir)

        trainer = pycrfsuite.Trainer(verbose=False)

        trainer.set_params({
            'c1': 1.0,
            'c2': 1e-3,
            'max_iterations': 50,
            'feature.possible_transitions': True
        })

        for file in self.xtrain:
            xtrain, ytrain = [], []

            if file == "#####":
                self.first_uter = True
                continue

            features, labels = self.make_feature(file)

            xtrain.extend(features)
            ytrain.extend(labels)

            trainer.append(xtrain, ytrain)

        trainer.train('advanced_data.crfsuite')

    # -------------------------------------------

    def tag_dialogs(self):

        # Read Data
        global output_dir
        self.X_test = self.read(output_dir)

        tagger = pycrfsuite.Tagger()
        tagger.open('advanced_data.crfsuite')

        # Re-init
        self.first_uter = False
        self.prv_uter = ""

        for file in self.X_test:
            xtest, ytest = [], []

            if file == "#####":
                self.first_uter = True
                continue

            xtest, ytest = self.make_feature(file)

            self.y_pred += tagger.tag(xtest)
            self.y_pred.append('eof')

            self.y_test += ytest
            self.y_test.append('eof')

    # ---------------------------------------

    def write(self):

        global output_filenm

        # Measure Accuracy
        count = 0
        for i in range(len(self.y_test)):
            if self.y_test[i] == self.y_pred[i]:
                count += 1

        print("Accuracy: {0:0.2%}".format(count / len(self.y_test)))

        with open(output_filenm, 'w') as f:
            for pred in self.y_pred:
                if pred == 'eof':
                    f.write("\n")
                else:
                    f.write(pred + "\n")


# -------------------------------------------


if __name__ == '__main__':
    # start_time = timeit.default_timer()

    obj = Process()
    obj.train_dialogs()
    obj.tag_dialogs()
    obj.write()

    # print("Time Taken: ", timeit.default_timer() - start_time)
