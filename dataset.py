from torchaudio.datasets import SPEECHCOMMANDS
import os


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]



def get_data():
    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")

    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    # plt.plot(waveform.t().numpy())
    # plt.show()
    labels = sorted(set([datapoint[2] for datapoint in train_set]))
    # print(labels)

    return train_set, test_set, labels, sample_rate
