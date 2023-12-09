import torch
class Voter(object):
    def __init__(self):
        pass

    def vote(self, candidates, type='vector'):
        """
        :param candidates: list of candidates
        :param type: 'vector' or 'label', 'vector' means voting the center candidates, 'label' means voting the label
        :return: the winner
        """
        if type=='vector':
            all_different = False
            winner = None

            t = torch.tensor(candidates)
            winner = t.mode(0).values
            return winner
        else:
            # compare the candidate labels
            all_different = False
            winner = None

            # find the most frequent label
            labels = [candidate[2] for candidate in candidates]
            winner = max(set(labels), key=labels.count)
            return winner

