class TrainedHMM:
    """Usage
    --------
    transition = {
        ('Noun', 'Josa'): 0.7,
        ('Noun', 'Noun'): 0.3,
        ('Verb', 'Eomi'): 0.5,
        ('Verb', 'Noun'): 0.5,
        ('Verb', 'Josa'): -0.1,
    }
    generation = {
        'Noun': {
            '아이오아이': 0.5,
            '청하': 0.2,
        }
    }

    hmm_model = TrainedHMM(transition, generation)
    hmm_model.cost(('아이오아이', 'Noun', 0, 5), ('는', 'Josa', 5, 6))
    """
    def __init__(self, transition, generation,
        transition_smoothing=0.0001, generation_smoothing=0.0001):

        self.transition_smoothing = transition_smoothing
        self.generation_smoothing = generation_smoothing
        self.transition = transition
        self.generation = generation

    def __call__(self, edge):
        return self.cost(edge)

    def cost(self, edge):
        prob = 0
        prob += (self.transition.get(
                     (edge[0][1], edge[1][1]), 0)
                 + self.transition_smoothing)
        for node in edge:
            prob += (self.generation.get(
                          node[1], {}).get(node[0], 0)
                      + self.generation_smoothing)
        return -1 * prob