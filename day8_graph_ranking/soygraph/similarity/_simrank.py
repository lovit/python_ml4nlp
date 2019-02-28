from collections import defaultdict

class SimRank:
    def __init__(self, graph=None, max_iter=10, decaying_factor=0.8,
        min_similarity=0.005, verbose=True):

        self.g = graph
        self.max_iter = max_iter
        self.df = decaying_factor
        self.sim = {}
        self.min_similarity = min_similarity
        self.verbose = verbose

    def train(self, graph=None):
        if graph:
            self.g = graph

        nodes = self.g.nodes()
        
        # normalized by inbound weight sum
        sum_inb_weight = {node:sum([w for _, w in self.g.inbounds(node)])
            for node in nodes}

        for n_iter in range(1, self.max_iter+1):

            sim_ = defaultdict(lambda: defaultdict(lambda: 0))

            for a in nodes:
                print('\rnode = {}'.format(a), end='', flush=True)
                for b in nodes:

                    if a == b:
                        continue

                    inbs_a = self.g.inbounds(a)
                    inbs_b = self.g.inbounds(b)
                    sim_ab = 0

                    for inb_a, wa in inbs_a:
                        for inb_b, wb in inbs_b:
                            if inb_a == inb_b:
                                sim_ab_ = 1.0
                            else:
                                sim_ab_ = self.sim.get(inb_a, {}).get(inb_b, 0)

                            sim_ab += sim_ab_ * wa * wb
                    sim_ab *= (self.df / (sum_inb_weight[a] * sum_inb_weight[b]))

                    # pruning using minimum similarity to prevent out of memory
                    if sim_ab >= self.min_similarity:
                        sim_[a][b] = sim_ab

            self.sim = {node:dict(sim_vec) for node, sim_vec in sim_.items()}

            if self.verbose:
                print('#iter = %d' % n_iter)

        return self.sim

    
class SingleVectorSimRank:
    def __init__(self, graph, max_iter=10, decaying_factor=0.80):
        
        self.g = graph
        self.max_iter = max_iter
        self.df = decaying_factor

        norm = lambda x: sum([w for node, w in x]) if x else 0
        _nodes = graph.nodes()

        self.sum_inb_weights = {node:norm(graph.inbounds(node)) for node in _nodes}
        self.sum_outb_weights = {node:norm(graph.outbounds(node)) for node in _nodes}

    def most_similar(self, q, topk=10, max_iter=-1):

        if max_iter < 0:
            max_iter = self.max_iter

        sims, meets = {q:1}, {q:1}
        topk_sim = 0

        for n_iter in range(max_iter):
            df = pow(self.df, n_iter + 1)
            
            sorted_sim = sorted(sims.items(), key=lambda x:-x[1])
            topk_sim = sorted_sim[:topk][-1][1]

            # remove nodes that cannot be included in top k
#             delta = (topk_sim - sorted_sim[:topk+1][-1][1])
#             if delta > df:
#                 break

            # move forward
            meets_ = {}
            for meet, m_w in meets.items():
                for inb, inb_w in self.g.inbounds(meet):
                    inb_w /= self.sum_inb_weights[meet]
                    meets_[inb] = meets_.get(inb, 0) + m_w * inb_w

            # move backward
            ## initialize
            backward = {}
            for meet, m_w in meets_.items():
                if meet == q:
                    sims = {node:sim * (1 + df * m_w) for node, sim in sims.items()}
                    continue
                for outb, outb_w in self.g.outbounds(meet):
                    if outb in meets:
                        continue
                    outb_w /= self.sum_inb_weights[outb]
                    backward[outb] = backward.get(outb, 0) + m_w * outb_w
            ## iteration
            for n_backstep in range(n_iter):
                backward_ = {}
                for back, b_w in backward.items():
                    for outb, outb_w in self.g.outbounds(back):
                        outb_w /= self.sum_inb_weights[outb]
                        backward_[outb] = backward_.get(outb, 0) + b_w * outb_w
                backward = backward_

            # for debugging
#             print('\niter = {}'.format(n_iter+1))
#             print('meets')
#             pprint(meets_)
#             print('similars')
#             pprint(backward)
            
            # replace meets with meets_
            meets = meets_
            
            # cumulate similarity
            for back, b_w in backward.items():
                sims[back] = sims.get(back, 0) + b_w * df

        del sims[q]

        similars = sorted(sims.items(), key=lambda x:-x[1])
        if topk > 0:
            similars = similars[:topk]

        return similars


class SinglePairSimRank:
    def __init__(self, graph, max_iter=10, decaying_factor=0.85):
        self.g = graph
        self.max_iter = max_iter
        self.df = decaying_factor

    def query(self, q1, q2):
        sim = 0
        # TODO
        return sim
