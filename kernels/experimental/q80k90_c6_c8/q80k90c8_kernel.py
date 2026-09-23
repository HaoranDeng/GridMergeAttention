from q80k90_kernel import Q80K90, candidate_table
class Q80K90C8(Q80K90):
    def __init__(self, **config):
        super().__init__(capacity=8, **config)
