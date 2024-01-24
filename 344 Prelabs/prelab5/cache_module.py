from cachesim import CacheSimulator, Cache, MainMemory

import numpy

class cache_module:
    def __init__(self, l1_, l2_, l3_, m_):
        mem = MainMemory()
        l3 = Cache(l3_[0], l3_[1], l3_[2], l3_[3], l3_[4])
        mem.load_to(l3)
        mem.store_from(l3)
        l2 = Cache(l2_[0], l2_[1], l2_[2], l2_[3], l2_[4], store_to=l3, 
load_from=l3)
        l1 = Cache(l1_[0], l1_[1], l1_[2], l1_[3], l1_[4], store_to=l2, 
load_from=l2)
        self.cs = CacheSimulator(l1, mem)
        self.memory = numpy.zeros([m_], dtype=numpy.uint8)

    def read(self, index):
        self.cs.load(index)
        return self.memory[index]

    def write(self, index, value):
        self.memory[index] = value
        self.cs.store(index)

    def finish(self):
        self.cs.force_write_back()
        self.cs.print_stats()

