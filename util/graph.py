#-*-coding:utf-8-*-
import collections

def exclusive_combine(*in_list):
    res = set()
    in_list = list(*in_list)
    for n_l in in_list:
        for i in n_l:
            res.add(i)
    return list(res)

class SkillGraph(object):
    def __init__(self, path, edgelist=True):
        self.neighbor_dict = {}
        if edgelist:
            fin = open(path,'r')
            for l in fin.readlines():
                e = l.split("\t")
                i,j = int(e[0]), int(e[1])
                self.update_edge(i,j)
                self.update_edge(j,i)
            fin.close()

        for key in self.neighbor_dict.keys():
            self.neighbor_dict[key] = list(self.neighbor_dict[key])

        self.node_list = list(self.neighbor_dict.keys())
        self.node_list.sort()
        self.node_num = len(self.node_list)


    def update_edge(self, i, j):
        if i in self.neighbor_dict:
            self.neighbor_dict[i].add(j)
        else:
            self.neighbor_dict[i] = {j}

        if j in self.neighbor_dict:
            self.neighbor_dict[j].add(i)
        else:
            self.neighbor_dict[j] = {i}

    def get_neighbors(self, in_list):
        neighbors = [self.neighbor_dict[i] for i in in_list]
        return exclusive_combine(neighbors)

    def get_commonneighborids(self, in_list):
        neighb_count = []
        common_neighb = []
        for i in in_list:
            if i in self.neighbor_dict.keys():
                neighbors = [self.neighbor_dict[i]]
                neighb_count.extend(exclusive_combine(neighbors))

        neighbskill = collections.Counter()
        for neib in neighb_count:
            neighbskill.update([neib])

        for k,v in neighbskill.items():
            if v > 1:
                common_neighb.append(k)
        return common_neighb

    def get_comneibwords(self, word_list):
        node_dict = {}
        id_dict={}
        fNode = open('/home/leating/datasets/ACL_all_under/graph_data/node_100.txt','r')
        nodedata = fNode.readlines()
        fNode.close()
        for l in nodedata[1:]:
            llist = l.strip().split('\t')
            id, word = llist[0],llist[1]
            node_dict[id] = word
            id_dict[word] = id

        in_list = []
        for word in word_list:
            in_list.append(int(id_dict.get(word)))

        comneibids = self.get_commonneighborids(in_list)
        comneibwords = []
        for id in comneibids:
            word = node_dict.get(str(id))
            comneibwords.append(word)
        return comneibwords

    def skillnet_vocab(self):
        id2word_dict = {}
        word2id_dict = {}
        fNode = open('/home/leating/datasets/ACL_all_under/graph_data/node_100.txt', 'r')
        nodedata = fNode.readlines()
        fNode.close()
        for l in nodedata[1:]:
            llist = l.strip().split('\t')
            id, word = llist[0], llist[1]
            id2word_dict[id] = word
            word2id_dict[word] = id
        return id2word_dict, word2id_dict

    def cluster_skill(self, in_list):
        neighbors = self.get_commonneighborids(in_list)
        print(len(neighbors))
        for neighbor in neighbors:
            in_list = [neighbor]
            neiblist = self.get_neighbors(in_list)
            new_list = []
            for n in neiblist:
                if n in neighbors:
                    new_list.append(n)
            print(len(new_list))


if __name__ == '__main__':
    path = '../data/edge.txt'
    graph = SkillGraph(path)
    # graph.get_commonneighbors([5,474])
    # graph.get_comneibwords(['空间设计师','1-49'])
    #graph.skillnet_vocab()
    graph.cluster_skill([5,474])