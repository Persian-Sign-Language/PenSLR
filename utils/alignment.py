import numpy as np
import pprint

MATCH = 3
MISMATCH = -1
GAP = -2

def global_align(x, y, s_match=MATCH, s_mismatch=MISMATCH, s_gap=GAP):
    A = []
    for i in range(len(y) + 1):
        A.append([0] * (len(x) + 1))
    for i in range(len(y) + 1):
        A[i][0] = s_gap * i
    for i in range(len(x) + 1):
        A[0][i] = s_gap * i
    for i in range(1, len(y) + 1):
        for j in range(1, len(x) + 1):
            A[i][j] = max(
                A[i][j - 1] + s_gap,
                A[i - 1][j] + s_gap,
                A[i - 1][j - 1] + (s_match if (y[i - 1] == x[j - 1] and y[i - 1] != '-') else 0) + (
                    s_mismatch if (y[i - 1] != x[j - 1] and y[i - 1] != '-' and x[j - 1] != '-') else 0) + (
                    s_gap if (y[i - 1] == '-' or x[j - 1] == '-') else 0)
            )
    align_X = ""
    align_Y = ""
    i = len(x)
    j = len(y)
    while i > 0 or j > 0:
        current_score = A[j][i]
        if i > 0 and j > 0 and (
                ((x[i - 1] == y[j - 1] and y[j - 1] != '-') and current_score == A[j - 1][i - 1] + s_match) or
                ((y[j - 1] != x[i - 1] and y[j - 1] != '-' and x[i - 1] != '-') and current_score == A[j - 1][
                    i - 1] + s_mismatch) or
                ((y[j - 1] == '-' or x[i - 1] == '-') and current_score == A[j - 1][i - 1] + s_gap)
        ):
            align_X = x[i - 1] + align_X
            align_Y = y[j - 1] + align_Y
            i = i - 1
            j = j - 1
        elif i > 0 and (current_score == A[j][i - 1] + s_gap):
            align_X = x[i - 1] + align_X
            align_Y = "-" + align_Y
            i = i - 1
        else:
            align_X = "-" + align_X
            align_Y = y[j - 1] + align_Y
            j = j - 1
    return (align_X, align_Y, A[len(y)][len(x)], 1)

def get_input():
    n = int(input())
    seq_list = []
    for i in range(n):
        seq_list.append(input())
    return seq_list

class StarAlignment:
    def __init__(self, seq_list, s_match=MATCH, s_mismatch=MISMATCH, s_gap=GAP):
        self.seq_list = seq_list
        self.s_match = s_match
        self.s_mismatch = s_mismatch
        self.s_gap = s_gap
        self.score_matrix = self.create_score_matrix()
        self.center = ""
        self.index_center = 0
    
    def align(self, i, j):
        return global_align(self.seq_list[i], self.seq_list[j], self.s_match, self.s_mismatch, self.s_gap)
    
    def create_score_matrix(self):
        score_matrix = {}
        for i in range(len(self.seq_list)):
            score_matrix[self.seq_list[i]] = [0, {}]
            for j in range(len(self.seq_list)):
                if i != j:
                    ans = self.align(j, i)
                    score_matrix[self.seq_list[i]][1][self.seq_list[j]] = ans
                    score_matrix[self.seq_list[i]][0] += ans[2]
        return score_matrix
    
    def choose_center(self):
        max_score = self.score_matrix[self.seq_list[0]][0]
        center = self.seq_list[0]
        for i in range(1, len(self.seq_list)):
            if self.score_matrix[self.seq_list[i]][0] > max_score:
                max_score = self.score_matrix[self.seq_list[i]][0]
                center = self.seq_list[i]
                self.index_center = i
        self.center = center
        for i in range (len(self.seq_list)):
            if self.seq_list[i] == center:
                self.index_center = i
        
    def calculate_multi_seq_alignment(self, aligned_list):
        score = 0
        for i in range(len(aligned_list[0])):
            for j in range(len(aligned_list)):
                for k in range(j + 1, len(aligned_list)):
                    if aligned_list[j][i] == '-' and aligned_list[k][i] == '-':
                        pass
                    elif aligned_list[j][i] == '-' or aligned_list[k][i] == '-':
                        score += self.s_gap
                    elif aligned_list[j][i] != aligned_list[k][i]:
                        score += self.s_mismatch
                    else:
                        score += self.s_match
        return score
    
    def update_center(self, seq1, seq2, original_center):
        tmp = ""
        L = max(self.find_(seq1, original_center[0], True), self.find_(seq2, original_center[0], True))
        R = max(self.find_(seq1, original_center[-1], False), self.find_(seq2, original_center[-1], False))
        i_s = 0
        while(seq1[i_s] == '-'):
            i_s += 1
        j_s = 0
        while(seq2[j_s] == '-'):
            j_s += 1
        i_e = len(seq1) - 1
        while(seq1[i_e] == '-'):
            i_e -= 1
        j_e = len(seq2) - 1
        while(seq2[j_e] == '-'):
            j_e -= 1
        while(i_s <= i_e and j_s <= j_e):
            if seq1[i_s] == seq2[j_s]:
                tmp += seq1[i_s]
                i_s += 1
                j_s += 1
            elif seq1[i_s] == '-':
                tmp += seq1[i_s]
                i_s += 1
            elif seq2[j_s] == '-':
                tmp += seq2[j_s]
                j_s += 1
        return L * "-" + tmp + R * "-"
    
# 1: ABC
# 2: ABBC
# 3: AC

# 1:2 : AB-C
# 1:3: ABC
    
# 1:2:3: AB-C

    def find_(self, seq, c, L=True):
        if L:
            i = 0
            for k in range(len(seq)):
                if seq[k] == c:
                    i = k
                    return i
        else:
            i = len(seq) - 1
            for k in range(len(seq) - 1, -1, -1):
                if seq[k] == c:
                    i = k
                    return len(seq) - 1 - i
    
    def update_seq_by_center(self, new_center, last_center, original_center, last_seq, original_seq):
        if new_center == last_center:
            return last_seq
        else:
            L_new = self.find_(new_center, original_center[0], True)
            L_old = self.find_(last_center, original_center[0], True)
            R_new = self.find_(new_center, original_center[-1], False)
            R_old = self.find_(last_center, original_center[-1], False)
            L = L_new - L_old
            R = R_new - R_old
            i_s = 0
            if R != 0:
                tmp_new_center = new_center[L:-R]
            else:
                tmp_new_center = new_center[L:]
            tmp = ""
            i = 0
            j = 0
            while(i < len(tmp_new_center) and j < len(last_center)):
                if tmp_new_center[i] == last_center[j]:
                    tmp += last_seq[j]
                    i += 1
                    j += 1
                elif tmp_new_center[i] == '-':
                    tmp += '-'
                    i += 1
                elif original_seq[j] == '-':
                    tmp += '-'
                    j += 1
            return L * '-' + tmp + R * '-'

    def align_together(self):
        centers = self.score_matrix[self.center][1]
        Cs = []
        for i in range(len(self.seq_list)):
            if self.seq_list[i] in centers:
                Cs.append(centers[self.seq_list[i]][1])
        center = Cs[0]
        for i in range(1, len(Cs)):
            center = self.update_center(center, Cs[i], self.center)
        aligned_list = []
        for i in range(len(self.seq_list)):
            if self.seq_list[i] == self.center:
                aligned_list.append(center)
            else:
                tmp = self.update_seq_by_center(center, centers[self.seq_list[i]][1], self.center, centers[self.seq_list[i]][0], self.seq_list[i])
                aligned_list.append(tmp)
        return aligned_list
    
    def show_alignment(self, aligned_list):
        for i in range(len(aligned_list)):
            print(aligned_list[i])

def perform_star_alignment(input_sequences: list, match=0, mismatch=-0, gap=0):
    SA = StarAlignment(input_sequences, s_match=match, s_mismatch=mismatch, s_gap=gap)
    SA.choose_center()
    result = SA.align_together()
    # print(np.array(list(map(lambda s: list(s), result))))
    return SA.calculate_multi_seq_alignment(result), np.array(list(map(lambda s: list(s), result)))
    # print(SA.calculate_multi_seq_alignment())
    # SA.show_alignment(SA.align_together())

def series_alignment(input_sequences: list, match=0, mismatch=0, gap=0):
    SA = StarAlignment(input_sequences, s_match=match, s_mismatch=mismatch, s_gap=gap)
    SA.choose_center()
    result = SA.align_together()
    # print(np.array(list(map(lambda s: list(s), result))))
    return SA.calculate_multi_seq_alignment(result), np.array(list(map(lambda s: list(s), result)))
    # print(SA.calculate_multi_seq_alignment())
    # SA.show_alignment(SA.align_together())


if __name__ == '__main__':
    import time
    arr = get_input()
    tic = time.time()
    
    results = perform_star_alignment(arr)
    print(time.time() - tic)
    print(results)






# SA = StarAlignment(get_input())
# print (SA.choose_center())
# print (SA.order_to_align(center=SA.choose_center()))
