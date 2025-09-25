import numpy as np
import awkward as ak

def add_fin_state_reco(Z1Flav, Z2Flav):
    i, j = np.abs(Z1Flav), np.abs(Z2Flav)

    fin = -ak.ones_like(i)

    fin = ak.where((i == 0) & (j == 0), 0, fin)  # other
    fin = ak.where((i == 121) & (j == 121), 1, fin)  # 4e
    fin = ak.where((i == 169) & (j == 169), 2, fin)  # 4mu
    fin = ak.where(((i == 121) & (j == 169)) | ((i == 169) & (j == 121)), 3, fin)  # 2e2mu
    fin = ak.where(((i == 225) & (j == 169)) | ((i == 169) & (j == 225)), 4, fin)  # 2tau2mu
    fin = ak.where(((i == 225) & (j == 121)) | ((i == 121) & (j == 225)), 5, fin)  # 2tau2e
    fin = ak.where((i == 225) & (j == 225), 6, fin)  # 4tau
    print(fin)
    mapping = ak.Array(["other", "4e", "4mu", "2e2mu", "2tau2mu", "2tau2e", "4tau"])
    fin = mapping[fin]

    return fin

# a = ak.Array([0, 121, 169, 121, 225, 225, 225])
# b = ak.Array([[0, 121, 169, 121, 225, 225, 225],[0, 121, 169, 121, 225, 225, 225]])

# print(add_fin_state_reco(b, b))

def add_fin_state_gen(lepId, Hindex):
    print('Started add_fin_state_gen')
    fin = -ak.ones_like(lepId[:,0])

    lepId = np.abs(lepId)
    Hindex = np.abs(Hindex)

    # Check for invalid indices
    # invalid_mask = (Hindex[:,0] == 99) | (Hindex[:,1] == 99) | (Hindex[:,2] == 99) | (Hindex[:,3] == 99)
    invalid_mask = ak.any(Hindex == 99, axis=1)

    # Replace 99 with a safe index (e.g. 0) so no out-of-bounds
    safe_Hindex = ak.where(Hindex == 99, -1, Hindex)
    print('safe_Hindex:', safe_Hindex)
    print('lepId:', lepId)
    print('Hindex:', Hindex)

    row_idx = ak.local_index(lepId, axis=0)
    print('row_idx:', row_idx)

    lep_0 = lepId[row_idx, safe_Hindex[:,0]]
    lep_2 = lepId[row_idx, safe_Hindex[:,2]]

    # Set final states
    fin = ak.where(invalid_mask, 0, fin)  # other
    fin = ak.where((lep_0 == 11) & (lep_2 == 11) & ~invalid_mask, 1, fin)  # 4e
    fin = ak.where((lep_0 == 13) & (lep_2 == 13) & ~invalid_mask, 2, fin)  # 4mu
    fin = ak.where((((lep_0 == 11) & (lep_2 == 13)) | ((lep_0 == 13) & (lep_2 == 11))) & ~invalid_mask, 3, fin)  # 2e2mu

    mapping = ak.Array(["other", "4e", "4mu", "2e2mu"])
    fin = mapping[fin]

    print('Finished add_fin_state_gen')
    return fin

lepId = ak.Array([[11, -11, -13, 13], [13, 13, 11], [13, -13, -13, 13]])  
Hindex = ak.Array([[0, 1, 2, 3], [1, 0, 2, 99], [0, 2, 1, 3]]) 

print(add_fin_state_gen(lepId, Hindex))

def add_fin_state_gen_out(ZdauId):
    fin = ak.zeros_like(ZdauId[:,0])

    ZdauId_0 = abs(ZdauId[:,0])
    ZdauId_1 = abs(ZdauId[:,1])

    print('ZdauId_0:',ZdauId_0)
    print('ZdauId_1:',ZdauId_1)

    # fin = ak.where((((abs(ZdauId_0)!=11) | (abs(ZdauId_1)!=13)) | ((abs(ZdauId_0)!=13) | (abs(ZdauId_1)!=11))), 0, fin)  # other
    fin = ak.where((abs(ZdauId_0)==11) & (abs(ZdauId_1)==11), 1, fin)  # 4e
    fin = ak.where((abs(ZdauId_0)==13) & (abs(ZdauId_1)==13), 2, fin)  # 4mu
    fin = ak.where((((abs(ZdauId_0)==11) & (abs(ZdauId_1)==13)) | ((abs(ZdauId_0)==13) & (abs(ZdauId_1)==11))), 3, fin)  # 2e2mu

    mapping = ak.Array(["other", "4e", "4mu", "2e2mu"])
    fin = mapping[fin]

    return fin

# ZdauId = ak.Array([[11, 11], [13, 11], [13, 13], [99, 11], [99, 99]])
# print(ZdauId)
# print(add_fin_state_gen_out(ZdauId))

def add_fin_state_gen_out_ZH(ZdauId,momId):
    fin = ak.zeros_like(ZdauId[:,0])

    ZdauId_0 = np.abs(ZdauId[:,0])
    ZdauId_1 = np.abs(ZdauId[:,1])
    ZdauId_2 = np.abs(ZdauId[:,2])

    momId_0 = momId[:,0]
    momId_1 = momId[:,1]
    momId_2 = momId[:,2]
    print('momId_0:',momId_0)
    print('momId_1:',momId_1)
    print('momId_2:',momId_2)

    fin = ak.where(((momId_0==25) & (momId_1==25) & (ZdauId_0==11) & (ZdauId_1==11)) |
                   ((momId_0==25) & (momId_2==25) & (ZdauId_0==11) & (ZdauId_2==11)) |
                   ((momId_1==25) & (momId_2==25) & (ZdauId_1==11) & (ZdauId_2==11)), 1, fin)  # 4e

    fin = ak.where(((momId_0==25) & (momId_1==25) & (ZdauId_0==13) & (ZdauId_1==13)) |
                   ((momId_0==25) & (momId_2==25) & (ZdauId_0==13) & (ZdauId_2==13)) |
                   ((momId_1==25) & (momId_2==25) & (ZdauId_1==13) & (ZdauId_2==13)), 2, fin)  # 4mu
 
    fin = ak.where(((momId_0==25) & (momId_1==25) & (ZdauId_0!=ZdauId_1) & ((ZdauId_0==11) | (ZdauId_0==13))) & (((ZdauId_1==11) | (ZdauId_1==13))) |
                   ((momId_0==25) & (momId_2==25) & (ZdauId_0!=ZdauId_2) & ((ZdauId_0==11) | (ZdauId_0==13))) & (((ZdauId_2==11) | (ZdauId_2==13))) |
                   ((momId_1==25) & (momId_2==25) & (ZdauId_1!=ZdauId_2) & ((ZdauId_1==11) | (ZdauId_1==13))) & (((ZdauId_2==11) | (ZdauId_2==13))), 3, fin) # 2e2mu

    mapping = ak.Array(["other", "4e", "4mu", "2e2mu"])
    fin = mapping[fin]

    return fin

# ZdauId = ak.Array([[11, 11, 13], [13, 11, 13], [13, 11, -13], [99, 11, 99], [13, 11, 13]])
# momId = ak.Array([[25, 25, 23], [25, 99, 23], [25, 23, 25], [23, 25, 99], [23, 25, 25]])
# print(momId)
# print(add_fin_state_gen_out_ZH(ZdauId, momId))

def add_cuth4l_gen(momMomId,Hindex):
    output = ak.zeros_like(momMomId[:,0])

    invalid_mask = ak.any(Hindex == 99, axis=1)  
    safe_Hindex = ak.where(Hindex == 99, -1, Hindex)

    momMomId_0 = momMomId[np.arange(len(momMomId)), safe_Hindex[:,0]]
    momMomId_1 = momMomId[np.arange(len(momMomId)), safe_Hindex[:,1]]
    momMomId_2 = momMomId[np.arange(len(momMomId)), safe_Hindex[:,2]]
    momMomId_3 = momMomId[np.arange(len(momMomId)), safe_Hindex[:,3]] 

    print('momMomId_0:', momMomId_0)
    print('momMomId_1:', momMomId_1)    
    print('momMomId_2:', momMomId_2)
    print('momMomId_3:', momMomId_3)

    output = ak.where((momMomId_0 == 25) & (momMomId_1 == 25) & (momMomId_2 == 25) & (momMomId_3 == 25) & (~invalid_mask), 1, output)

    mapping = ak.Array([False, True])
    output = mapping[output]

    return output

# momMomId = ak.Array([[25, 25, 25, 25], [25, 99, 23, 23], [25, 25, 25, 25], [23, 25, 99, 23], [99, 99, 99, 99]])
# Hindex = ak.Array([[0, 1, 2, 3], [1, 2, 3, 0], [0, 3, 2, 1], [2, 1, 0, 3], [99, 99, 99, 99]])

# print(add_cuth4l_gen(momMomId, Hindex))

def add_cuth4l_reco(Hindex, genIndex, momMomId, momId):
    invalid_mask = ak.any((Hindex == 99) | (Hindex == -1), axis=1) | ak.any(genIndex < 0, axis=1)
    safe_Hindex = ak.where((Hindex == 99) | (Hindex == -1), 0, Hindex)
    
    row_idx = ak.local_index(genIndex, axis=0)
    
    genIdx_0 = genIndex[row_idx, safe_Hindex[:,0]]
    genIdx_1 = genIndex[row_idx, safe_Hindex[:,1]]
    genIdx_2 = genIndex[row_idx, safe_Hindex[:,2]]
    genIdx_3 = genIndex[row_idx, safe_Hindex[:,3]]

    safe_genIdx_0 = ak.where((genIdx_0 < 0) | (genIdx_0 == 99), 0, genIdx_0)
    safe_genIdx_1 = ak.where((genIdx_1 < 0) | (genIdx_1 == 99), 0, genIdx_1)
    safe_genIdx_2 = ak.where((genIdx_2 < 0) | (genIdx_2 == 99), 0, genIdx_2)
    safe_genIdx_3 = ak.where((genIdx_3 < 0) | (genIdx_3 == 99), 0, genIdx_3)

    momMom_0 = momMomId[row_idx, safe_genIdx_0]
    momMom_1 = momMomId[row_idx, safe_genIdx_1]
    momMom_2 = momMomId[row_idx, safe_genIdx_2]
    momMom_3 = momMomId[row_idx, safe_genIdx_3]
    
    mom_0 = momId[row_idx, safe_genIdx_0]
    mom_1 = momId[row_idx, safe_genIdx_1]
    mom_2 = momId[row_idx, safe_genIdx_2]
    mom_3 = momId[row_idx, safe_genIdx_3]
    
    all_valid = ((genIdx_0 > -0.5) & (momMom_0 == 25) & (mom_0 == 23) &
                 (genIdx_1 > -0.5) & (momMom_1 == 25) & (mom_1 == 23) &
                 (genIdx_2 > -0.5) & (momMom_2 == 25) & (mom_2 == 23) &
                 (genIdx_3 > -0.5) & (momMom_3 == 25) & (mom_3 == 23) &
                 (~invalid_mask))

    return all_valid

# Hindex = ak.Array([[0, 1, 2, 3], [1, 2, 3, 0], [0, 3, 2, 1], [2, 1, 0, 3], [99, 99, 99, 99]])
# genIndex = ak.Array([[0, 1, 2, 3], [1, 2, 3, 0], [0, 3, 2, 1], [2, 1, 0, 3], [99, 99, 99, 99]])
# momMomId = ak.Array([[25, 25, 25, 25], [25, 99, 23, 23], [25, 25, 25, 25], [23, 25, 99, 23], [99, 99, 99, 99]])
# momId = ak.Array([[23, 23, 23, 23], [23, 99, 23, 23], [23, 23, 23, 23], [23, 23, 99, 23], [99, 99, 99, 99]])

# print(add_cuth4l_reco(Hindex, genIndex, momMomId, momId))
