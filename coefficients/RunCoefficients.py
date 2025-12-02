import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
# import pandas as pd
import awkward as ak 
import uproot #3 as uproot
from math import sqrt, log
import sys,os
import optparse
import itertools
# import math
import ROOT
import json

import cProfile
import pstats
from collections import defaultdict

sys.path.append('../helperstuff/')

from observables import observables
from binning import binning
from paths import path

import time 

start = time.time()

print('Welcome in RunCoefficients!')

def parseOptions():

    global opt, args, runAllSteps

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)

    # input options
    parser.add_option('',   '--obsName',  dest='OBSNAME',  type='string',default='costhetaZ1',   help='Name of the observable, supported: "inclusive", "pT4l", "eta4l", "massZ2", "nJets"')#pT4l
    parser.add_option('',   '--obsBins',  dest='OBSBINS',  type='string',default='|-1.0|-0.75|-0.50|-0.25|0.0|0.25|0.50|0.75|1.0|',   help='Bin boundaries for the diff. measurement separated by "|", e.g. as "|0|50|100|", use the defalut if empty string')#|0|30|80|200|10000|
    parser.add_option('',   '--year',  dest='YEAR',  type='string', default='2022',   help='Year -> 2016 or 2017 or 2018 or Full')
    parser.add_option('',   '--verbose', action='store_true', dest='VERBOSE', default=False, help='print values')
    parser.add_option('',   '--AC', action='store_true', dest='AC', default=False, help='AC samples')
    parser.add_option('',   '--m4lLower',  dest='LOWER_BOUND',  type='int',default=105.0,   help='Lower bound for m4l')
    parser.add_option('',   '--m4lUpper',  dest='UPPER_BOUND',  type='int',default=140.0,   help='Upper bound for m4l')
    # The following two options are used together to calculate the acceptance in AC scenario to plot AC predictions on fiducial plot
    parser.add_option('',   '--AC_onlyAcc', action='store_true', dest='AC_ONLYACC', default=False, help='Flag in case we are interested in only the acceptance')
    parser.add_option('',   '--AC_hypothesis', dest='AC_HYP',  type='string',default='',   help='Name of the AC hypothesis, e.g. 0M, 0PM')
    # The following option are used in case of interpolation to calculate acceptance at 125.38 GeV
    parser.add_option('',   '--interpolation', action='store_true', dest='INTER', default=False, help='Calculate acceptances at 124 and 126 GeV')
    parser.add_option('',   '--hypothesis', dest='HYP',  type='string',default='24', help='specify mass value: 24(124) or 26(126)')
    # store options and arguments as global variables
    global opt, args
    (opt, args) = parser.parse_args()

    # if (opt.OBSBINS=='' and opt.OBSNAME!='inclusive'):
    #     parser.error('Bin boundaries not specified for differential measurement. Exiting...')
    #     sys.exit()


# parse the arguments and options
global opt, args, runAllSteps
parseOptions()


# ------------------------------- FUNCTIONS TO GENERATE DATAFRAMES ----------------------------------------------------
# Weights for histogram
def weight(df, fail, xsec, gen, lumi, additional = None):
    #Coefficient to calculate weights for histograms
    coeff = (lumi * 1000 * xsec) / gen
    #Gen
    weight_gen = np.sign(df['genHEPMCweight'])# * df.PUWeight
    weight_histo_gen = weight_gen * coeff
    #Reco
    if(fail == False):
        if not opt.AC_ONLYACC: #AC samples are ReReco, there is no SFcorr for ReReco
            weight_reco = np.sign(df['genHEPMCweight']) * df['PUWeight'] * df['dataMCWeight'] #* df['L1prefiringWeight'] * df['SFcorr']
        else:
            weight_reco = np.sign(df['genHEPMCweight']) * df['PUWeight'] * df['dataMCWeight'] #* df['L1prefiringWeight'] * df['SFcorr']
        weight_histo_reco = weight_reco * coeff
    elif(fail == True):
        weight_reco = 0
        weight_histo_reco = weight_reco * coeff
    
    df['weight_gen'] = weight_gen #Powheg
    df['weight_reco'] = weight_reco #Powheg
    df['weight_histo_gen'] = weight_histo_gen #Powheg
    df['weight_histo_reco'] = weight_histo_reco #Powheg
    if additional == 'ggH': #Applies extra NNLOPS weights for gluon-gluon fusion Higgs production
        weight_gen_NNLOPS = weight_gen * df['ggH_NNLOPS_weight']
        weight_reco_NNLOPS = weight_reco * df['ggH_NNLOPS_weight']
        weight_histo_gen_NNLOPS = weight_histo_gen * df['ggH_NNLOPS_weight']
        weight_histo_reco_NNLOPS = weight_histo_reco * df['ggH_NNLOPS_weight']
        df['weight_gen_NNLOPS'] = weight_gen_NNLOPS #NNLOPS (only ggH)
        df['weight_reco_NNLOPS'] = weight_reco_NNLOPS #NNLOPS (only ggH)
        df['weight_histo_gen_NNLOPS'] = weight_histo_gen_NNLOPS #NNLOPS (only ggH)
        df['weight_histo_reco_NNLOPS'] = weight_histo_reco_NNLOPS #NNLOPS (only ggH)
    return df

#Calculates different types of event weights for histogram filling:
#weight_gen: based on generator-level sign of event weights.
#weight_reco: additional corrections like pile-up and data-MC scale factors.
#weight_histo_*: scaled by luminosity and cross-section to normalize MC.
#if additional == 'ggH': #Applies extra NNLOPS weights for gluon-gluon fusion Higgs production


# Uproot to generate pandas
def prepareTrees(year):
    d_sig = {}
    d_sig_failed = {}
    for signal in signals_original:
        fname = path['eos_path_sig']+"MC/"+year+"/"+signal+"/ZZ4lAnalysis_SKIMMED.root"
        print(fname)
        d_sig[signal] = uproot.open(fname)[key]
        d_sig_failed[signal] = uproot.open(fname)[key_failed]

    return d_sig, d_sig_failed
#Loads ROOT files using uproot and returns the TTree objects for each signal process for both:
#Passed events
#Failed events (didn’t pass fiducial/reco cuts)

# Calculate cross sections
def xsecs(year):
    xsec_sig = {}
    d_sig, d_sig_failed = prepareTrees(year)
    for signal in signals_original:
        
        #total_weight = d_sig[signal].pandas.df('overallEventWeight').overallEventWeight
        #puweight = d_sig[signal].pandas.df('PUWeight').PUWeight
        #genweight = d_sig[signal].pandas.df('genHEPMCweight').genHEPMCweight
        if 'ggH' in signal:  
            df = d_sig[signal].arrays(['overallEventWeight', 'PUWeight', 'genHEPMCweight', 'ggH_NNLOPS_weight'], library="ak") 
        else: 
            df = d_sig[signal].arrays(['overallEventWeight', 'PUWeight', 'genHEPMCweight'], library="ak") 
        
        total_weight = df['overallEventWeight'] 
        puweight = df['PUWeight'] 
        genweight = df['genHEPMCweight'] 

        if 'ggH' in signal:
            #nnlops = d_sig[signal].pandas.df('ggH_NNLOPS_weight').ggH_NNLOPS_weight
            nnlops = df['ggH_NNLOPS_weight'] 
            xsec = total_weight/(puweight*genweight*nnlops)

        else:
            xsec = total_weight/(puweight*genweight)
            
        xsec_sig[signal] = xsec[0]
    print(signal, xsec_sig[signal])
    return xsec_sig
#Computes cross-section values for each signal sample using:
#xsec = sum of overall weights / (PUWeight * genHEPMCweight * (ggH weight if applicable) )
 
def add_fin_state_reco(Z1Flav, Z2Flav):
    i, j = np.abs(Z1Flav), np.abs(Z2Flav)

    fin = ak.zeros_like(i, dtype=np.int32)

    # fin = ak.where((i == 0) & (j == 0), 0, fin)  # other
    fin = ak.where((i == 121) & (j == 121), 1, fin)  # 4e
    fin = ak.where((i == 169) & (j == 169), 2, fin)  # 4mu
    fin = ak.where(((i == 121) & (j == 169)) | ((i == 169) & (j == 121)), 3, fin)  # 2e2mu
    fin = ak.where(((i == 225) & (j == 169)) | ((i == 169) & (j == 225)), 4, fin)  # 2tau2mu
    fin = ak.where(((i == 225) & (j == 121)) | ((i == 121) & (j == 225)), 5, fin)  # 2tau2e
    fin = ak.where((i == 225) & (j == 225), 6, fin)  # 4tau

    mapping = ak.Array(["other", "4e", "4mu", "2e2mu", "2tau2mu", "2tau2e", "4tau"])
    fin = mapping[fin]

    return fin
# Determines the reconstructed final state of an event based on Z1 and Z2 flavors (e.g., 4e, 4mu, 2e2mu, etc).

def add_fin_state_gen(lepId, Hindex):
    fin = -ak.ones_like(lepId[:,0], dtype=np.int32)

    lepId = np.abs(lepId)
    Hindex = np.abs(Hindex)

    # Check for invalid indices
    # invalid_mask = (Hindex[:,0] == 99) | (Hindex[:,1] == 99) | (Hindex[:,2] == 99) | (Hindex[:,3] == 99)
    invalid_mask = ak.any(Hindex == 99, axis=1)

    # Replace 99 with a safe index (e.g. 0) so no out-of-bounds
    safe_Hindex = ak.where(Hindex == 99, -1, Hindex)

    row_idx = ak.local_index(lepId, axis=0)

    lep_0 = lepId[row_idx, safe_Hindex[:,0]]
    lep_2 = lepId[row_idx, safe_Hindex[:,2]]

    # Set final states
    fin = ak.where(invalid_mask, 0, fin)  # other
    fin = ak.where((lep_0 == 11) & (lep_2 == 11) & ~invalid_mask, 1, fin)  # 4e
    fin = ak.where((lep_0 == 13) & (lep_2 == 13) & ~invalid_mask, 2, fin)  # 4mu
    fin = ak.where((((lep_0 == 11) & (lep_2 == 13)) | ((lep_0 == 13) & (lep_2 == 11))) & ~invalid_mask, 3, fin)  # 2e2mu

    mapping = ak.Array(["other", "4e", "4mu", "2e2mu"])
    fin = mapping[fin]

    return fin
# Determines generator-level final state using lepton IDs and Higgs decay product indices.

def add_fin_state_gen_out(ZdauId):
    fin = ak.zeros_like(ZdauId[:,0], dtype=np.int32)

    ZdauId_0 = abs(ZdauId[:,0])
    ZdauId_1 = abs(ZdauId[:,1])

    # fin = ak.where((((abs(ZdauId_0)!=11) | (abs(ZdauId_1)!=13)) | ((abs(ZdauId_0)!=13) | (abs(ZdauId_1)!=11))), 0, fin)  # other
    fin = ak.where((abs(ZdauId_0)==11) & (abs(ZdauId_1)==11), 1, fin)  # 4e
    fin = ak.where((abs(ZdauId_0)==13) & (abs(ZdauId_1)==13), 2, fin)  # 4mu
    fin = ak.where((((abs(ZdauId_0)==11) & (abs(ZdauId_1)==13)) | ((abs(ZdauId_0)==13) & (abs(ZdauId_1)==11))), 3, fin)  # 2e2mu

    mapping = ak.Array(["other", "4e", "4mu", "2e2mu"])
    fin = mapping[fin]
    return fin
# Alternative gen-level final state classifier using Z daughters (simplified version, not using Higgs indices).

def add_fin_state_gen_out_ZH(ZdauId,momId):
    fin = ak.zeros_like(ZdauId[:,0])

    ZdauId_0 = np.abs(ZdauId[:,0])
    ZdauId_1 = np.abs(ZdauId[:,1])
    ZdauId_2 = np.abs(ZdauId[:,2])

    momId_0 = momId[:,0]
    momId_1 = momId[:,1]
    momId_2 = momId[:,2]

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
# Tailored version of gen final state classification for ZH samples, requires mother ID check (ensure daughters came from Higgs).

def add_cuth4l_gen(momMomId,Hindex):
    output = ak.zeros_like(momMomId[:,0], dtype=np.int32)

    invalid_mask = ak.any(Hindex == 99, axis=1)  
    safe_Hindex = ak.where(Hindex == 99, -1, Hindex)

    row_idx = ak.local_index(momMomId, axis=0)

    momMomId_0 = momMomId[row_idx, safe_Hindex[:,0]]
    momMomId_1 = momMomId[row_idx, safe_Hindex[:,1]]
    momMomId_2 = momMomId[row_idx, safe_Hindex[:,2]]
    momMomId_3 = momMomId[row_idx, safe_Hindex[:,3]]

    output = ak.where((momMomId_0 == 25) & (momMomId_1 == 25) & (momMomId_2 == 25) & (momMomId_3 == 25) & (~invalid_mask), 1, output)

    mapping = ak.Array([False, True])
    output = mapping[output]

    return output
#Checks if all four generator-level leptons are descendants of a Higgs boson (i.e., if it’s a true H→ZZ→4l decay).

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
#Checks if all reconstructed leptons originated from a Z that came from a Higgs (via gen association).

# Get the "number" of MC events to divide the weights
def generators(year):
    gen_sig = {}
    for signal in signals_original:
        fname = path['eos_path_sig']+"MC/"+year+"/"+signal+"/ZZ4lAnalysis_SKIMMED.root"
        #gen_sig[signal] = uproot.open(fname)["candTree/Counter"].array()[0]
        gen_sig[signal] = uproot.open(fname)["Counters"].values()[39]  
        print("Counters is: ", gen_sig[signal])
    return gen_sig
#Retrieves the number of generated events from a special histogram (Counters) in the ROOT file. This number is needed to normalize event weights.

def createDataframe(d_sig,fail,gen,xsec,signal,lumi,obs_reco,obs_gen,obs_reco_2nd='None',obs_gen_2nd='None'):
    b_sig = ['EventNumber','GENmass4l', 'GENlep_id', 'GENlep_MomId',
             'GENlep_MomMomId', 'GENlep_Hindex', 'GENZ_DaughtersId',
             'GENZ_MomId', 'passedFiducial', 'genHEPMCweight', 'PUWeight']
    if (obs_gen != 'GENmass4l'): 
        b_sig.append(obs_gen)
    if (obs_gen_2nd!='None'): 
        b_sig.append(obs_gen_2nd)
    if 'ggH' in signal and not opt.AC_ONLYACC: 
        b_sig.append('ggH_NNLOPS_weight') #Additional entry for the weight in case of ggH
    if not fail:
        b_sig.extend(['ZZMass', 'Z1Flav', 'Z2Flav', 'dataMCWeight', 'overallEventWeight', 'lep_genindex', 'lep_Hindex'])
                      # 'L1prefiringWeight','dataMCWeight', 'trigEffWeight'])
        # if not opt.AC_ONLYACC: b_sig.append('SFcorr') #AC samples are ReReco, there is no SFcorr for ReReco
        if (obs_reco!='ZZMass'): b_sig.append(obs_reco) #We need to include the if condition otherwise for mass4l ZZMass would be repeated twice
        if (obs_reco_2nd!='None'): b_sig.append(obs_reco_2nd)
        
    #df = d_sig.pandas.df(b_sig, flatten = False)
    df = d_sig.arrays(b_sig, library="ak") 

    if not fail: 
        df['ZZMass'] = ak.flatten(df['ZZMass'], axis=-1) 
        df['Z1Flav'] = ak.flatten(df['Z1Flav'], axis=-1) 
        df['Z2Flav'] = ak.flatten(df['Z2Flav'], axis=-1) 
        df['dataMCWeight'] = ak.flatten(df['dataMCWeight'], axis=-1) 
        df['overallEventWeight'] = ak.flatten(df['overallEventWeight'], axis=-1) 
        # df['lep_genindex'] = df['lep_genindex'].tolist() 
        # df['lep_Hindex'] = df['lep_Hindex'].tolist() 
        if (obs_reco != 'ZZMass'): 
            df[obs_reco] = ak.flatten(df[obs_reco], axis=-1) 
        if (obs_reco_2nd!='None'): 
            df[obs_reco_2nd] = ak.flatten(df[obs_reco_2nd], axis=-1) 
        
    if fail: #Negative branches for failed events (it is useful when creating fiducial pandas)
        df['ZZMass'] = -1
        df['Z1Flav'] = -1
        df['Z2Flav'] = -1
        df['dataMCWeight'] = -1
        df['overallEventWeight'] = -1
        #df['lep_genindex'] = -1 
        df['lep_genindex'] = [[-1, -1, -1, -1]] * len(df) 
        #df['lep_Hindex'] = -1 
        df['lep_Hindex'] = [[-1, -1, -1, -1]] * len(df)	
        # df['L1prefiringWeight'] = -1
        # df['trigEffWeight'] = -1
        if (obs_reco != 'ZZMass'): df[obs_reco] = -1
        if (obs_reco_2nd!='None'): df[obs_reco_2nd] = -1

    df['gen'] = gen
    df['xsec'] = xsec
    if opt.AC_ONLYACC:
        df['ggH_NNLOPS_weight'] = 1 # Set to 1 for ggH, in CJLST ntuple the values is always the same (PERHAPS TO BE UNDERSTOOD)
    if not fail:
        df['FinState_reco'] = add_fin_state_reco(df['Z1Flav'], df['Z2Flav'])
    elif fail:
        df['FinState_reco'] = 'fail'
        
    # df['FinState_gen'] = [add_fin_state_gen(row[0],row[1],row[2]) for row in df[['GENlep_id', 'GENlep_Hindex', 'EventNumber']].values]
    df['FinState_gen'] = add_fin_state_gen(df['GENlep_id'], df['GENlep_Hindex'])
    
    if not 'ZH' in signal:
        df['FinState_gen_out'] = add_fin_state_gen_out(df['GENZ_DaughtersId'])
    else:
        df['FinState_gen_out'] = add_fin_state_gen_out_ZH(df['GENZ_DaughtersId'], df['GENZ_MomId'])

    df['cuth4l_gen'] = add_cuth4l_gen(df['GENlep_MomMomId'], df['GENlep_Hindex'])
    if not fail:
        # df['cuth4l_reco'] = [add_cuth4l_reco(row[0],row[1],row[2],row[3]) for row in df[['lep_Hindex','lep_genindex','GENlep_MomMomId','GENlep_MomId']].values]
        df['cuth4l_reco'] = add_cuth4l_reco(df['lep_Hindex'], df['lep_genindex'], df['GENlep_MomMomId'], df['GENlep_MomId'])
        
    elif fail:
        df['cuth4l_reco'] = False

    df = weight(df, fail, xsec, gen, lumi)
    if not 'ggH' in signal:
        df = weight(df, fail, xsec, gen, lumi)
    else:
        df = weight(df, fail, xsec, gen, lumi, 'ggH')
        # df = df.drop(columns=['ggH_NNLOPS_weight'])

    return df
# Central function to:
# Extract event variables from the ROOT files using uproot
# Assign final states (reco & gen)
# Apply fiducial cuts at gen & reco level
# Apply weights
# Return a clean pandas DataFrame for further analysis


# Set up data frames
def dataframes(year, doubleDiff):
    if year == '2016post':
        lumi = 36.31
    elif year == '2017':
        lumi = 41.48
    elif year == '2018':
        lumi = 59.83
    elif year == '2022EE':
        lumi = 26.6728
    elif year == '2022':
        lumi = 7.9804
    elif year == '2023preBPix':
        lumi = 17.794
    elif year == '2023postBPix':
        lumi = 9.451
    d_df_sig = {}
    d_df_sig_failed = {}
    d_sig, d_sig_failed = prepareTrees(year)
    gen_sig = generators(year)
    xsec_sig = xsecs(year)
    for signal in signals_original:
        print('Processing', signal, year)
        if doubleDiff:
            d_df_sig[signal] = createDataframe(d_sig[signal],False,gen_sig[signal],xsec_sig[signal],signal,lumi,obs_reco,obs_gen,obs_reco_2nd,obs_gen_2nd)
        else:
            d_df_sig[signal] = createDataframe(d_sig[signal],False,gen_sig[signal],xsec_sig[signal],signal,lumi,obs_reco,obs_gen)
        print('Signal created')
        if doubleDiff:
            d_df_sig_failed[signal] = createDataframe(d_sig_failed[signal],True,gen_sig[signal],xsec_sig[signal],signal,lumi,obs_reco,obs_gen,obs_reco_2nd,obs_gen_2nd)
        else:
            d_df_sig_failed[signal] = createDataframe(d_sig_failed[signal],True,gen_sig[signal],xsec_sig[signal],signal,lumi,obs_reco,obs_gen)
        print('Signal failed created')
    return d_df_sig, d_df_sig_failed

# Set Luminosity (lumi): Assigns luminosity based on the year.
# Prepare Data: Loads signal (d_sig) and failed signal (d_sig_failed) data using prepareTrees(year).
# Calculate Cross Sections: Gets cross-sections for signals using xsecs(year).
# Process Signals: For each signal, creates dataframes for both passed and failed events using createDataframe().
# Return: Returns dictionaries with signal dataframes (d_df_sig and d_df_sig_failed).

# Merge WplusH125 and WminusH125
def skim_df(year, doubleDiff):
    d_df_sig, d_df_sig_failed = dataframes(year, doubleDiff)
    d_skim_sig = {}
    d_skim_sig_failed = {}
    frames = []
    for signal in signals_original:
        if ('WplusH1' in signal) or ('WminusH1' in signal):
            frames.append(d_df_sig[signal])
        else:
            d_skim_sig[signal] = d_df_sig[signal]
    if frames: d_skim_sig['WH1'+signal[len(signal)-2]+signal[len(signal)-1]] = ak.concatenate(frames)
    frames = []
    for signal in signals_original:
        if ('WplusH1' in signal) or ('WminusH1' in signal):
            frames.append(d_df_sig_failed[signal])
        else:
            d_skim_sig_failed[signal] = d_df_sig_failed[signal]
    if frames: d_skim_sig_failed['WH1'+signal[len(signal)-2]+signal[len(signal)-1]] = ak.concatenate(frames)
    print('%s SKIMMED df CREATED' %year)
    return d_skim_sig, d_skim_sig_failed

# ------------------------------- FUNCTIONS TO CALCULATE COEFFICIENTS ----------------------------------------------------
def getCoeff(m4l_low, m4l_high, obs_reco, obs_gen, obs_bins, obs_name, type, year, obs_reco_2nd = 'None', obs_gen_2nd = 'None', obs_name_2nd = 'None'):
    if obs_reco != 'ZZMass':
        chans = ['4e', '4mu', '2e2mu']
    else:
        chans = ['4l', '4e', '4mu', '2e2mu']

    # if not doubleDiff:
    #     # #RecoBin limits I'm considering
    #     # obs_reco_low = obs_bins[recobin]
    #     # obs_reco_high = obs_bins[recobin+1]
    #     # #GenBin limits I'm considering
    #     # obs_gen_low = obs_bins[genbin]
    #     # obs_gen_high = obs_bins[genbin+1]
    #     # #Extrimities of gen area
    #     # obs_gen_lowest = obs_bins[0]
    #     # obs_gen_highest = obs_bins[len(obs_bins)-1]
    # elif doubleDiff:
    #     obs_reco_low = obs_bins[recobin][0]
    #     obs_reco_high = obs_bins[recobin][1]
    #     obs_gen_low = obs_bins[genbin][0]
    #     obs_gen_high = obs_bins[genbin][1]
    #     obs_gen_lowest = min(x[0] for x in obs_bins.values())
    #     obs_gen_highest = max(x[1] for x in obs_bins.values())
    #     #Second variable
    #     obs_reco_2nd_low = obs_bins[recobin][2]
    #     obs_reco_2nd_high = obs_bins[recobin][3]
    #     obs_gen_2nd_low = obs_bins[genbin][2]
    #     obs_gen_2nd_high = obs_bins[genbin][3]
    #     obs_gen_2nd_lowest = min(x[2] for x in obs_bins.values())
    #     obs_gen_2nd_highest = max(x[3] for x in obs_bins.values())

    coefficients = defaultdict(lambda: defaultdict(dict))
    for channel in chans:
        for signal in signals:
            if type=='std':
                datafr = d_sig_tot[year][signal]
                genweight = 'weight_gen'
                recoweight = 'weight_reco'
            elif type=='full' or type=='ACggH' or type=='run3' or type=='2022full' or type=='2023full':
                datafr = d_sig_full[signal]
                genweight = 'weight_gen'
                recoweight = 'weight_reco'
            elif type=='fullNNLOPS' and 'ggH' in signal:
                datafr = d_sig_full[signal]
                genweight = 'weight_gen_NNLOPS'
                recoweight = 'weight_reco_NNLOPS'
            elif type=='fullNNLOPS' and not 'ggH' in signal: # In case of fullNNLOPS we are interested in ggH125 only
                continue

            obs_gen_col = datafr[obs_gen]
            obs_reco_col = datafr[obs_reco]
            
            if doubleDiff:
                obs_gen_2nd_col = datafr[obs_gen_2nd]
                obs_reco_2nd_col = datafr[obs_reco_2nd]

            # Selections (in case of Dcp - always 1D - we do not use the absolute value)
            # cutobs_reco = (obs_reco_col >= obs_reco_low) & (obs_reco_col < obs_reco_high)
            # if (obs_name=='Dcp'): 
            #     cutobs_reco = (obs_reco_col >= obs_reco_low) & (obs_reco_col < obs_reco_high)
            #     # cutobs_reco &= (datafr['Z2Mass'] < 60)

            # cutobs_gen = (obs_gen_col >= obs_gen_low) & (obs_gen_col < obs_gen_high)
            # if (obs_name=='Dcp'): 
            #     cutobs_gen = (obs_gen_col >= obs_gen_low) & (obs_gen_col < obs_gen_high)
                
            # if doubleDiff:
            #     cutobs_reco &= (obs_reco_2nd_col >= obs_reco_2nd_low) & (obs_reco_2nd_col < obs_reco_2nd_high)
            #     cutobs_gen &= (obs_gen_2nd_col >= obs_gen_2nd_low) & (obs_gen_2nd_col < obs_gen_2nd_high)
            #     # cutobs_gen &= (datafr['GENmassZ2'] < 60)

            # cutobs_gen_otherfid = ((obs_gen_col >= obs_gen_lowest) & (obs_gen_col < obs_gen_low)) | ((obs_gen_col >= obs_gen_high) & (obs_gen_col <= obs_gen_highest))
            # if (obs_name=='Dcp'): 
            #     cutobs_gen_otherfid = ((obs_gen_col >= obs_gen_lowest) & (obs_gen_col < obs_gen_low)) | ((obs_gen_col >= obs_gen_high) & (obs_gen_col <= obs_gen_highest))
            # if doubleDiff:
            #     cutobs_gen_otherfid |= ((obs_gen_2nd_col >= obs_gen_2nd_lowest) & (obs_gen_2nd_col < obs_gen_2nd_low)) | ((obs_gen_2nd_col >= obs_gen_2nd_high) & (obs_gen_2nd_col <= obs_gen_2nd_highest))

            GENmass4l_col = datafr['GENmass4l']
            cuth4l_gen_col = datafr['cuth4l_gen']
            cuth4l_reco_col = datafr['cuth4l_reco']
            FinState_gen_col = datafr['FinState_gen']
            FinState_reco_col = datafr['FinState_reco']
            passedFiducial_col = datafr['passedFiducial']
            ZZMass_col = datafr['ZZMass']
            FinState_gen_out_col = datafr['FinState_gen_out']

            cutm4l_gen = (GENmass4l_col > m4l_low) & (GENmass4l_col < m4l_high)
            cutnotm4l_gen = (GENmass4l_col <= m4l_low) | (GENmass4l_col >= m4l_high)
            cuth4l_gen = cuth4l_gen_col == True
            cutnoth4l_gen = cuth4l_gen_col == False
            cuth4l_reco = cuth4l_reco_col == True
            cutnoth4l_reco = cuth4l_reco_col == False
            passedFullSelection = FinState_reco_col != 'fail'
            passedFiducialSelection = passedFiducial_col == True
            notPassedFiducialSelection = passedFiducial_col == False

            if channel != '4l':
                cutm4l_reco = (ZZMass_col > m4l_low) & (ZZMass_col < m4l_high) & (FinState_reco_col == channel)
                cutchan_gen = FinState_gen_col == channel
                cutchan_gen_out = FinState_gen_out_col == channel
            else:
                cutm4l_reco = (ZZMass_col > m4l_low) & (ZZMass_col < m4l_high)
                cutchan_gen = (FinState_gen_col == '2e2mu') | (FinState_gen_col == '4e') | (FinState_gen_col == '4mu')
                cutchan_gen_out = (FinState_gen_out_col == '2e2mu') | (FinState_gen_out_col == '4e') | (FinState_gen_out_col == '4mu')

            # if doubleDiff:
            #     processBin = signal+'_'+channel+'_'+obs_name+'_'+obs_name_2nd+'_genbin'+str(genbin)+'_recobin'+str(recobin)
            # else:
            #     processBin = signal+'_'+channel+'_'+obs_name+'_genbin'+str(genbin)+'_recobin'+str(recobin)

            # if type=='fullNNLOPS':
            #     processBin = signal+'_NNLOPS_'+channel+'_'+obs_name+'_genbin'+str(genbin)+'_recobin'+str(recobin)
            # if type=='fullNNLOPS' and doubleDiff:
            #     processBin = signal+'_NNLOPS_'+channel+'_'+obs_name+'_'+obs_name_2nd+'_genbin'+str(genbin)+'_recobin'+str(recobin)

            bin_edges = np.array(obs_bins)
            nBins = len(bin_edges) - 1

            genweight_col = datafr[genweight]
            recoweight_col = datafr[recoweight]

            reco_bin_idx = np.digitize(obs_reco_col, bin_edges) - 1
            gen_bin_idx = np.digitize(obs_gen_col, bin_edges) - 1

            # --------------- Masks for selections ---------------
            mask_gen = passedFiducialSelection & cutm4l_gen & cutchan_gen & cuth4l_gen
            mask_reco = passedFullSelection & cutm4l_reco & cuth4l_reco
            
            mask_gen_reco = mask_gen & mask_reco

            # --------------- acceptance ---------------
            genweight_col_np = ak.to_numpy(genweight_col[mask_gen])
            acc_num = np.histogram(ak.to_numpy(gen_bin_idx[mask_gen]),
                                bins=np.arange(nBins+1),
                                weights=genweight_col_np)[0]

            acc_den = ak.sum(genweight_col[cutchan_gen_out])
            
            acceptances = -ak.ones_like(acc_num, dtype=np.int32)
            acceptances = np.where(acc_den > 0, acc_num / acc_den, -1.0)

            errs_acc = np.sqrt((acceptances*(1-acceptances))/acc_den)
            # print("Acceptances: ", acceptances)
            # print("Errors: ", errs_acc)

            coefficients[signal][channel]['acc'] = np.broadcast_to(acceptances, (nBins, nBins))
            coefficients[signal][channel]['errs_acc'] = np.broadcast_to(errs_acc, (nBins, nBins))

            # In case of fullNNLOPS, we are interested in acceptance only
            if type=='fullNNLOPS' or type=='ACggH': 
                continue 

            # --------------- EffRecoToFid ---------------
            eff_num_matrix = np.histogram2d(ak.to_numpy(reco_bin_idx[mask_gen_reco]),
                                            ak.to_numpy(gen_bin_idx[mask_gen_reco]),
                                            bins=[np.arange(nBins+1), np.arange(nBins+1)],
                                            weights=ak.to_numpy(recoweight_col[mask_gen_reco]))[0]
            
            eff_den = np.histogram(ak.to_numpy(gen_bin_idx[mask_gen]),
                                bins=np.arange(nBins+1),
                                weights=genweight_col_np)[0]
            
            valid_den = eff_den > 0
            efficiencies = np.where(valid_den, eff_num_matrix / eff_den, -1.0)
            efficiencies = np.where(efficiencies == 0, 1e-06, efficiencies)

            # print("Efficiencies: ", efficiencies)

            err_efficiencies = np.sqrt(efficiencies * (1 - efficiencies) / eff_den)
            err_efficiencies = np.where((err_efficiencies > 0) & valid_den, err_efficiencies, 1e-06)
            err_efficiencies = np.where(~valid_den, -1, err_efficiencies)
            # print("Efficiency errors: ", err_efficiencies)

            coefficients[signal][channel]['eff'] = efficiencies
            coefficients[signal][channel]['err_eff'] = err_efficiencies

            # --------------- outinratio ---------------
            oir_num = np.histogram(ak.to_numpy(reco_bin_idx[mask_reco & cutchan_gen_out & (notPassedFiducialSelection | cutnoth4l_gen | cutnotm4l_gen)]),
                                bins=np.arange(nBins+1),
                                weights=ak.to_numpy(recoweight_col[mask_reco & cutchan_gen_out & (notPassedFiducialSelection | cutnoth4l_gen | cutnotm4l_gen)]))[0]

            oir_den = np.histogram(ak.to_numpy(reco_bin_idx[mask_reco & mask_gen]),
                                    bins=np.arange(nBins+1),
                                    weights=ak.to_numpy(recoweight_col[mask_reco & mask_gen]))[0]
            #print('Vectorized oir_den:', oir_den)
            oir = np.where(oir_den > 0, oir_num / oir_den, 0.0)
            # print("Vectorized outinratio: ", oir)

            err_oir = np.sqrt((oir * (1 - oir)) / oir_den)
            err_oir = np.where((oir_den > 0) & (err_oir > 0), err_oir, 1e-06)
            err_oir = np.where(oir_den == 0, 0.0, err_oir)
            # print("Vectorized err_outinratio: ", err_oir)

            coefficients[signal][channel]['outinratio'] = np.broadcast_to(oir[:, None], (nBins, nBins))
            coefficients[signal][channel]['err_outinratio'] = np.broadcast_to(err_oir[:, None], (nBins, nBins))

            # --------------- wrongfrac ---------------
            wf_num = ak.sum(recoweight_col[passedFullSelection & cutm4l_reco & cutnoth4l_reco])
            wf_den = ak.sum(recoweight_col[passedFullSelection & cutm4l_reco])

            wrongfrac = np.where(wf_den > 0, wf_num / wf_den, -1.0)

            coefficients[signal][channel]['wrongfrac'] = np.broadcast_to(wrongfrac, (nBins, nBins))

            # --------------- binfrac_wrongfrac ---------------
            binwf_num = np.histogram(ak.to_numpy(reco_bin_idx[passedFullSelection & cutm4l_reco & cutnoth4l_reco]),
                                        bins=np.arange(nBins+1),
                                        weights=ak.to_numpy(recoweight_col[passedFullSelection & cutm4l_reco & cutnoth4l_reco]))[0]
            binwf_den = ak.sum(recoweight_col[passedFullSelection & cutm4l_reco & cutnoth4l_reco])

            binwf = np.where(binwf_den > 0, binwf_num / binwf_den, -1.0)
            # print("Vectorized binfrac_wrongfrac: ", binwf)

            coefficients[signal][channel]['binfrac_wrongfrac'] = np.broadcast_to(binwf[:, None], (nBins, nBins))

            # --------------- numberFake ---------------
            numberFake = -1
            lambdajesup = 0.0
            lambdajesdn = 0.0

            coefficients[signal][channel]['numberFake'] = np.broadcast_to(numberFake, (nBins, nBins))
            coefficients[signal][channel]['lambdajesup'] = np.broadcast_to(lambdajesup, (nBins, nBins))
            coefficients[signal][channel]['lambdajesdn'] = np.broadcast_to(lambdajesdn, (nBins, nBins))

    import pprint
    pprint.pprint(coefficients, width=120, depth=4)
    return coefficients

def doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, type, obs_reco_2nd = 'None', obs_gen_2nd = 'None', obs_name_2nd = 'None'):
    m4l_low = opt.LOWER_BOUND
    m4l_high = opt.UPPER_BOUND

    nBins = len(obs_bins)
    if not doubleDiff:
        nBins = len(obs_bins)-1 # In case of 1D measurement, the number of bins is -1 the length of obs_bins(=bin boundaries)
    if(opt.AC==True):
        add_ac = 'AC_'
    elif(opt.AC_ONLYACC==True):
        add_ac = 'ACggH_'+opt.AC_HYP+'_'
    elif opt.INTER:
        add_ac = '1'+opt.HYP+'_'
    else:
        add_ac = ''
    if type=='std':
        for year in years:
            test = getCoeff( m4l_low, m4l_high, obs_reco, obs_gen, obs_bins, obs_name, type, year, obs_reco_2nd, obs_gen_2nd, obs_name_2nd)
            # Write dictionaries
            if doubleDiff:
                obs_name_dic = obs_name+'_'+obs_name_2nd
            else:
                obs_name_dic = obs_name
            #Fix 2016post to 2016
            if '2016post' in year:
                year_label = '2016'
            else:
                year_label = year
            if (os.path.exists('../inputs/inputs_sig_'+add_ac+obs_name_dic+'_'+str(year_label)+'_ORIG.py')):
                os.system('rm ../inputs/inputs_sig_'+add_ac+obs_name_dic+'_'+str(year_label)+'_ORIG.py')
            with open('../inputs/inputs_sig_'+add_ac+obs_name_dic+'_'+str(year_label)+'.py', 'w') as f:
                f.write('observableBins = '+str(obs_bins)+';\n')
                f.write('acc = '+str(acceptance)+' \n')
                f.write('err_acc = '+str(err_acceptance)+' \n')
                f.write('eff = '+str(effrecotofid)+' \n')
                f.write('err_eff = '+str(err_effrecotofid)+' \n')
                f.write('outinratio = '+str(outinratio)+' \n')
                f.write('err_outinratio = '+str(err_outinratio)+' \n')
                f.write('inc_wrongfrac = '+str(wrongfrac)+' \n')
                f.write('binfrac_wrongfrac = '+str(binfrac_wrongfrac)+' \n')
                f.write('number_fake = '+str(numberFake)+' \n')
                f.write('lambdajesup = '+str(lambdajesup)+' \n')
                f.write('lambdajesdn = '+str(lambdajesup))

    elif type=='full' or type=='fullNNLOPS' or type=='ACggH' or type=='run3' or type=='2022full' or type=='2023full':
        getCoeff(m4l_low, m4l_high, obs_reco, obs_gen, obs_bins, obs_name, type, 'None', obs_reco_2nd, obs_gen_2nd, obs_name_2nd)

        # Write dictionaries
        if doubleDiff: obs_name_dic = obs_name+'_'+obs_name_2nd
        else: obs_name_dic = obs_name
        if type=='full' or type=='ACggH' or type=='run3' or type=='2022full' or type=='2023full':
            year_label = 'Full'
            if type=='run3': year_label = 'Run3'
            if type=='2022full': year_label = '2022full'
            if type=='2023full': year_label = '2023full'
            if (os.path.exists('../inputs/inputs_sig_'+add_ac+obs_name_dic+'_'+year_label+'_ORIG.py')):
                os.system('rm ../inputs/inputs_sig_'+add_ac+obs_name_dic+'_'+year_label+'_ORIG.py')
            with open('../inputs/inputs_sig_'+add_ac+obs_name_dic+'_'+year_label+'.py', 'w') as f:
                f.write('observableBins = '+str(obs_bins)+';\n')
                f.write('acc = '+str(acceptance)+' \n')
                f.write('err_acc = '+str(err_acceptance)+' \n')
                if type=='full' or type=='run3' or type=='2022full' or type=='2023full':
                    f.write('eff = '+str(effrecotofid)+' \n')
                    f.write('err_eff = '+str(err_effrecotofid)+' \n')
                    f.write('outinratio = '+str(outinratio)+' \n')
                    f.write('err_outinratio = '+str(err_outinratio)+' \n')
                    f.write('inc_wrongfrac = '+str(wrongfrac)+' \n')
                    f.write('binfrac_wrongfrac = '+str(binfrac_wrongfrac)+' \n')
                    f.write('number_fake = '+str(numberFake)+' \n')
                    f.write('lambdajesup = '+str(lambdajesup)+' \n')
                    f.write('lambdajesdn = '+str(lambdajesup))
        elif type=='fullNNLOPS':
            if(opt.YEAR == 'Full'):
                with open('../inputs/inputs_sig_'+add_ac+obs_name_dic+'_NNLOPS_Full.py', 'w') as f:
                    f.write('observableBins = '+str(obs_bins)+';\n')
                    f.write('acc = '+str(acceptance)+' \n')
                    f.write('err_acc = '+str(err_acceptance)+' \n')
            else:
                with open('../inputs/inputs_sig_'+add_ac+obs_name_dic+'_NNLOPS_'+opt.YEAR+'_vectorized.py', 'w') as f:
                    f.write('observableBins = '+str(obs_bins)+';\n')
                    f.write('acc = '+str(acceptance)+' \n')
                    f.write('err_acc = '+str(err_acceptance)+' \n')

# -----------------------------------------------------------------------------------------
# ------------------------------- MAIN ----------------------------------------------------
# -----------------------------------------------------------------------------------------

# print("Starting profiling")
# profiler = cProfile.Profile()
# profiler.enable()

if(opt.AC or opt.AC_ONLYACC):
    if opt.AC_ONLYACC: signals_AC_bare = ['ggH'] #Currently we use only ggH (reweighted to the sum of all production modes) to plot AC predictions
    else: signals_AC_bare = ['VBF', 'WH', 'ggH', 'ZH', 'ttH']
    signals_AC = [root+opt.AC_HYP+'_M125' for root in signals_AC_bare]
    print('AC samples', signals_AC)
    signals_original = signals_AC
    signals = signals_AC
elif opt.INTER:
    signals_original = ['ggH1', 'VBFH1', 'WminusH1', 'WplusH1', 'ZH1']
    signals = ['ggH1', 'VBFH1', 'WH1', 'ZH1']
    signals_original = [root+opt.HYP for root in signals_original]
    signals = [root+opt.HYP for root in signals]
else:
    signals_original = ['ggH125', 'VBFH125', 'WminusH125', 'WplusH125', 'ZH125', "ttH125"]
    signals = ['ggH125', 'VBFH125', 'WH125', 'ZH125', 'ttH125']
eos_path_sig = path['eos_path_sig']
#key = 'candTree'
#key_failed = 'candTree_failed'
key = 'ZZTree/candTree' 
key_failed = 'ZZTree/candTree_failed' 

if (opt.YEAR == '2016'): years = ['2016post']
if (opt.YEAR == '2017'): years = ['2017']
if (opt.YEAR == '2018'): years = ['2018']
if (opt.YEAR == 'Run3'): years = ['2022', '2022EE', '2023preBPix', '2023postBPix']
if (opt.YEAR == 'Full'): years = ['2016post','2017','2018']

if (opt.YEAR == '2022'): years = ['2022']
if (opt.YEAR == '2022EE'): years = ['2022EE']
if (opt.YEAR == '2023preBPix'): years = ['2023preBPix']
if (opt.YEAR == '2023postBPix'): years = ['2023postBPix']

if (opt.YEAR == '2022full'): years = ['2022', '2022EE']
if (opt.YEAR == '2023full'): years = ['2023preBPix', '2023postBPix']

obs_bins, doubleDiff = binning(opt.OBSNAME)
if doubleDiff:
    obs_name = opt.OBSNAME.split(' vs ')[0]
    obs_name_2nd = opt.OBSNAME.split(' vs ')[1]
    obs_name_2d = opt.OBSNAME
else:
    obs_name = opt.OBSNAME

#_temp = __import__('observables', globals(), locals(), ['observables'], -1)
_temp = __import__('observables', globals(), locals(), ['observables'], 0) 

observables = _temp.observables
if doubleDiff:
    obs_reco = observables[obs_name_2d]['obs_reco']
    obs_reco_2nd = observables[obs_name_2d]['obs_reco_2nd']
    obs_gen = observables[obs_name_2d]['obs_gen']
    obs_gen_2nd = observables[obs_name_2d]['obs_gen_2nd']
else:
    obs_reco = observables[obs_name]['obs_reco']
    obs_gen = observables[obs_name]['obs_gen']

print(obs_reco)

print('Following observables extracted from dictionary: RECO = ',obs_reco,' GEN = ',obs_gen)
if doubleDiff:
    print('It is a double-differential measurement: RECO_2nd = ',obs_reco_2nd,' GEN_2nd = ',obs_gen_2nd)

# Generate dataframes
d_sig = {}
d_sig_failed = {}
for year in years:
    sig, sig_failed = skim_df(year, doubleDiff)
    d_sig[year] = sig
    d_sig_failed[year] = sig_failed

# Create dataframe with all the events
d_sig_tot = {}
for year in years:
    d_sup = {}
    for signal in signals:
        print(year, signal)

        d_sup[signal] = ak.concatenate([d_sig[year][signal], d_sig_failed[year][signal]]) # , ignore_index=True, sort=True)        
    d_sig_tot[year] = d_sup

# Create dataframe FullRun2
if((opt.YEAR == 'Full') or (opt.YEAR == 'Run3') or (opt.YEAR == '2022full') or (opt.YEAR == '2023full')):
    d_sig_full = {}
    for signal in signals:
        frame = [d_sig_tot[year][signal] for year in years]
        d_sig_full[signal] = ak.concatenate(frame) # , ignore_index=True, sort=True)
else: # If I work with one year only, the FullRun2 df coincides with d_sig_tot (it is useful when fullNNLOPS is calculated)
    if opt.YEAR == '2016':
        d_sig_full = d_sig_tot[opt.YEAR+'post']
    else:
        d_sig_full = d_sig_tot[opt.YEAR]
print('Dataframes created successfully')

if not opt.AC_ONLYACC:
    print('Coeff std')
    wrongfrac = {}
    binfrac_wrongfrac = {}
    binfrac_outfrac = {}
    outinratio = {}
    err_outinratio = {}
    effrecotofid = {}
    err_effrecotofid = {}
    acceptance = {}
    err_acceptance = {}
    lambdajesup = {}
    lambdajesdn = {}
    numberFake = {}
    if doubleDiff:
        if (opt.YEAR == 'Run3'):
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'run3',  obs_reco_2nd, obs_gen_2nd, obs_name_2nd)
        elif (opt.YEAR == '2022full'):
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, '2022full', obs_reco_2nd, obs_gen_2nd, obs_name_2nd)
        elif (opt.YEAR == '2023full'):
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, '2023full', obs_reco_2nd, obs_gen_2nd, obs_name_2nd)
        else:
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'std', obs_reco_2nd, obs_gen_2nd, obs_name_2nd)
    else:
        if (opt.YEAR == 'Run3'):
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'run3')
        elif (opt.YEAR == '2022full'):
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, '2022full') 
        elif (opt.YEAR == '2023full'):
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, '2023full') 
        else :
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'std')
        
    if (opt.YEAR == 'Full'):
        print('Coeff full')
        wrongfrac = {}
        binfrac_wrongfrac = {}
        binfrac_outfrac = {}
        outinratio = {}
        effrecotofid = {}
        err_effrecotofid = {}
        acceptance = {}
        err_acceptance = {}
        numberFake = {}
        lambdajesup = {}
        lambdajesdn = {}
        if doubleDiff:
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'full', obs_reco_2nd, obs_gen_2nd, obs_name_2nd)
        else:
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'full')

    print('Coeff fullNNLOPS')
    acceptance = {}
    err_acceptance = {}
    # For AC there is no NNLOPS samples
    if not opt.AC:
        if doubleDiff:
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'fullNNLOPS', obs_reco_2nd, obs_gen_2nd, obs_name_2nd)
        else:
            doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'fullNNLOPS')
else:
    print('Coeff full AC ggH')
    acceptance = {}
    err_acceptance = {}
    if doubleDiff:
        doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'ACggH', obs_reco_2nd, obs_gen_2nd, obs_name_2nd)
    else:
        doGetCoeff(obs_reco, obs_gen, obs_name, obs_bins, 'ACggH')

end = time.time()

print(f"RunCoefficients completed {end - start:.5f} seconds")

# profiler.disable()
# print("Profiling complete. Generating report...")
# stats = pstats.Stats(profiler)
# stats.sort_stats('cumulative')
# stats.print_stats(30)  # Show top 30 functions

# # Optional: save to file for later analysis
# stats.dump_stats('test_profile.prof')
