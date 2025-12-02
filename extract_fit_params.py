#!/usr/bin/env python3
import ROOT
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

def extractParameters(file_path):
    bestfitVals = {}
    
    inputfile = ROOT.TFile.Open(file_path)
    TTree_limit = inputfile.Get("limit")

    branches = TTree_limit.GetListOfBranches()
    skip_branches = ['limit', 'limitErr', 'mh', 'syst', 'iToy', 'iSeed', 'iChannel', 't_cpu', 't_real', 'quantileExpected', 'deltaNLL']

    nEntries = TTree_limit.GetEntries()
    print(f"Number of entries in tree: {nEntries}")
    
    # Check what each entry represents
    # print("\nChecking quantileExpected for each entry:")
    for i in range(nEntries):
        TTree_limit.GetEntry(i)
        # quantile_leaf = TTree_limit.GetLeaf("quantileExpected")
        # if quantile_leaf:
        #     quantile = quantile_leaf.GetValue()
        #     print(f"  Entry {i}: quantileExpected = {quantile}")
    
    # Now get the bestfit values (typically entry 0 or the one with quantile=0.5)
    TTree_limit.GetEntry(0)

    for branch in branches:
        branch_name = branch.GetName()
 
        if branch_name in skip_branches:
            continue

        leaf = TTree_limit.GetLeaf(branch_name)
        if leaf:
            value = leaf.GetValue()
            bestfitVals[branch_name] = value

    inputfile.Close()
    return bestfitVals

def extract_poi_results(file_path):
    results = {}
    
    inputfile = ROOT.TFile.Open(file_path)
    ttree_limit = inputfile.Get("limit")
    branches = ttree_limit.GetListOfBranches()
    skip_branches = ['limit', 'limitErr', 'mh', 'syst', 'iToy', 'iSeed', 'iChannel', 't_cpu', 't_real', 'quantileExpected', 'deltaNLL']
 
    # nEntries = ttree_limit.GetEntries()
    ttree_limit.GetEntry(0)
    idx = 0
    for branch in branches:
        name = branch.GetName()

        if name in skip_branches:
            continue

        leaf = ttree_limit.GetLeaf(name)
        bestfit_val = leaf.GetValue()

        results[name] = {}
        results[name]['bestfit'] = bestfit_val

        ttree_limit.GetEntry(2*idx + 1)
        lower_bound = ttree_limit.GetLeaf(name).GetValue()
        results[name]['lower_bound'] = bestfit_val - lower_bound

        ttree_limit.GetEntry(2*idx + 2)
        upper_bound = ttree_limit.GetLeaf(name).GetValue()
        results[name]['upper_bound'] = upper_bound - bestfit_val

        idx += 1

    print(results)
    inputfile.Close()
    
    return results

def plotPOIS(path1, path2, doErrors):

    file1_dict = extract_poi_results(path1)
    file2_dict = extract_poi_results(path2)

    file1_dict = extract_poi_results(path1)
    file2_dict = extract_poi_results(path2)

    file1_names = list(file1_dict.keys())
    file1_vals = [file1_dict[name]['bestfit'] for name in file1_names]
    file1_lower_errs = [file1_dict[name]['lower_bound'] for name in file1_names]
    file1_upper_errs = [file1_dict[name]['upper_bound'] for name in file1_names]

    # Same for file2
    file2_names = list(file2_dict.keys())
    file2_vals = [file2_dict[name]['bestfit'] for name in file2_names]
    file2_lower_errs = [file2_dict[name]['lower_bound'] for name in file2_names]
    file2_upper_errs = [file2_dict[name]['upper_bound'] for name in file2_names]

    y_positions = {}
    y_offset = 0.1
    y_spacing = 0.5

    for i, param_name in enumerate(file1_names + file2_names):
        y_positions[param_name] = i * y_spacing

    otherPOIs_y_positions = [y_positions[name] - y_offset for name in file1_names]
    profiled_y_positions = [y_positions[name] + y_offset for name in file2_names]

    _, ax = plt.subplots(figsize=(10, 18))

    ax.axvline(x=0, color='red', linestyle='--', label='SM expected')

    if doErrors:
        ax.errorbar(file1_vals, otherPOIs_y_positions, 
                    xerr=[file1_lower_errs, file1_upper_errs], 
                    fmt='o', color='blue', zorder=2, label='Linear')
        
        ax.errorbar(file2_vals, profiled_y_positions, 
                    xerr=[file2_lower_errs, file2_upper_errs], 
                    fmt='o', color='orange', zorder=2, label='Linear + Quadratic')
    else:
        ax.scatter(file1_vals, otherPOIs_y_positions, 
                   color='blue', zorder=2, label='Linear')
        ax.scatter(file2_vals, profiled_y_positions, 
                   color='orange', zorder=2, label='Linear + Quadratic')

    ax.set_yticks([y_positions[name] for name in file1_names + file2_names])
    ax.set_yticklabels(file1_names + file2_names)

    ax.set_xlabel("Parameter value")
    ax.set_xlim(-2.5, 2.5)
    
    ax.tick_params(axis='y', which='minor', left=False)
    ax.tick_params(axis='y', which='both', right=False)
    
    ax.text(0.0, 1.03,
            rf"$\it{{Private\ work\  (CMS\ simulation)}}$",
            transform=ax.transAxes,
            verticalalignment='top', 
            fontsize=25, 
            fontproperties='Tex Gyre Heros:italic')
    ax.grid(axis='x')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("bestfit_linear_vs_quad.pdf")
    plt.show()



# Example usage:
# Assuming you have your ROOT file loaded and tree accessible
# poi_names = ["chgtil", "cqj18", "chg"]  # First three POIs
# results = extract_poi_results(limit, poi_names)
# 
# # Access results
# for poi, data in results.items():
#     bestfit = data['bestfit']
#     lower_unc, upper_unc = data['uncertainty']
#     print(f"{poi}: {bestfit:.6f} +{upper_unc:.6f} -{lower_unc:.6f}")

if __name__ == "__main__":
    linear_fit = "/eos/user/c/csammora/CMSSW_14_1_0_pre4/src/FiducialXSFWK/datacard/higgsCombinefidXS_fitall_nonProfiled_onlyLinear.MultiDimFit.mH125.38.root"
    lin_quad_fit = "/eos/user/c/csammora/CMSSW_14_1_0_pre4/src/FiducialXSFWK/datacard/higgsCombinefidXS_fitall_nonProfiled.MultiDimFit.mH125.38.root"

    extract_poi_results(lin_quad_fit)
    print("------------------")
    extract_poi_results(linear_fit)

    # test = "/eos/user/c/csammora/CMSSW_14_1_0_pre4/src/FiducialXSFWK/datacard/higgsCombine_pT4l_fitall_Profiled.MultiDimFit.mH125.38.root"
    # print(extractParameters(test))

    # plotPOIS(linear_fit, lin_quad_fit, doErrors=True)
