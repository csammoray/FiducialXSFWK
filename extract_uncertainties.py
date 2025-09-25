import numpy as np
from LHScans.resultsXS_LHScan_expected_pT4l_v3 import resultsXS

def extract_values():    
    total_uncerDn = []
    total_uncerUp = []
    total_central = []
    
    stat_uncerDn = []
    stat_uncerUp = []
    stat_central = []
    
    bins = np.arange(9)

    for bin in bins:
        key = f'SM_125_pT4l_genbin{bin}'
        data = resultsXS[key]
        total_uncerDn.append(round(abs(data['uncerDn']), 3))
        total_uncerUp.append(round(data['uncerUp'], 3))
        total_central.append(round(data['central'], 3))
    
    for bin in bins:
        key = f'SM_125_pT4l_genbin{bin}_statOnly'
        data = resultsXS[key]
        stat_uncerDn.append(round(abs(data['uncerDn']), 3))
        stat_uncerUp.append(round(data['uncerUp'], 3))
        stat_central.append(round(data['central'], 3))
    
    return (total_uncerDn, total_uncerUp, total_central, 
            stat_uncerDn, stat_uncerUp, stat_central)


total_uncerDn, total_uncerUp, total_central, stat_uncerDn, stat_uncerUp, stat_central = extract_values()

print("Total uncertainties:")
print(f"Central: {total_central}")
print(f"Uncer Up: {total_uncerUp}")
print(f"Uncer Down: {total_uncerDn}")
print("\nStatistical uncertainties:")
print(f"Central: {stat_central}")
print(f"Uncer Up: {stat_uncerUp}")
print(f"Uncer Down: {stat_uncerDn}")
