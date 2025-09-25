from importlib import import_module
import json
from collections import defaultdict

def extractDict(filename):
    
    all = import_module(f"test.{filename}")
    acc = all.acc
    eff = all.eff
    outinratio = all.outinratio
    inc_wrongfrac = all.inc_wrongfrac
    binfrac_wrongfrac = all.binfrac_wrongfrac
    number_fake = all.number_fake
    lambdajesup = all.lambdajesup
    lambdajesdn = all.lambdajesdn

    output_dict = {
        "acc": acc,
        "eff": eff,
        "outinratio": outinratio,
        "inc_wrongfrac": inc_wrongfrac,
        "binfrac_wrongfrac": binfrac_wrongfrac,
        "number_fake": number_fake,
        "lambdajesup": lambdajesup,
        "lambdajesdn": lambdajesdn
    }

    return output_dict


def compareFiles(file1, file2):
    dict1 = extractDict(file1)
    dict2 = extractDict(file2)

    rel_err = defaultdict(dict)
    differences = defaultdict(dict)

    for variable in dict1:
        for key in dict1[variable]:
            if key in dict2[variable]:
                abs_diff = abs(dict1[variable][key] - dict2[variable][key])
                if abs_diff > 0:
                    rel_err[variable][key] = (abs_diff/dict1[variable][key]) * 100
                    differences[variable][key] = abs_diff
                else:
                    continue
            else:
                print(f'{key} not found in second file')

    with open('relative_errors.json', 'w') as f:
        json.dump(rel_err, f, indent=2)

    with open('differences.json', 'w') as f:
        json.dump(differences, f, indent=2)

compareFiles("inputs_sig_pT4l_2022full", "inputs_sig_pT4l_2022full_vectorized")