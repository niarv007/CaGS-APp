import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

# ========= LOAD PIPELINE OBJECTS =========
vt = joblib.load("final_models/variance_threshold_selector.pkl")
fs = joblib.load("final_models/feature_selector.pkl")
model = joblib.load("final_models/tuned_random_forest_model.pkl")


# ========= FINGERPRINT GENERATOR =========
def generate_fingerprints_from_smiles(smiles_list):
    ecfp_list, fcfp_list, maccs_list = []
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fcfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=True)
        maccs = MACCSkeys.GenMACCSKeys(mol)

        ecfp_list.append(np.array(ecfp))
        fcfp_list.append(np.array(fcfp))
        maccs_list.append(np.array(maccs))

        valid_smiles.append(smi)

    ecfp_df = pd.DataFrame(ecfp_list, columns=[f'ecfp_{i}' for i in range(1024)])
    fcfp_df = pd.DataFrame(fcfp_list, columns=[f'fcfp_{i}' for i in range(1024)])
    maccs_df = pd.DataFrame(maccs_list, columns=[f'maccs_{i}' for i in range(maccs_list[0].shape[0])])

    X_full = pd.concat([ecfp_df, fcfp_df, maccs_df], axis=1)

    return X_full, valid_smiles


# ========= MAIN PREDICT FUNCTION =========
def predict_from_smiles(smiles):
    if isinstance(smiles, str):
        smiles = [smiles]

    X_full, valid = generate_fingerprints_from_smiles(smiles)

    X_var = vt.transform(X_full)
    X_sel = fs.transform(X_var)

    preds = model.predict(X_sel)
    probs = model.predict_proba(X_sel)[:, 1]

    return list(zip(valid, preds, probs))
