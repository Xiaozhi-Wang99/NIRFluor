import csv
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import AllChem, Crippen, Lipinski
import warnings
warnings.simplefilter("ignore")

def smiles_standardize(input_file, output_file):
    data = pd.read_csv(input_file)
    data_smiles = data['SMILES'].values.tolist()

    with open(output_file, 'w') as f:
        f.write('Canonical SMILES\n')

        for i, smi in enumerate(data_smiles):
            mol = Chem.MolFromSmiles(smi)

            if mol:
                canonical_smi = Chem.MolToSmiles(mol)
                f.write(canonical_smi + '\n')
            else:
                print("The entered SMILES is incorrect")


def remove_duplicates(input_file, output_file):
    data = pd.read_csv(input_file, encoding='utf-8')
    cleaned_data = data.drop_duplicates(subset=['Canonical SMILES', 'SolventA', 'SolventB', 'Ratio_SolventA'],
                                        keep='last')
    cleaned_data.to_csv(output_file, index=False)


def csv_to_dict(file_path):
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = {}
        for row in csv_reader:
            key = row['Final solvent use']
            values = [row[column] for column in csv_reader.fieldnames if column != 'Final solvent use']
            data[key] = values
        return data


def solvent_matching(input_file, output_file, solvent_dict, solvent_column):
    new_df = pd.DataFrame(columns=['Solvent', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5'])
    for solvent in solvent_column:
        if solvent in solvent_dict:
            data_values = solvent_dict[solvent]
            new_df = new_df.append({'Solvent': solvent,
                                    'Data1': data_values[0],
                                    'Data2': data_values[1],
                                    'Data3': data_values[2],
                                    'Data4': data_values[3],
                                    'Data5': data_values[4]}, ignore_index=True)
        else:
            print("The entered solvent is incorrect")

    new_df.to_csv(output_file, index=False)


class ECFP_fingerprints:
    @classmethod
    def calc_fp(self, mols, radius=3, bit_len=2048):
        ecfp = self.calc_ecfp(mols, radius=radius, bit_len=bit_len)
        phch = self.calc_physchem(mols)
        fps = np.concatenate([ecfp, phch], axis=1)
        return fps

    @classmethod
    def calc_ecfp(cls, mols, radius=3, bit_len=2048):
        fps = np.zeros((len(mols), bit_len))
        for i, mol in enumerate(mols):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, fps[i, :])
            except:
                pass
        return fps

    @classmethod
    def calc_physchem(cls, mols):
        prop_list = ['logP', 'HBA', 'HBD', 'Rotable', 'Amide',
                     'Bridge', 'Hetero', 'Heavy', 'Spiro', 'FCSP3', 'Ring',
                     'Aliphatic', 'Aromatic', 'Saturated', 'HeteroR', 'TPSA', 'MW']
        fps = np.zeros((len(mols), 17))
        props = Property()
        for i, prop in enumerate(prop_list):
            props.prop = prop
            fps[:, i] = props(mols)
        return fps


class Property:
    def __init__(self, prop='MW'):
        self.prop = prop
        self.prop_dict = {'logP': Crippen.MolLogP,
                          'HBA': AllChem.CalcNumLipinskiHBA,
                          'HBD': AllChem.CalcNumLipinskiHBD,
                          'Rotable': AllChem.CalcNumRotatableBonds,
                          'Amide': AllChem.CalcNumAmideBonds,
                          'Bridge': AllChem.CalcNumBridgeheadAtoms,
                          'Hetero': AllChem.CalcNumHeteroatoms,
                          'Heavy': Lipinski.HeavyAtomCount,
                          'Spiro': AllChem.CalcNumSpiroAtoms,
                          'FCSP3': AllChem.CalcFractionCSP3,
                          'Ring': Lipinski.RingCount,
                          'Aliphatic': AllChem.CalcNumAliphaticRings,
                          'Aromatic': AllChem.CalcNumAromaticRings,
                          'Saturated': AllChem.CalcNumSaturatedRings,
                          'HeteroR': AllChem.CalcNumHeterocycles,
                          'TPSA': AllChem.CalcTPSA,
                          'MW': desc.MolWt
                          }

    def __call__(self, mols):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                scores[i] = self.prop_dict[self.prop](mol)
            except:
                continue
        return scores


def generate_fingerprints(input_file, output_file):
    df = pd.read_csv(input_file)
    data_x = ECFP_fingerprints.calc_fp([Chem.MolFromSmiles(mol) for mol in df['Canonical SMILES'].values])
    data_df = pd.DataFrame(data_x)
    data_df.to_csv(output_file, index=False)


def data_processing(input_file, solvent_dict, result_path):
    Standardized_smiles = result_path + '01_Standardized_smiles.csv'
    Cleaned_smiles = result_path + '03_Cleaned_smiles.csv'
    SolventA_matching = result_path + '04_SolventA_matching.csv'
    SolventB_matching = result_path + '05_SolventB_matching.csv'
    Fingerprints = result_path + '06_Fingerprints.csv'

    smiles_standardize(input_file, Standardized_smiles)
    Standardized_smiles_data = pd.read_csv(Standardized_smiles)
    drop_smiles = pd.read_csv(input_file).drop(columns=['SMILES'])
    Standardized_smiles_con = pd.concat([Standardized_smiles_data, drop_smiles], axis=1)
    Standardized_smiles_con.to_csv(result_path + '02_Standardized_smiles_con.csv', index=False)

    Standardized_smiles_all = result_path + '02_Standardized_smiles_con.csv'

    remove_duplicates(Standardized_smiles_all, Cleaned_smiles)
    solvent_matching(Cleaned_smiles, SolventA_matching, solvent_dict, pd.read_csv(Cleaned_smiles)['SolventA'])
    solvent_matching(Cleaned_smiles, SolventB_matching, solvent_dict, pd.read_csv(Cleaned_smiles)['SolventB'])
    generate_fingerprints(Cleaned_smiles, Fingerprints)

    smiles_data = pd.read_csv(Cleaned_smiles)
    solventA_data = pd.read_csv(SolventA_matching)
    solventB_data = pd.read_csv(SolventB_matching)
    fingerprints_data = pd.read_csv(Fingerprints)

    concatenated_data = pd.concat(
        [smiles_data['Canonical SMILES'], solventA_data.drop(columns=['Solvent']), solventB_data.drop(columns=['Solvent']),smiles_data['Ratio_SolventA'],
         fingerprints_data], axis=1)
    concatenated_data.to_csv(result_path + '07_Concatenated_data.csv', index=False)