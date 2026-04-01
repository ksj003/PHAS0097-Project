#!/usr/bin/env python3
"""
KRAS Docking Benchmark
Evaluates docking and structure-prediction tools on KRAS proxy structures
to identify the best method for the KRAS-G12D-NF1 + ligand complex.
"""

import os, sys, subprocess, shutil, urllib.request, re, glob, csv, time, json
import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from Bio import PDB
from Bio.PDB import Select
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress verbose RDKit warnings to keep the console output clean
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Run configuration
SKIP_RUN = False          # Set to True to score existing outputs without re-running tools
RUN_PHASE1 = True         
RUN_PHASE2 = True         
RUN_PHASE3 = True         
TARGET_LIST = []          # Leave empty to process all targets, or specify IDs
DOCKING_ONLY = False      # Set to True to bypass Boltz/Chai

# File and directory paths
BASE_DIR = "/mnt/data2/student/Karam/docking_validation"
RAW_DIR = os.path.join(BASE_DIR, "raw_pdbs")
CSV_DIR = os.path.join(BASE_DIR, "results_csv")
LOG_DIR = os.path.join(BASE_DIR, "tool_logs")
ALIGNED_REFS_DIR = os.path.join(BASE_DIR, "aligned_refs")

COFACTORS = {
    "HOH","WAT","SO4","PO4","MG","NA","CL","CA","ZN","FE","MN","CO","NI","CU",
    "GDP","GTP","GNP","GSP","GCP","ADP","ATP","AMP","FAD","NAD","NAP",
    "EDO","GOL","DMS","ACT","FMT","MPD","PEG","IMD","ACE","NH2","BOG",
    "TRS","EPE","MES","CIT","HED","BME","MLI","IOD","BMA","NAG","FUC",
}

TARGET_INFO = {
    "7RPZ": {"ligand": "6IC", "mutation": "G12D", "pocket": "SII-P", "drug": "MRTX1133", "selectivity": "G12D-selective", "source": "Mirati", "covalent": False},
    "7T47": {"ligand": "6IC", "mutation": "G12D", "pocket": "SII-P", "drug": "MRTX1133(GTP)", "selectivity": "G12D-selective", "source": "Mirati", "covalent": False},
    "8AZV": {"ligand": "OFU", "mutation": "G12D", "pocket": "SII-P", "drug": "BI-2865", "selectivity": "pan-KRAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "8TXE": {"ligand": "VM9", "mutation": "G12D", "pocket": "SII-P", "drug": "ERAS-5024", "selectivity": "G12D-selective", "source": "Erasca", "covalent": False},
    "8TXG": {"ligand": "VQT", "mutation": "G12D", "pocket": "SII-P", "drug": "Erasca-cpd8", "selectivity": "G12D-selective", "source": "Erasca", "covalent": False},
    "8TXH": {"ligand": "VR5", "mutation": "G12D", "pocket": "SII-P", "drug": "Erasca-cpd14", "selectivity": "G12D-selective", "source": "Erasca", "covalent": False},
    "8QUG": {"ligand": "AUTO", "mutation": "G12D", "pocket": "SII-P", "drug": "BI-degrader-WH", "selectivity": "pan-KRAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "8T4V": {"ligand": "Y63", "mutation": "G12D", "pocket": "SII-P", "drug": "malolactone-1", "selectivity": "G12D-covalent", "source": "Shokat-lab", "covalent": True},
    "6OIM": {"ligand": "MOV", "mutation": "G12C", "pocket": "SII-P", "drug": "sotorasib", "selectivity": "G12C-covalent", "source": "Amgen", "covalent": True},
    "6UT0": {"ligand": "AUTO", "mutation": "G12C", "pocket": "SII-P", "drug": "adagrasib", "selectivity": "G12C-covalent", "source": "Mirati", "covalent": True},
    "8B6I": {"ligand": "AUTO", "mutation": "G12C", "pocket": "SII-P", "drug": "AZD4747", "selectivity": "G12C-covalent-CNS", "source": "AstraZeneca", "covalent": True},
    "8X6R": {"ligand": "AUTO", "mutation": "G12C", "pocket": "SII-P", "drug": "ASP6918", "selectivity": "G12C-covalent", "source": "Astellas", "covalent": True},
    "8AZX": {"ligand": "OFU", "mutation": "G12C", "pocket": "SII-P", "drug": "BI-2865(G12C)", "selectivity": "pan-KRAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "8AZY": {"ligand": "OFU", "mutation": "G12V", "pocket": "SII-P", "drug": "BI-2865(G12V)", "selectivity": "pan-KRAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "8B00": {"ligand": "OFU", "mutation": "G13D", "pocket": "SII-P", "drug": "BI-2865(G13D)", "selectivity": "pan-KRAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "8ONV": {"ligand": "VU6", "mutation": "G13D", "pocket": "SII-P", "drug": "BI-2493", "selectivity": "pan-KRAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "8UN3": {"ligand": "AUTO", "mutation": "G13D", "pocket": "SII-P", "drug": "Genentech-G13D-1", "selectivity": "G13D-selective", "source": "Genentech", "covalent": False},
    "8UN4": {"ligand": "AUTO", "mutation": "G13D", "pocket": "SII-P", "drug": "Genentech-G13D-2", "selectivity": "G13D-selective", "source": "Genentech", "covalent": False},
    "8UN5": {"ligand": "AUTO", "mutation": "G13D", "pocket": "SII-P", "drug": "Genentech-G13D-3", "selectivity": "G13D-selective", "source": "Genentech", "covalent": False},
    "6GJ7": {"ligand": "F0B", "mutation": "G12D", "pocket": "SI/SII", "drug": "BI-2852-analog", "selectivity": "pan-RAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "6GJ8": {"ligand": "F0K", "mutation": "G12D", "pocket": "SI/SII", "drug": "BI-2852", "selectivity": "pan-RAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "6ZL5": {"ligand": "F0K", "mutation": "G12D(C118S)", "pocket": "SI/SII", "drug": "BI-2852", "selectivity": "pan-RAS", "source": "Boehringer-Ingelheim", "covalent": False},
    "8R7X": {"ligand": "GCP", "mutation": "G12D", "pocket": "GDP-site", "drug": "GCP-analog", "selectivity": "nucleotide-site", "source": "n/a", "covalent": False},
}

GROUND_TRUTH_SMILES = {
    "6IC": "C#Cc1c(ccc2c1c(cc(c2)O)c3c(c4c(cn3)c(nc(n4)OC[C@@]56CCCN5C[C@@H](C6)F)N7C[C@H]8CC[C@@H](C7)N8)F)F",
    "OFU": "C[C@@H]([C@@H]1CCC[NH+]1C)Oc2ccnc(n2)c3nc(on3)[C@]4(CCCc5c4c(c(s5)N)C#N)C",
    "MOV": "CCC(=O)N1CCN([C@H](C1)C)C2=NC(=O)N(c3c2cc(c(n3)c4c(cccc4F)O)F)c5c(ccnc5C(C)C)C",
    "M1X": "C[C@@H](C(=O)N1CCN(C[C@@H]1CC#N)c2c3c(nc(n2)OC[C@@H]4CCCN4C)CN(CC3)c5cccc6c5c(ccc6)Cl)F",
    "PQI": "CCC(=O)N1CCN2Cc3ccc(c(c3OC[C@@H]2C1)Cl)c4c(ccc5c4c[nH]n5)C",
    "Y9D": "CCC(=O)N1CC2(C1)CCN(CC2)c3c4cc(c(c(c4nc(n3)OC5CCN(CC5)C)OCC)c6c(ccc7c6cn[nH]7)C)C=C",
    "VM9": "Cc1cc(nc(c1C(F)(F)F)c2c(cc3c(c2F)nc(nc3N4C[C@H]5CC[C@@H](C4)N5)OC[C@@]67CCCN6C[C@@H](C7)F)Cl)N",
    "VQT": "c1cc(c2c(c1c3c(cc4c(c3F)nc(nc4N5C[C@H]6CC[C@@H](C5)N6)OC[C@@]78CCCN7C[C@@H](C8)F)Cl)nc(s2)N)F",
    "VR5": "c1cc(c2c(c1c3c(cc4c(c3F)nc(nc4N5C[C@H]6CC[C@@H](C5)N6)OC[C@@]78CCCN7C[C@@H](C8)F)C(F)(F)F)c(c(s2)N)C#N)F",
    "WYU": "C[C@H]1CNCCCN1c2nccc(n2)c3nc(on3)[C@]4(CCCc5c4c(c(s5)N)C#N)C",
    "Y63": "C#Cc1cccc2c1c(ccc2)c3c(c4c(cn3)c(nc(n4)OCC56CCCN5CCC6)N7C[C@H]8CC[C@@H](C7)N8C(=O)CCC(=O)O)F",
    "VU6": "C[C@H]1CNCCN1c2nccc(n2)c3c4c(on3)[C@@]5(CCC4)CCCc6c5c(c(s6)N)C#N",
    "XOI": "Cc1cc(nc(c1C(F)(F)F)c2cc3c4c(c2Cl)OC[C@@H]5CN(CCN5c4nc(n3)OC[C@@H]6C[C@H](CN6C)F)C(=O)C=C)N",
    "XV3": "Cc1cc(nc(c1C(F)(F)F)c2c(cc3c(c2F)nc(nc3N4CCN(C[C@@H]4C)C(=O)/C=C/c5cc(c(cn5)C)CN(C)C)OC[C@@]67CCCN6CC(=C)C7)Cl)N",
    "XQ6": "Cc1cc(nc(c1C(F)(F)F)c2c(cc3c(c2F)nc(nc3N4CCN(C[C@@H]4C)C(=O)/C=C/c5cccc6c5CNCC6)OC[C@@]78CCCN7CC(=C)C8)Cl)N",
    "F0B": "c1ccc(cc1)Cn2ccc3c2cc(cc3)CNCc4c(c5ccccc5[nH]4)[C@@H]6c7cc(ccc7C(=O)N6)O",
    "F0K": "Cn1cc(nc1)Cn2ccc3c2cc(cc3)CNCc4c(c5ccccc5[nH]4)[C@@H]6c7cc(ccc7C(=O)N6)O",
    "GCP": "c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)O[P@@](=O)(CP(=O)(O)O)O)O)O)N=C(NC2=O)N",
}

CROSSDOCK_REFERENCES = {
    "SII-P": ["7RPZ", "8AZV"],
    "SI/SII": ["6GJ8"],
}

APO_TEMPLATES = {
    "G12D": "5US4",   # KRAS G12D-GDP
    "G13D": "4TQA",   # KRAS G13D-GDP
}

POCKET_RESIDUES = {
    "SII-P": [12, 13, 58, 59, 60, 61, 62, 63, 64, 68, 72, 92, 95, 96, 99, 100],
    "SI/SII": [12, 25, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40,
               56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72],
    "GDP-site": [13, 14, 15, 16, 17, 18, 28, 29, 30, 116, 117, 118, 119, 145, 146, 147, 148],
}

CONTACT_DIST = 4.0

# Tool paths and configurations
TMALIGN_BIN = "TMalign"
OBABEL_BIN = "obabel"
BOLTZ_CACHE = "/mnt/data2/student/Karam/boltz/database"
BOLTZ_ENV = "/mnt/data2/student/Karam/env/boltz_env"
BOLTZ_BIN = os.path.join(BOLTZ_ENV, "bin/boltz")
BOLTZ_OUT_DIR = os.path.join(BASE_DIR, "outputs_boltz")
CHAI_ENV = "/mnt/data2/student/Karam/env/chai_env"
CHAI_OUT_DIR = os.path.join(BASE_DIR, "outputs_chai")
DD_OUT_DIR = os.path.join(BASE_DIR, "outputs_diffdock_nim")
NVIDIA_NIM_API_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "REMOVED")
AD_VINA_ENV = "/mnt/data2/student/Karam/env/ad_vina"
AD_OUT_DIR = os.path.join(BASE_DIR, "outputs_ad_vina")
BOLTZ_VINA_OUT_DIR = os.path.join(BASE_DIR, "outputs_boltz_vina")
BOLTZ_DD_OUT_DIR = os.path.join(BASE_DIR, "outputs_boltz_nim")
VINA_BIN = os.path.join(AD_VINA_ENV, "bin/vina")
DDP_OUT_DIR = os.path.join(BASE_DIR, "outputs_diffdock_nim_pocket")
SMINA_BIN = "/mnt/data2/student/Karam/docking_validation/bin/smina"
SMINA_OUT_DIR = os.path.join(BASE_DIR, "outputs_smina")

DOCKING_TOOLS = ["Vina", "Smina", "DiffDock", "DiffDock-P"]
STRUCTPRED_TOOLS = ["Boltz", "Chai-1"]
PIPELINE_TOOLS = ["Boltz+Vina", "Boltz+DD"]
ALL_TOOLS = DOCKING_TOOLS + STRUCTPRED_TOOLS + PIPELINE_TOOLS

AA3TO1 = dict(zip(
    ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU',
     'MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR'],
    list("ACDEFGHIKLMNPQRSTVWY")))

def format_rmsd(val): return f"{val:.2f}" if val is not None else "FAIL"
def format_pct(val): return f"{val:.0f}%" if val is not None else "N/A"
def format_frac(val): return f"{val:.0%}" if val is not None else "N/A"

def calc_avg(values):
    valid = [v for v in values if v is not None]
    return np.mean(valid) if valid else None

def log_result(tool, result):
    """Outputs comprehensive quality metrics for a single tool iteration."""
    rmsd_str = f"{result['rmsd']:.2f}A" if result.get("rmsd") is not None else "FAIL"
    parts = [f"{tool}: RMSD={rmsd_str}"]
    metrics = [
        ("contact_recovery", "CR", "{:.0f}%"),
        ("ifp_tanimoto", "IFT", "{:.3f}"),
        ("burial_frac", "Bur", "{:.0%}"),
        ("pocket_ca_rmsd", "PkCA", "{:.1f}A"),
        ("tm_score", "TM", "{:.3f}"),
        ("validity_pct", "Valid", "{:.0f}%"),
        ("clash_pct", "Clash", "{:.1f}%"),
    ]
    for key, label, fmt in metrics:
        val = result.get(key)
        if val is not None:
            parts.append(f"{label}={fmt.format(val)}")
    msg = result.get("msg", "")
    if msg:
        parts.append(f"[{msg}]")
    print(f"  {' | '.join(parts)}")

def _folder_matches_job(folder, job_id):
    if folder == job_id: return True
    if "___" in folder:
        last_part = folder.split("___")[-1]
        if last_part.startswith(job_id + "_") or last_part == job_id: return True
        return False
    if folder.startswith(job_id + "_") or folder.startswith(job_id + "-") or folder.startswith(job_id + "/"): return True
    stripped = re.sub(r'^(index_?)?\d+[_\-]', '', folder)
    if stripped == job_id or stripped.startswith(job_id + '_') or stripped.startswith(job_id + '-'): return True
    return False

def fetch_ccd_smiles(lig_code):
    """Fetch canonical SMILES from the PDB chemical component dictionary if not hardcoded."""
    if lig_code in GROUND_TRUTH_SMILES: return GROUND_TRUTH_SMILES[lig_code]
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{lig_code}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        descriptors = data.get("pdbx_chem_comp_descriptor", [])
        for d in descriptors:
            if d.get("type") == "SMILES_CANONICAL" and d.get("program") == "OpenEye OEToolkits":
                smi = d.get("descriptor", "")
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    return Chem.MolToSmiles(mol) if mol else smi
        for d in descriptors:
            if "SMILES" in d.get("type", ""):
                smi = d.get("descriptor", "")
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    return Chem.MolToSmiles(mol) if mol else smi
    except: pass
    return None

def load_mol(path):
    if path.endswith('.sdf'): return safe_load_sdf(path)
    return Chem.MolFromPDBFile(path, removeHs=False)

def safe_load_sdf(path):
    try:
        return next(Chem.SDMolSupplier(path, removeHs=False), None)
    except (OSError, Exception):
        return None

def get_mol_coords_and_elements(mol):
    coords, elements = [], []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
            elements.append(atom.GetAtomicNum())
    return np.array(coords), np.array(elements)

def calculate_element_wise_rmsd(ref_coords, ref_elems, pred_coords, pred_elems):
    total_sq, total_n = 0.0, 0
    for elem in np.unique(ref_elems):
        ri = np.where(ref_elems == elem)[0]
        pi = np.where(pred_elems == elem)[0]
        if len(ri) != len(pi):
            raise ValueError(f"Element {elem}: {len(ri)} vs {len(pi)}")
        if len(ri) == 1:
            total_sq += np.linalg.norm(ref_coords[ri[0]] - pred_coords[pi[0]])**2
        else:
            D = cdist(ref_coords[ri], pred_coords[pi])
            row, col = linear_sum_assignment(D)
            total_sq += sum(D[r, c]**2 for r, c in zip(row, col))
        total_n += len(ri)
    return np.sqrt(total_sq / total_n)

def calculate_rmsd(ref_mol, pred_mol):
    try:
        ref_heavy = sum(1 for a in ref_mol.GetAtoms() if a.GetAtomicNum() > 1)
        pred_heavy = sum(1 for a in pred_mol.GetAtoms() if a.GetAtomicNum() > 1)
        if ref_heavy != pred_heavy:
            return None, f"Atom count mismatch: {ref_heavy} vs {pred_heavy}"
        rc, re_ = get_mol_coords_and_elements(ref_mol)
        pc, pe = get_mol_coords_and_elements(pred_mol)
        if len(rc) > 0 and np.mean(np.linalg.norm(rc - pc, axis=1)) < 0.01:
            return 0.0, "Identical coordinates"
        try:
            ref_noH = Chem.RemoveHs(Chem.RWMol(ref_mol))
            pred_noH = Chem.RemoveHs(Chem.RWMol(pred_mol))
            try: Chem.SanitizeMol(ref_noH)
            except: pass
            try: Chem.SanitizeMol(pred_noH)
            except: pass
            return AllChem.GetBestRMS(ref_noH, pred_noH), "GetBestRMS"
        except: pass
        ru, rcnt = np.unique(re_, return_counts=True)
        pu, pcnt = np.unique(pe, return_counts=True)
        if not (np.array_equal(ru, pu) and np.array_equal(rcnt, pcnt)):
            return None, "Element composition differs"
        return calculate_element_wise_rmsd(rc, re_, pc, pe), "Hungarian-fallback"
    except Exception as e:
        return None, str(e)

def calculate_rmsd_tolerant(ref_mol, pred_mol):
    """Calculates RMSD, tolerating atom count mismatches using partial Hungarian matching."""
    try: ref_noH = Chem.RemoveHs(ref_mol)
    except: ref_noH = ref_mol
    try: pred_noH = Chem.RemoveHs(pred_mol)
    except: pred_noH = pred_mol
    ref_heavy = sum(1 for a in ref_noH.GetAtoms() if a.GetAtomicNum() > 1)
    pred_heavy = sum(1 for a in pred_noH.GetAtoms() if a.GetAtomicNum() > 1)
    if ref_heavy == pred_heavy:
        return calculate_rmsd(ref_noH, pred_noH)
    try:
        rc, re_ = get_mol_coords_and_elements(ref_noH)
        pc, pe = get_mol_coords_and_elements(pred_noH)
        total_sq, total_n = 0.0, 0
        for elem in np.unique(re_):
            ri = np.where(re_ == elem)[0]
            pi = np.where(pe == elem)[0]
            n_common = min(len(ri), len(pi))
            if n_common == 0: continue
            D = cdist(rc[ri], pc[pi])
            row, col = linear_sum_assignment(D)
            for r_idx, c_idx in zip(row[:n_common], col[:n_common]):
                total_sq += D[r_idx, c_idx]**2
            total_n += n_common
        if total_n > 0:
            return np.sqrt(total_sq / total_n), f"coord-RMSD({total_n}/{ref_heavy} matched, pred={pred_heavy})"
    except Exception as e:
        return None, f"RMSD-err: {e}"
    return None, "No atoms matched"

def _strip_hydrogens_aggressive(mol):
    """Purge all hydrogen atoms, including those mislabelled during CIF extraction."""
    try: mol = Chem.RemoveHs(mol)
    except: pass
    keep = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    if len(keep) < mol.GetNumAtoms():
        em = Chem.RWMol(mol)
        for i in sorted(set(range(mol.GetNumAtoms())) - set(keep), reverse=True):
            em.RemoveAtom(i)
        mol = em.GetMol()
    return mol

def fix_mol_from_template(extracted_mol, ref_mol, known_smiles=None):
    """
    Standardises bond orders and atom counts on an extracted ligand to match the reference.
    Necessary for covalent targets where the extracted ligand might include amino acid remnants.
    """
    ref_noH = _strip_hydrogens_aggressive(ref_mol)
    ext_noH = _strip_hydrogens_aggressive(extracted_mol)

    ref_n = ref_noH.GetNumAtoms()
    ext_n = ext_noH.GetNumAtoms()
    smi = known_smiles if known_smiles else (Chem.MolToSmiles(Chem.RemoveHs(ref_mol)) if ref_mol else None)

    # Attempt direct template assignment if atom counts match
    if ref_n == ext_n:
        try:
            fixed = AllChem.AssignBondOrdersFromTemplate(ref_noH, ext_noH)
            if fixed and fixed.GetNumAtoms() > 0:
                try: Chem.SanitizeMol(fixed)
                except: pass
                return fixed, "template-fixed"
        except: pass
        if smi:
            try:
                tmpl = Chem.MolFromSmiles(smi)
                if tmpl:
                    fixed = AllChem.AssignBondOrdersFromTemplate(tmpl, ext_noH)
                    if fixed and fixed.GetNumAtoms() > 0:
                        try: Chem.SanitizeMol(fixed)
                        except: pass
                        return fixed, "smiles-template-fixed"
            except: pass

    if known_smiles and smi:
        try:
            tmpl = Chem.MolFromSmiles(smi)
            if tmpl and tmpl.GetNumHeavyAtoms() == ext_n:
                fixed = AllChem.AssignBondOrdersFromTemplate(tmpl, ext_noH)
                if fixed and fixed.GetNumAtoms() > 0:
                    try: Chem.SanitizeMol(fixed)
                    except: pass
                    return fixed, "smiles-template-fixed"
        except: pass

    # Fallback to coordinate transfer via Hungarian matching if topologies don't map cleanly
    try:
        if not smi: return extracted_mol, "template-fix-failed(no-SMILES)"
        tmpl = Chem.MolFromSmiles(smi)
        if not tmpl: return extracted_mol, "template-fix-failed(bad-SMILES)"
        tmpl = Chem.AddHs(tmpl)
        AllChem.EmbedMolecule(tmpl, AllChem.ETKDGv3())
        tmpl = _strip_hydrogens_aggressive(Chem.RemoveHs(tmpl))

        ext_coords, ext_elems = get_mol_coords_and_elements(ext_noH)
        tmpl_coords, tmpl_elems = get_mol_coords_and_elements(tmpl)

        if len(ext_coords) == 0 or len(tmpl_coords) == 0:
            return extracted_mol, "template-fix-failed(no-coords)"

        n_tmpl = len(tmpl_coords)
        new_coords = np.zeros((n_tmpl, 3))
        assigned_ext = set()
        unmatched = []

        for elem in np.unique(tmpl_elems):
            ti = np.where(np.array(tmpl_elems) == elem)[0]
            ei = np.where(np.array(ext_elems) == elem)[0]
            if len(ei) == 0:
                unmatched.extend(ti.tolist()); continue
            D = cdist(tmpl_coords[ti], ext_coords[ei])
            row, col = linear_sum_assignment(D)
            n_matched = min(len(ti), len(ei))
            for r, c in zip(row[:n_matched], col[:n_matched]):
                if ti[r] < n_tmpl:
                    new_coords[ti[r]] = ext_coords[ei[c]]
                    assigned_ext.add(ei[c])
            matched_t = set(ti[r] for r in row[:n_matched] if ti[r] < n_tmpl)
            unmatched.extend([x for x in ti if x not in matched_t and x < n_tmpl])

        if len(unmatched) > n_tmpl * 0.3:
            return extracted_mol, f"template-fix-failed(too-many-unmatched={len(unmatched)}/{n_tmpl})"

        # Shift ETKDG coordinates to match assigned centroid for any unmatched template atoms
        if unmatched and len(assigned_ext) > 0:
            assigned_t = [i for i in range(n_tmpl) if i not in unmatched]
            if assigned_t:
                offset = np.mean(new_coords[assigned_t], axis=0) - np.mean(tmpl_coords[assigned_t], axis=0)
                for ui in unmatched:
                    new_coords[ui] = tmpl_coords[ui] + offset

        conf = tmpl.GetConformer()
        for i in range(min(n_tmpl, tmpl.GetNumAtoms())):
            conf.SetAtomPosition(i, (float(new_coords[i][0]), float(new_coords[i][1]), float(new_coords[i][2])))

        n_matched = n_tmpl - len(unmatched)
        return tmpl, f"coord-mapped({n_matched}/{n_tmpl})"
    except Exception as e:
        return extracted_mol, f"template-fix-failed({e})"

def is_pose_valid(pred_mol, prot_pdb, clash_dist=1.5, max_clash_frac=0.30):
    """Check if the predicted ligand pose has excessive steric clashes with the protein."""
    try:
        parser = PDB.PDBParser(QUIET=True)
        s = parser.get_structure("p", prot_pdb)
        prot_coords = np.array([a.get_coord() for a in s.get_atoms() if a.element.upper() not in ("H", "")])
        if len(prot_coords) == 0: return True, 0.0
        lig_coords, _ = get_mol_coords_and_elements(pred_mol)
        if len(lig_coords) == 0: return True, 0.0
        tree = cKDTree(prot_coords)
        n_clashes = sum(1 for lc in lig_coords if tree.query(lc, k=1)[0] < clash_dist)
        clash_pct = 100.0 * n_clashes / len(lig_coords)
        return (n_clashes / len(lig_coords)) <= max_clash_frac, round(clash_pct, 1)
    except:
        return True, 0.0

def get_crystal_contacts(prot_pdb, ref_mol, pocket_type):
    if pocket_type not in POCKET_RESIDUES: return set(), set()
    known_pocket = set(POCKET_RESIDUES[pocket_type])
    lig_coords, _ = get_mol_coords_and_elements(ref_mol)
    if len(lig_coords) == 0: return known_pocket, set()
    parser = PDB.PDBParser(QUIET=True)
    try: s = parser.get_structure("p", prot_pdb)
    except: return known_pocket, set()
    contacted = set()
    for model in s:
        for chain in model:
            for res in chain:
                if not PDB.is_aa(res, standard=True): continue
                resnum = res.get_id()[1]
                if resnum not in known_pocket: continue
                for atom in res:
                    if atom.element.upper() == "H": continue
                    for lc in lig_coords:
                        if np.linalg.norm(atom.get_coord() - lc) <= CONTACT_DIST:
                            contacted.add(resnum); break
                    if resnum in contacted: break
    return known_pocket, contacted

def compute_pocket_contact_recovery(pred_mol, prot_pdb, pocket_type, ref_contacts=None):
    """Calculate what percentage of the native pocket contacts the prediction manages to recover."""
    if pocket_type not in POCKET_RESIDUES: return 0, 0, 0.0
    known_pocket = set(POCKET_RESIDUES[pocket_type])
    denominator = ref_contacts if ref_contacts else known_pocket
    if not denominator: return 0, 0, 0.0
    try:
        lig_coords, _ = get_mol_coords_and_elements(pred_mol)
        if len(lig_coords) == 0: return 0, len(denominator), 0.0
    except: return 0, len(denominator), 0.0
    parser = PDB.PDBParser(QUIET=True)
    try: s = parser.get_structure("p", prot_pdb)
    except: return 0, len(denominator), 0.0
    pred_contacted = set()
    for model in s:
        for chain in model:
            for res in chain:
                if not PDB.is_aa(res, standard=True): continue
                resnum = res.get_id()[1]
                if resnum not in known_pocket: continue
                for atom in res:
                    if atom.element.upper() == "H": continue
                    for lc in lig_coords:
                        if np.linalg.norm(atom.get_coord() - lc) <= CONTACT_DIST:
                            pred_contacted.add(resnum); break
                    if resnum in pred_contacted: break
    recovered = pred_contacted & denominator
    n_ref = len(denominator)
    return len(recovered), n_ref, round(100.0 * len(recovered) / n_ref, 1) if n_ref > 0 else 0.0

def compute_sasa_burial(pred_mol, prot_pdb):
    try:
        lig_coords, _ = get_mol_coords_and_elements(pred_mol)
        if len(lig_coords) == 0: return 0.0
        parser = PDB.PDBParser(QUIET=True)
        s = parser.get_structure("p", prot_pdb)
        prot_coords = np.array([a.get_coord() for a in s.get_atoms() if a.element.upper() not in ("H", "")])
        if len(prot_coords) == 0: return 0.0
        tree = cKDTree(prot_coords)
        n_buried = sum(1 for lc in lig_coords if len(tree.query_ball_point(lc, 4.0)) >= 3)
        return round(n_buried / len(lig_coords), 3)
    except:
        return 0.0

def compute_ifp_tanimoto(ref_mol, pred_mol, prot_pdb, cutoff=4.5):
    """Computes Interaction Fingerprint Tanimoto similarity to serve as an orientation-aware metric."""
    try:
        try: ref_noH = Chem.RemoveHs(ref_mol)
        except: ref_noH = ref_mol
        try: pred_noH = Chem.RemoveHs(pred_mol)
        except: pred_noH = pred_mol
        ref_coords, ref_elems = get_mol_coords_and_elements(ref_noH)
        pred_coords, pred_elems = get_mol_coords_and_elements(pred_noH)
        if len(ref_coords) == 0 or len(pred_coords) == 0: return None
        atom_mapping = []
        for elem in np.unique(ref_elems):
            ri = np.where(np.array(ref_elems) == elem)[0]
            pi = np.where(np.array(pred_elems) == elem)[0]
            if len(ri) == 0 or len(pi) == 0: continue
            D = cdist(ref_coords[ri], pred_coords[pi])
            row, col = linear_sum_assignment(D)
            for r, c in zip(row[:min(len(ri), len(pi))], col[:min(len(ri), len(pi))]):
                atom_mapping.append((ri[r], pi[c]))
        if not atom_mapping: return None
        parser = PDB.PDBParser(QUIET=True)
        s = parser.get_structure("p", prot_pdb)
        prot_atoms = []
        for model in s:
            for chain in model:
                for res in chain:
                    if not PDB.is_aa(res, standard=True): continue
                    resnum = res.get_id()[1]
                    for atom in res:
                        prot_atoms.append((atom.get_vector().get_array(), resnum))
            break
        if not prot_atoms: return None
        prot_coords_arr = np.array([a[0] for a in prot_atoms])
        prot_resnums = [a[1] for a in prot_atoms]
        tree = cKDTree(prot_coords_arr)
        ref_contacts = set()
        for i in range(len(ref_coords)):
            for n in tree.query_ball_point(ref_coords[i], cutoff):
                ref_contacts.add((i, prot_resnums[n]))
        pred_contacts = set()
        for ref_idx, pred_idx in atom_mapping:
            for n in tree.query_ball_point(pred_coords[pred_idx], cutoff):
                pred_contacts.add((ref_idx, prot_resnums[n]))
        intersection = len(ref_contacts & pred_contacts)
        union = len(ref_contacts | pred_contacts)
        return round(intersection / union, 3) if union > 0 else 0.0
    except:
        return None

def compute_pocket_backbone_rmsd(ref_pdb, pred_file, pocket_type, R=None, t=None):
    if pocket_type not in POCKET_RESIDUES: return None, 0, None
    pocket_res_set = set(POCKET_RESIDUES[pocket_type])
    try:
        pdb_parser = PDB.PDBParser(QUIET=True)
        ref_struct = pdb_parser.get_structure("ref", ref_pdb)
        if pred_file.endswith('.cif'):
            from Bio.PDB.MMCIFParser import MMCIFParser
            pred_struct = MMCIFParser(QUIET=True).get_structure("pred", pred_file)
        else:
            pred_struct = pdb_parser.get_structure("pred", pred_file)
    except:
        return None, 0, None
    def get_ca(struct):
        ca = []
        for model in struct:
            for chain in model:
                for res in chain:
                    if not PDB.is_aa(res, standard=True): continue
                    for atom in res:
                        if atom.get_name() == "CA":
                            ca.append((res.get_id()[1], atom.get_vector().get_array().copy())); break
            break
        return ca
    ref_ca, pred_ca = get_ca(ref_struct), get_ca(pred_struct)
    n = min(len(ref_ca), len(pred_ca))
    if n < 10: return None, 0, None
    pocket_sq, pocket_n, all_sq, all_n = 0.0, 0, 0.0, 0
    for i in range(n):
        ref_rn, rc = ref_ca[i]
        _, pc = pred_ca[i]
        pc = pc.copy()
        if R is not None and t is not None: pc = R @ pc + t
        d2 = np.sum((rc - pc)**2)
        all_sq += d2; all_n += 1
        if ref_rn in pocket_res_set: pocket_sq += d2; pocket_n += 1
    return (np.sqrt(pocket_sq / pocket_n) if pocket_n > 0 else None,
            pocket_n,
            np.sqrt(all_sq / all_n) if all_n > 0 else None)

def compute_all_metrics(mol, ref_mol, prot_pdb, pocket_type, ref_contacts):
    valid, clash_pct = is_pose_valid(mol, prot_pdb)
    _, _, contact_pct = compute_pocket_contact_recovery(mol, prot_pdb, pocket_type, ref_contacts)
    bf = compute_sasa_burial(mol, prot_pdb)
    ift = compute_ifp_tanimoto(ref_mol, mol, prot_pdb)
    rmsd, rmsd_msg = calculate_rmsd(ref_mol, mol)
    return {
        "rmsd": rmsd, "rmsd_msg": rmsd_msg, "ifp_tanimoto": ift,
        "contact_recovery": contact_pct, "burial_frac": bf,
        "validity_pct": 100.0 if valid else 0.0,
        "clash_pct": clash_pct, "valid": valid,
    }

def check_prerequisites():
    missing = []
    if not shutil.which(OBABEL_BIN): missing.append("OpenBabel")
    if not shutil.which(TMALIGN_BIN): missing.append("TM-align")
    if missing:
        print(f"Error: Missing system dependencies - {', '.join(missing)}"); return False
    print("Dependencies successfully verified.")
    return True

def download_pdb(pdb_id):
    os.makedirs(RAW_DIR, exist_ok=True)
    pdb_file = os.path.join(RAW_DIR, f"{pdb_id}.pdb")
    if os.path.exists(pdb_file): return pdb_file
    try:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb_file)
        return pdb_file
    except Exception as e:
        print(f"  Error downloading {pdb_id}: {e}"); return None

def download_mmcif(pdb_id):
    os.makedirs(RAW_DIR, exist_ok=True)
    cif_file = os.path.join(RAW_DIR, f"{pdb_id}.cif")
    if os.path.exists(cif_file): return cif_file
    try:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.cif", cif_file)
        return cif_file
    except Exception as e:
        print(f"  Error downloading mmCIF {pdb_id}: {e}"); return None

def auto_detect_ligand(structure, pdb_id):
    best_res, best_count = None, 0
    for model in structure:
        for chain in model:
            for res in chain:
                if PDB.is_aa(res, standard=True): continue
                rn = res.get_resname().strip()
                if rn in COFACTORS: continue
                heavy = [a for a in res if a.element.upper() != "H"]
                if len(heavy) > best_count:
                    best_count = len(heavy); best_res = res
    return best_res

def parse_target(pdb_id):
    pdb_file = download_pdb(pdb_id)
    if not pdb_file: return None
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)

    RETAIN_COFACTORS = {"GDP", "GTP", "GNP", "GSP", "GCP", "MG"}
    class ProteinWithCofactors(Select):
        def accept_residue(self, r):
            if PDB.is_aa(r, standard=True): return 1
            return 1 if r.get_resname().strip() in RETAIN_COFACTORS else 0

    io_pdb = PDB.PDBIO(); io_pdb.set_structure(structure)
    prot_pdb = os.path.join(RAW_DIR, f"{pdb_id}_protein.pdb")
    io_pdb.save(prot_pdb, ProteinWithCofactors())

    lig_code = TARGET_INFO[pdb_id]["ligand"]
    found_residue, lig_atoms = None, []
    if lig_code == "AUTO":
        found_residue = auto_detect_ligand(structure, pdb_id)
        if found_residue:
            lig_atoms = [a for a in found_residue if a.element.upper() != "H"]
            lig_code = found_residue.get_resname().strip()
            print(f"[Auto-detected: {lig_code} ({len(lig_atoms)} atoms)] ", end="")
    else:
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.get_resname().strip() == lig_code:
                        heavy = [a for a in res if a.element.upper() != "H"]
                        if len(heavy) > 5:
                            found_residue, lig_atoms = res, heavy; break
                if found_residue: break
            if found_residue: break

    if not found_residue or len(lig_atoms) < 5:
        found_residue = auto_detect_ligand(structure, pdb_id)
        if found_residue:
            lig_atoms = [a for a in found_residue if a.element.upper() != "H"]
            lig_code = found_residue.get_resname().strip()
            print(f"[Fallback detection: {lig_code} ({len(lig_atoms)} atoms)] ", end="")

    if not found_residue:
        print(f"  Error: Cannot find ligand in {pdb_id}"); return None

    center = np.mean([a.get_coord() for a in lig_atoms], axis=0)
    class LigSelect(Select):
        def accept_residue(self, r): return r == found_residue
    lig_pdb = os.path.join(RAW_DIR, f"{pdb_id}_lig.pdb")
    io_pdb.save(lig_pdb, LigSelect())

    lig_sdf = os.path.join(RAW_DIR, f"{pdb_id}_lig_ref.sdf")
    mol = Chem.MolFromPDBFile(lig_pdb, removeHs=False)
    if mol:
        mol = Chem.AddHs(mol, addCoords=True)
        w = Chem.SDWriter(lig_sdf); w.write(mol); w.close()
    else:
        subprocess.run([OBABEL_BIN, "-ipdb", lig_pdb, "-osdf", "-O", lig_sdf, "-h"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    smiles = ""
    if lig_code in GROUND_TRUTH_SMILES:
        smiles = GROUND_TRUTH_SMILES[lig_code]
        if not Chem.MolFromSmiles(smiles):
            print(f"  Warning: Hardcoded SMILES for {lig_code} failed RDKit evaluation."); smiles = ""
    if not smiles:
        ccd_smi = fetch_ccd_smiles(lig_code)
        if ccd_smi: smiles = ccd_smi; print(f"[Fetched CCD SMILES] ", end="")
    if not smiles:
        if mol: smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        else:
            try:
                r = subprocess.run([OBABEL_BIN, "-ipdb", lig_pdb, "-osmi"], capture_output=True, text=True, timeout=30)
                if r.stdout.strip():
                    smi = r.stdout.strip().split('\n')[0].split('\t')[0].strip()
                    t = Chem.MolFromSmiles(smi)
                    smiles = Chem.MolToSmiles(t) if t else smi
            except: pass
        if smiles: print(f"[Derived SMILES fallback] ", end="")

    longest_seq = ""
    for model in structure:
        for chain in model:
            seq = "".join([AA3TO1.get(r.get_resname(), 'X') for r in chain if PDB.is_aa(r)])
            if len(seq) > len(longest_seq): longest_seq = seq

    return {
        "pdb_id": pdb_id, "prot_pdb": prot_pdb,
        "lig_sdf": lig_sdf, "lig_pdb": lig_pdb,
        "center": center, "seq": longest_seq, "smiles": smiles,
        "lig_code": lig_code, "n_heavy": len(lig_atoms), "n_residues": len(longest_seq),
    }

def parse_tmalign_matrix(matrix_file):
    try:
        with open(matrix_file, 'r') as f: lines = f.readlines()
        rows = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try: rows.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                except ValueError: continue
        if len(rows) >= 3:
            t = np.array([rows[0][0], rows[1][0], rows[2][0]])
            R = np.array([[rows[i][j] for j in range(1,4)] for i in range(3)])
            return R, t
    except: pass
    return None, None

def align_structures(ref_prot, query_prot):
    matrix_file = os.path.join(ALIGNED_REFS_DIR, "tmalign_matrix.txt")
    result = subprocess.run([TMALIGN_BIN, query_prot, ref_prot, "-m", matrix_file],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tm_match = re.search(r'TM-score\s*=\s*([0-9.]+)', result.stdout)
    tm_score = float(tm_match.group(1)) if tm_match else 0.0
    if not os.path.exists(matrix_file): return None
    R, t = parse_tmalign_matrix(matrix_file)
    if R is None: return None
    return R, t, tm_score

def transform_mol(mol, R, t):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        old = np.array([pos.x, pos.y, pos.z])
        new = R @ old + t
        conf.SetAtomPosition(i, (float(new[0]), float(new[1]), float(new[2])))
    return mol

def generate_crossdock_reference(prot_target, lig_source):
    out_sdf = os.path.join(ALIGNED_REFS_DIR, f"xdock_ref_{prot_target['pdb_id']}_{lig_source['pdb_id']}.sdf")
    if os.path.exists(out_sdf) and os.path.getsize(out_sdf) > 50: return out_sdf
    alignment = align_structures(prot_target["prot_pdb"], lig_source["prot_pdb"])
    if alignment is None: return None
    R, t, _ = alignment
    mol = load_mol(lig_source["lig_sdf"]) or load_mol(lig_source["lig_pdb"])
    if not mol: return None
    mol = transform_mol(mol, R, t)
    w = Chem.SDWriter(out_sdf); w.write(mol); w.close()
    return out_sdf

def _find_boltz_cif(search_dir):
    if not os.path.exists(search_dir): return None
    for root, _, files in os.walk(search_dir):
        for f in sorted(files):
            if f.endswith('.cif') and not f.startswith('.'):
                full = os.path.join(root, f)
                if os.path.getsize(full) > 100: return full
    return None

def _find_all_boltz_cifs(search_dir):
    cifs = []
    if not os.path.exists(search_dir): return cifs
    for root, _, files in os.walk(search_dir):
        for f in sorted(files):
            if f.endswith('.cif') and not f.startswith('.'):
                full = os.path.join(root, f)
                if os.path.getsize(full) > 100: cifs.append(full)
    return cifs

def find_all_boltz_models(job_id):
    for prefix in [f"boltz_results_{job_id}", job_id]:
        cifs = _find_all_boltz_cifs(os.path.join(BOLTZ_OUT_DIR, prefix))
        if cifs: return cifs
    if os.path.exists(BOLTZ_OUT_DIR):
        for entry in os.listdir(BOLTZ_OUT_DIR):
            if _folder_matches_job(entry, job_id):
                cifs = _find_all_boltz_cifs(os.path.join(BOLTZ_OUT_DIR, entry))
                if cifs: return cifs
    return []

def find_all_chai_models(job_id):
    chai_dir = os.path.join(CHAI_OUT_DIR, job_id)
    if not os.path.exists(chai_dir): return []
    cifs = sorted(glob.glob(os.path.join(chai_dir, "pred.model_idx_*.cif")))
    return [c for c in cifs if os.path.getsize(c) > 100]

def run_boltz(job_id, seq, smiles, template_pdb=None, recycling=3, pocket_type=None):
    result_dir = os.path.join(BOLTZ_OUT_DIR, f"boltz_results_{job_id}")
    for d in [result_dir, os.path.join(BOLTZ_OUT_DIR, job_id)]:
        if _find_boltz_cif(d): return True
    if not smiles or not smiles.strip():
        print(f"    Boltz ({job_id}): Skipped - Missing SMILES"); return False
    try:
        test_mol = Chem.MolFromSmiles(smiles)
        if not test_mol: print(f"    Boltz ({job_id}): Skipped - Invalid SMILES"); return False
        canon_smiles = Chem.MolToSmiles(test_mol)
    except:
        print(f"    Boltz ({job_id}): Skipped - SMILES processing error"); return False

    print(f"    Running Boltz ({job_id})...")
    yaml_dir = os.path.join(BASE_DIR, "inputs_boltz")
    os.makedirs(yaml_dir, exist_ok=True); os.makedirs(BOLTZ_OUT_DIR, exist_ok=True)
    yaml_path = os.path.join(yaml_dir, f"{job_id}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"version: 1\nsequences:\n  - protein:\n      id: [A]\n"
                f"      sequence: {seq}\n  - ligand:\n      id: [B]\n"
                f"      smiles: \"{canon_smiles}\"\n")
        if template_pdb and os.path.exists(template_pdb):
            pdb_code = os.path.basename(template_pdb).replace('.pdb', '').upper()[:4]
            tmpl_cif = download_mmcif(pdb_code) if len(pdb_code) == 4 else None
            tmpl_file = tmpl_cif if (tmpl_cif and os.path.exists(tmpl_cif)) else template_pdb
            tmpl_ext = "cif" if tmpl_file.endswith('.cif') else "pdb"
            f.write(f"templates:\n  - {tmpl_ext}: {tmpl_file}\n    chain_id: [A]\n")
        if pocket_type and pocket_type in POCKET_RESIDUES:
            key_contacts = {"SII-P": [12, 60, 95, 99], "SI/SII": [12, 32, 40, 70], "GDP-site": [15, 117, 146]}
            contacts = key_contacts.get(pocket_type, POCKET_RESIDUES[pocket_type][:4])
            contact_str = ", ".join(f"[A, {r}]" for r in contacts)
            f.write(f"constraints:\n  - pocket:\n      binder: B\n      contacts: [{contact_str}]\n")

    for label, msa_args in [("with MSA server", ["--use_msa_server"]), ("without MSA", [])]:
        if os.path.exists(result_dir):
            if _find_boltz_cif(result_dir): return True
            shutil.rmtree(result_dir, ignore_errors=True)
        log_file = os.path.join(LOG_DIR, f"boltz_{job_id}.log")
        cmd = [BOLTZ_BIN, "predict", yaml_path, "--out_dir", BOLTZ_OUT_DIR,
               "--cache", BOLTZ_CACHE, "--recycling_steps", str(recycling),
               "--diffusion_samples", "10", "--accelerator", "gpu", "--devices", "1", "--no_kernels"] + msa_args
        try:
            with open(log_file, "w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"      Boltz timeout ({label})")
            if label == "with MSA server": continue
            return False
        found = _find_boltz_cif(result_dir)
        if not found and os.path.exists(BOLTZ_OUT_DIR):
            for entry in os.listdir(BOLTZ_OUT_DIR):
                if _folder_matches_job(entry, job_id):
                    found = _find_boltz_cif(os.path.join(BOLTZ_OUT_DIR, entry))
                    if found: break
        if found:
            print(f"      Boltz successful ({label})"); return True
        if os.path.exists(log_file):
            try:
                for line in open(log_file).readlines()[-20:]:
                    if any(k in line.lower() for k in ['error', 'exception', 'traceback']):
                        print(f"      Boltz error ({label}): {line.strip()}"); break
            except: pass
        if label == "with MSA server": print(f"      Boltz: Retrying without MSA...")
    print(f"      Boltz ({job_id}): Yielded no output files."); return False

def run_chai(job_id, seq, smiles, template_pdb=None, pocket_type=None):
    out_cif = os.path.join(CHAI_OUT_DIR, job_id, "pred.model_idx_0.cif")
    if os.path.exists(out_cif): return True
    if not smiles or not smiles.strip():
        print(f"    Chai-1 ({job_id}): Skipped - Missing SMILES"); return False
    print(f"    Running Chai-1 ({job_id})...")
    work_dir = os.path.join(CHAI_OUT_DIR, job_id)
    fasta_dir = os.path.join(BASE_DIR, "inputs_chai"); os.makedirs(fasta_dir, exist_ok=True)
    fasta = os.path.join(fasta_dir, f"{job_id}.fasta")
    with open(fasta, "w") as f: f.write(f">protein|A\n{seq}\n>ligand|B\n{smiles}\n")

    tmpl_m8, tmpl_dir = None, None
    if template_pdb and os.path.exists(template_pdb):
        tmpl_dir = os.path.join(BASE_DIR, "chai_templates"); os.makedirs(tmpl_dir, exist_ok=True)
        tmpl_pdb_code = os.path.splitext(os.path.basename(template_pdb))[0].lower()
        tmpl_id = f"{tmpl_pdb_code}_A"
        tmpl_cif_gz = os.path.join(tmpl_dir, f"{tmpl_id}.cif.gz")
        if not os.path.exists(tmpl_cif_gz):
            pdb_code = tmpl_pdb_code.upper()[:4]
            rcsb_cif = download_mmcif(pdb_code)
            source_cif = rcsb_cif if (rcsb_cif and os.path.exists(rcsb_cif)) else None
            if not source_cif:
                tc = os.path.join(tmpl_dir, f"{tmpl_id}.cif")
                subprocess.run([OBABEL_BIN, "-ipdb", template_pdb, "-ocif", "-O", tc],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                if os.path.exists(tc): source_cif = tc
            if source_cif and os.path.exists(source_cif):
                import gzip
                with open(source_cif, 'rb') as fi, gzip.open(tmpl_cif_gz, 'wb') as fo: fo.write(fi.read())
        if os.path.exists(tmpl_cif_gz):
            m8_dir = os.path.join(BASE_DIR, "chai_m8_files"); os.makedirs(m8_dir, exist_ok=True)
            tmpl_m8 = os.path.join(m8_dir, f"{job_id}.m8")
            with open(tmpl_m8, "w") as f:
                f.write(f"A\t{tmpl_id}\t100.0\t{len(seq)}\t0\t0\t1\t{len(seq)}\t1\t{len(seq)}\t0\t0\n")

    constraint_path = None
    if pocket_type:
        cdir = os.path.join(BASE_DIR, "chai_constraints"); os.makedirs(cdir, exist_ok=True)
        constraint_path = _generate_chai_constraints(seq, pocket_type, os.path.join(cdir, f"{job_id}.restraints"))

    chai_python = os.path.join(CHAI_ENV, "bin", "python")
    attempts = []
    if (tmpl_m8 and tmpl_dir) or constraint_path: attempts.append(("with constraints & templates", True, True))
    if constraint_path: attempts.append(("with constraints only", False, True))
    attempts.append(("without constraints", False, False))

    for label, use_tmpl, use_constr in attempts:
        if os.path.exists(work_dir): shutil.rmtree(work_dir, ignore_errors=True); time.sleep(0.3)
        os.makedirs(work_dir, exist_ok=True)
        tmpl_env = f'os.environ["CHAI_TEMPLATE_CIF_FOLDER"] = "{tmpl_dir}"' if use_tmpl and tmpl_dir and tmpl_m8 else ""
        tmpl_arg = f'Path("{tmpl_m8}")' if use_tmpl and tmpl_m8 else "None"
        constr_arg = f'Path("{constraint_path}")' if use_constr and constraint_path else "None"
        ws_path = os.path.join(BASE_DIR, "chai_scripts", f"run_{job_id}.py")
        os.makedirs(os.path.dirname(ws_path), exist_ok=True)
        with open(ws_path, "w") as ws:
            ws.write(f'''import os, sys, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
{tmpl_env}
from chai_lab.chai1 import run_inference
try:
    run_inference(fasta_file=Path("{fasta}"), output_dir=Path("{work_dir}"),
                  constraint_path={constr_arg}, num_trunk_recycles=3,
                  num_diffn_timesteps=200, seed=42, device="cuda:0", use_esm_embeddings=False)
    print("CHAI_SUCCESS")
except Exception as e:
    print(f"CHAI_ERROR: {{e}}", file=sys.stderr); sys.exit(1)
''')
        log_file = os.path.join(LOG_DIR, f"chai_{job_id}.log")
        try:
            with open(log_file, "w") as log:
                subprocess.run([chai_python, ws_path], stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"      Chai-1 timeout ({label})"); continue
        if os.path.exists(out_cif): print(f"      Chai-1 successful ({label})"); return True
        if os.path.exists(log_file):
            try:
                for line in open(log_file).readlines()[-15:]:
                    if any(k in line.lower() for k in ['error', 'assert', 'traceback']):
                        print(f"      Chai-1 error ({label}): {line.strip()}")
            except: pass
        if use_tmpl or use_constr: print(f"      Chai-1 failed {label}, trying next configuration...")
    return False

def _get_residue_id(seq, resnum):
    idx = resnum - 1
    return f"{seq[idx]}{resnum}" if 0 <= idx < len(seq) else f"X{resnum}"

def _generate_chai_constraints(seq, pocket_type, output_path):
    key_contacts = {"SII-P": [12, 60, 95, 99], "SI/SII": [12, 32, 40, 70], "GDP-site": [15, 117, 146]}
    contacts = key_contacts.get(pocket_type)
    if not contacts: return None
    lines = ["chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id\n"]
    for i, r in enumerate(contacts):
        lines.append(f"B,,A,{_get_residue_id(seq, r)},pocket,1.0,0.0,11.0,{pocket_type},restraint_{i}\n")
    with open(output_path, "w") as f: f.writelines(lines)
    return output_path

def run_vina(job_id, prot_path, lig_sdf, center):
    wd = os.path.join(AD_OUT_DIR, job_id)
    if os.path.exists(os.path.join(wd, f"{job_id}_vina_out.sdf")) or os.path.exists(os.path.join(wd, f"{job_id}_vina_out.pdbqt")): return True
    print(f"    Running Vina ({job_id})..."); os.makedirs(wd, exist_ok=True)
    rec, lig = os.path.join(wd, "rec.pdbqt"), os.path.join(wd, "lig.pdbqt")
    if not os.path.exists(rec):
        subprocess.run([OBABEL_BIN, "-ipdb", prot_path, "-opdbqt", "-O", rec, "-xr", "-p", "7.4"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if not os.path.exists(lig):
        subprocess.run([OBABEL_BIN, "-isdf", lig_sdf, "-opdbqt", "-O", lig, "-p", "7.4"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if os.path.exists(rec) and os.path.exists(lig):
        with open(os.path.join(LOG_DIR, f"vina_{job_id}.log"), "w") as log:
            subprocess.run([VINA_BIN, "--receptor", rec, "--ligand", lig, "--out", os.path.join(wd, f"{job_id}_vina_out.pdbqt"),
                            "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
                            "--size_x", "25", "--size_y", "25", "--size_z", "25", "--exhaustiveness", "32"],
                           stdout=log, stderr=subprocess.STDOUT)
    return True

def run_smina(job_id, prot_path, lig_sdf, center):
    out_dir = os.path.join(SMINA_OUT_DIR, job_id)
    out_sdf = os.path.join(out_dir, f"{job_id}_docked.sdf")
    if os.path.exists(out_sdf): return True
    print(f"    Running Smina ({job_id})..."); os.makedirs(out_dir, exist_ok=True)
    prot_h = os.path.join(out_dir, "rec_protonated.pdb")
    if not os.path.exists(prot_h):
        subprocess.run([OBABEL_BIN, "-ipdb", prot_path, "-opdb", "-O", prot_h, "-p", "7.4"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    rec = prot_h if os.path.exists(prot_h) else prot_path
    with open(os.path.join(LOG_DIR, f"smina_{job_id}.log"), "w") as log:
        subprocess.run([SMINA_BIN, "--receptor", rec, "--ligand", lig_sdf, "--out", out_sdf,
                        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
                        "--size_x", "25", "--size_y", "25", "--size_z", "25", "--exhaustiveness", "32"],
                       stdout=log, stderr=subprocess.STDOUT)
    return True

def run_diffdock(job_id, prot_pdb, lig_sdf, num_poses=20, override_out_dir=None):
    import requests
    out_dir = override_out_dir or os.path.join(DD_OUT_DIR, job_id)
    if os.path.exists(os.path.join(out_dir, "rank1.sdf")): return True
    if not NVIDIA_NIM_API_KEY: print(f"    DiffDock ({job_id}): Skipped - Missing API key"); return False
    print(f"    Running DiffDock ({job_id})..."); os.makedirs(out_dir, exist_ok=True)
    with open(prot_pdb, 'r') as f: prot_content = f.read()
    with open(lig_sdf, 'r') as f: lig_content = f.read()
    headers = {"Authorization": f"Bearer {NVIDIA_NIM_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"ligand": lig_content, "ligand_file_type": "sdf", "protein": prot_content,
               "num_poses": num_poses, "time_divisions": 20, "num_steps": 40}
    try:
        response = requests.post("https://health.api.nvidia.com/v1/biology/mit/diffdock", headers=headers, json=payload, timeout=300)
        if response.status_code == 202:
            rid = response.headers.get("NVCF-REQID", "")
            for _ in range(60):
                time.sleep(5)
                pr = requests.get(f"https://health.api.nvidia.com/v1/status/{rid}", headers={"Authorization": f"Bearer {NVIDIA_NIM_API_KEY}"}, timeout=30)
                if pr.status_code == 200: response = pr; break
                elif pr.status_code != 202: return False
            else: return False
        if response.status_code != 200: print(f"      DiffDock: API error status {response.status_code}"); return False
        result = response.json()
        poses = result.get("docked_ligand", []) or result.get("ligand_positions", [])
        confidences = result.get("pose_confidence", []) or result.get("confidence", [])
        if not poses:
            if isinstance(result, list): poses = result
            elif "output" in result: poses = result["output"] if isinstance(result["output"], list) else [result["output"]]
            else: print(f"      DiffDock: No poses returned."); return False
        ranked = sorted(zip(confidences, poses), key=lambda x: -x[0]) if confidences and len(confidences) == len(poses) else [(0, p) for p in poses]
        for i, (_, pose_sdf) in enumerate(ranked):
            if not pose_sdf or len(pose_sdf.strip()) < 10: continue
            pp = os.path.join(out_dir, f"rank{i+1}.sdf")
            with open(pp, 'w') as f: f.write(pose_sdf)
            if safe_load_sdf(pp) is None: os.remove(pp)
        print(f"      DiffDock: Saved {len(ranked)} poses.")
        return os.path.exists(os.path.join(out_dir, "rank1.sdf"))
    except Exception as e:
        print(f"      DiffDock: Execution error - {e}"); return False

def trim_protein_to_pocket(prot_pdb, center, radius=15.0, out_pdb=None):
    parser = PDB.PDBParser(QUIET=True)
    struct = parser.get_structure("prot", prot_pdb)
    center = np.array(center)
    keep = set()
    for model in struct:
        for chain in model:
            for res in chain:
                if not PDB.is_aa(res, standard=True): continue
                for atom in res:
                    if np.linalg.norm(atom.get_vector().get_array() - center) < radius:
                        keep.add((chain.id, res.get_id())); break
        break
    if not keep: return prot_pdb
    class PS(Select):
        def accept_residue(self, res): return (res.get_parent().id, res.get_id()) in keep
    if out_pdb is None: out_pdb = prot_pdb.replace('.pdb', '_pocket.pdb')
    io = PDB.PDBIO(); io.set_structure(struct); io.save(out_pdb, PS())
    return out_pdb

def run_diffdock_pocket(job_id, prot_pdb, lig_sdf, center, num_poses=20):
    out_dir = os.path.join(DDP_OUT_DIR, job_id)
    if os.path.exists(os.path.join(out_dir, "rank1.sdf")): return True
    os.makedirs(out_dir, exist_ok=True)
    trimmed = trim_protein_to_pocket(prot_pdb, center, 20.0, os.path.join(out_dir, "pocket_protein.pdb"))
    print(f"    Running DiffDock-P ({job_id})...")
    return run_diffdock(job_id + "_DDP", trimmed, lig_sdf, num_poses, out_dir)

def _align_boltz_protein(boltz_cif, ref_prot_pdb, lig_sdf, work_dir):
    """Extracts protein from a Boltz CIF result and aligns it to the reference structure."""
    lig_pdb_ext, pred_pdb = extract_ligand_from_structpred(boltz_cif, lig_sdf)
    if not pred_pdb: return None
    alignment = align_structures(ref_prot_pdb, pred_pdb)
    if alignment is None: return None
    R, t, tm_score = alignment
    os.makedirs(work_dir, exist_ok=True)
    raw_pdb = os.path.join(work_dir, "boltz_raw.pdb")
    subprocess.run([OBABEL_BIN, "-icif", boltz_cif, "-opdb", "-O", raw_pdb], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if not os.path.exists(raw_pdb) or os.path.getsize(raw_pdb) < 100: return None
    try: boltz_struct = PDB.PDBParser(QUIET=True).get_structure("boltz", raw_pdb)
    except: return None
    for atom in boltz_struct.get_atoms(): atom.set_coord(R @ atom.get_vector().get_array() + t)
    aligned_prot = os.path.join(work_dir, "boltz_aligned_protein.pdb")
    class PS(Select):
        def accept_residue(self, res): return PDB.is_aa(res, standard=True)
    io = PDB.PDBIO(); io.set_structure(boltz_struct); io.save(aligned_prot, PS())
    if not os.path.exists(aligned_prot) or os.path.getsize(aligned_prot) < 100: return None
    boltz_center = None
    if lig_pdb_ext:
        try:
            m = load_mol(lig_pdb_ext) or Chem.MolFromPDBFile(lig_pdb_ext, removeHs=False, sanitize=False)
            if m:
                coords = np.array([list(m.GetConformer().GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
                boltz_center = ((R @ coords.T).T + t).mean(axis=0).tolist()
        except: pass
    return aligned_prot, boltz_center, R, t, tm_score

def run_boltz_vina(job_id, boltz_job_id, ref_prot_pdb, lig_sdf, center):
    wd = os.path.join(BOLTZ_VINA_OUT_DIR, job_id)
    if os.path.exists(os.path.join(wd, f"{job_id}_vina_out.sdf")) or os.path.exists(os.path.join(wd, f"{job_id}_vina_out.pdbqt")): return True
    boltz_cif = find_pred_file("Boltz", boltz_job_id)
    if not boltz_cif: print(f"    Boltz+Vina ({job_id}): Skipped - Missing Boltz structure"); return False
    result = _align_boltz_protein(boltz_cif, ref_prot_pdb, lig_sdf, wd)
    if not result: print(f"    Boltz+Vina ({job_id}): Skipped - Alignment failure"); return False
    aligned_prot, boltz_center, _, _, tm_score = result
    bc = boltz_center or center
    print(f"    Running Boltz+Vina ({job_id}, TM={tm_score:.3f})...")
    rec, lig = os.path.join(wd, "rec.pdbqt"), os.path.join(wd, "lig.pdbqt")
    if not os.path.exists(rec):
        subprocess.run([OBABEL_BIN, "-ipdb", aligned_prot, "-opdbqt", "-O", rec, "-xr", "-p", "7.4"], capture_output=True)
        if not os.path.exists(rec): print(f"      rec.pdbqt conversion failed"); return False
    if not os.path.exists(lig):
        subprocess.run([OBABEL_BIN, "-isdf", lig_sdf, "-opdbqt", "-O", lig, "-p", "7.4"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        if not os.path.exists(lig): print(f"      lig.pdbqt conversion failed"); return False
    out_pdbqt = os.path.join(wd, f"{job_id}_vina_out.pdbqt")
    with open(os.path.join(LOG_DIR, f"boltz_vina_{job_id}.log"), "w") as log:
        subprocess.run([VINA_BIN, "--receptor", rec, "--ligand", lig, "--out", out_pdbqt,
                        "--center_x", str(bc[0]), "--center_y", str(bc[1]), "--center_z", str(bc[2]),
                        "--size_x", "25", "--size_y", "25", "--size_z", "25", "--exhaustiveness", "32"],
                       stdout=log, stderr=subprocess.STDOUT)
    return os.path.exists(out_pdbqt)

def run_boltz_dd(job_id, boltz_job_id, ref_prot_pdb, lig_sdf, center):
    import requests
    out_dir = os.path.join(BOLTZ_DD_OUT_DIR, job_id)
    if os.path.exists(os.path.join(out_dir, "rank1.sdf")): return True
    if not NVIDIA_NIM_API_KEY: return False
    boltz_cif = find_pred_file("Boltz", boltz_job_id)
    if not boltz_cif: print(f"    Boltz+DD ({job_id}): Skipped - Missing Boltz structure"); return False
    result = _align_boltz_protein(boltz_cif, ref_prot_pdb, lig_sdf, out_dir)
    if not result: print(f"    Boltz+DD ({job_id}): Skipped - Alignment failure"); return False
    aligned_prot, _, _, _, tm_score = result
    print(f"    Running Boltz+DD ({job_id}, TM={tm_score:.3f})...")
    return run_diffdock(job_id + "_BDD", aligned_prot, lig_sdf, 20, out_dir)

def find_all_ranked_sdfs(directory):
    if not os.path.exists(directory): return []
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.startswith("rank") and f.endswith(".sdf")]

def _find_first_valid_sdf(out_dir):
    if not os.path.exists(out_dir): return None
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("rank") and f.endswith(".sdf"):
            p = os.path.join(out_dir, f)
            if safe_load_sdf(p) is not None: return p
    return None

def _resolve_vina_output(work_dir, job_id):
    sdf = os.path.join(work_dir, f"{job_id}_vina_out.sdf")
    pdbqt = os.path.join(work_dir, f"{job_id}_vina_out.pdbqt")
    if os.path.exists(sdf) and safe_load_sdf(sdf): return sdf
    if os.path.exists(sdf): os.remove(sdf)
    if not os.path.exists(pdbqt): return None
    subprocess.run([OBABEL_BIN, "-ipdbqt", pdbqt, "-osdf", "-O", sdf], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if os.path.exists(sdf) and safe_load_sdf(sdf): return sdf
    if os.path.exists(sdf): os.remove(sdf)
    pdb_tmp = os.path.join(work_dir, f"{job_id}_tmp.pdb")
    with open(pdbqt, 'r') as fin, open(pdb_tmp, 'w') as fout:
        for line in fin:
            if line.startswith(('ATOM', 'HETATM')): fout.write(line[:66].rstrip() + '\n')
            elif line.startswith('ENDMDL'): fout.write('END\n'); break
        else: fout.write('END\n')
    mol = Chem.MolFromPDBFile(pdb_tmp, removeHs=False, sanitize=False)
    if mol:
        try: Chem.SanitizeMol(mol)
        except: pass
        w = Chem.SDWriter(sdf); w.write(mol); w.close()
        if safe_load_sdf(sdf): return sdf
    return pdbqt if os.path.exists(pdbqt) else None

def find_pred_file(tool, job_id):
    if tool == "Boltz":
        for d in [os.path.join(BOLTZ_OUT_DIR, f"boltz_results_{job_id}"), os.path.join(BOLTZ_OUT_DIR, job_id)]:
            cif = _find_boltz_cif(d)
            if cif: return cif
        if os.path.exists(BOLTZ_OUT_DIR):
            for entry in os.listdir(BOLTZ_OUT_DIR):
                if _folder_matches_job(entry, job_id):
                    cif = _find_boltz_cif(os.path.join(BOLTZ_OUT_DIR, entry))
                    if cif: return cif
        return None
    elif tool == "Chai-1":
        f = os.path.join(CHAI_OUT_DIR, job_id, "pred.model_idx_0.cif")
        return f if os.path.exists(f) else None
    elif tool == "Vina": return _resolve_vina_output(os.path.join(AD_OUT_DIR, job_id), job_id)
    elif tool == "Smina":
        f = os.path.join(SMINA_OUT_DIR, job_id, f"{job_id}_docked.sdf")
        return f if os.path.exists(f) else None
    elif tool == "DiffDock": return _find_first_valid_sdf(os.path.join(DD_OUT_DIR, job_id))
    elif tool == "DiffDock-P": return _find_first_valid_sdf(os.path.join(DDP_OUT_DIR, job_id))
    elif tool == "Boltz+Vina": return _resolve_vina_output(os.path.join(BOLTZ_VINA_OUT_DIR, job_id), job_id)
    elif tool == "Boltz+DD": return _find_first_valid_sdf(os.path.join(BOLTZ_DD_OUT_DIR, job_id))
    return None

def extract_ligand_from_structpred(pred_path, ref_lig_path):
    ref_mol = load_mol(ref_lig_path) if ref_lig_path else None
    ref_heavy = sum(1 for a in ref_mol.GetAtoms() if a.GetAtomicNum() > 1) if ref_mol else 20
    KNOWN_LIG = {"LIG", "LIG2", "UNK", "UNL", "DRG", "LIG1", "MOL"}
    skip = set(AA3TO1.keys()) | (COFACTORS - {"GCP"})

    if pred_path.endswith('.cif'):
        try:
            from Bio.PDB.MMCIFParser import MMCIFParser
            structure = MMCIFParser(QUIET=True).get_structure("pred", pred_path)
        except Exception as e: print(f"      CIF parsing failed: {e}"); return None, None
        full_pdb = pred_path + ".full.pdb"
        io = PDB.PDBIO(); io.set_structure(structure); io.save(full_pdb)
        candidates = []
        for model in structure:
            for chain in model:
                n_aa = sum(1 for r in chain if PDB.is_aa(r, standard=True))
                is_prot = n_aa > 10
                for res in chain:
                    rn = res.get_resname().strip()
                    if PDB.is_aa(res, standard=True) or rn in skip: continue
                    heavy = [a for a in res if a.element.upper() not in ("H", "")]
                    if len(heavy) < 3: continue
                    coords = np.array([a.get_coord() for a in heavy])
                    if np.all(np.abs(coords) < 0.01): continue
                    diff = abs(len(heavy) - ref_heavy)
                    candidates.append({"res": res, "n_heavy": len(heavy), "diff": diff,
                                       "score": diff + (50 if is_prot else 0) + (-20 if rn in KNOWN_LIG else 0),
                                       "chain": chain.id, "resname": rn})
        if not candidates:
            from collections import Counter
            npa = [(a, r) for model in structure for chain in model for r in chain
                   if not PDB.is_aa(r, standard=True) and r.get_resname().strip() not in {"HOH","WAT"}
                   for a in r if a.element.upper() not in ("H", "")]
            if len(npa) >= 3:
                rc = Counter(x[1] for x in npa)
                br = max(rc, key=rc.get)
                if rc[br] >= 3:
                    candidates.append({"res": br, "n_heavy": rc[br], "diff": abs(rc[br]-ref_heavy),
                                       "score": abs(rc[br]-ref_heavy), "chain": "?", "resname": br.get_resname().strip()})
        if candidates:
            candidates.sort(key=lambda x: x["score"])
            best = candidates[0]
            if best["diff"] > 25 and best["resname"] not in KNOWN_LIG: return None, None
            target_res = best["res"]
            class LS(Select):
                def accept_residue(self, r): return r == target_res
            lig_pdb = pred_path + ".extracted_lig.pdb"; io.save(lig_pdb, LS())
            ext_mol = Chem.MolFromPDBFile(lig_pdb, removeHs=False, sanitize=False)
            if ext_mol:
                coords = np.array([[ext_mol.GetConformer().GetAtomPosition(i).x,
                                    ext_mol.GetConformer().GetAtomPosition(i).y,
                                    ext_mol.GetConformer().GetAtomPosition(i).z] for i in range(ext_mol.GetNumAtoms())])
                if np.all(np.abs(coords) < 0.01): return None, None
            print(f"      Extracted: {best['resname']} from chain {best['chain']} ({best['n_heavy']} atoms)")
            return lig_pdb, full_pdb
        return None, None

    if not os.path.exists(pred_path): return None, None
    try: s = PDB.PDBParser(QUIET=True).get_structure("pred", pred_path)
    except: return None, None
    best_res, best_score = None, float('inf')
    for model in s:
        for chain in model:
            n_aa = sum(1 for r in chain if PDB.is_aa(r, standard=True))
            for res in chain:
                rn = res.get_resname().strip()
                if rn in skip or PDB.is_aa(res, standard=True): continue
                heavy = [a for a in res if a.element.upper() not in ("H", "")]
                if len(heavy) < 3: continue
                score = abs(len(heavy)-ref_heavy) + (50 if n_aa > 10 else 0) + (-20 if rn in KNOWN_LIG else 0)
                if score < best_score: best_score, best_res = score, res
    if best_res:
        class S(Select):
            def accept_residue(self, r): return r == best_res
        lp = pred_path.replace('.pdb', '_extracted_lig.pdb')
        io = PDB.PDBIO(); io.set_structure(s); io.save(lp, S())
        return lp, pred_path
    return None, None

def score_multipose_ifp_oracle(pose_files, ref_mol, prot_pdb, pocket_type, ref_contacts):
    """Scores multiple docked poses and returns the best matching pose based on IFP Tanimoto."""
    scored = []
    for pf in pose_files:
        mol = safe_load_sdf(pf)
        if mol is None: continue
        m = compute_all_metrics(mol, ref_mol, prot_pdb, pocket_type, ref_contacts)
        m["file"] = pf; m["mol"] = mol; m["rank"] = os.path.basename(pf).split('.')[0]
        scored.append(m)
    if not scored: return None

    scored.sort(key=lambda x: -(x["ifp_tanimoto"] if x["ifp_tanimoto"] is not None else -1))
    oracle = scored[0]
    conf_r1_rmsd = None
    for sp in scored:
        if re.match(r'^rank1(_|$|\.|confidence)', sp["rank"]) or sp["rank"] == 'rank1':
            conf_r1_rmsd = sp["rmsd"]; break

    n_valid = sum(1 for sp in scored if sp["valid"])
    n_total = len(scored)
    vp = round(100.0 * n_valid / n_total, 1)

    return {
        "rmsd": oracle["rmsd"], "ifp_tanimoto": oracle["ifp_tanimoto"],
        "validity_pct": vp, "contact_recovery": oracle["contact_recovery"],
        "burial_frac": oracle["burial_frac"], "clash_pct": oracle["clash_pct"],
        "msg": f"IFP-oracle={format_frac(oracle['ifp_tanimoto'])} RMSD={format_rmsd(oracle['rmsd'])} conf_RMSD={format_rmsd(conf_r1_rmsd)} valid={vp:.0f}%",
    }

def score_structpred_ifp_oracle(tool, job_id, model_cifs, ref_mol, ref_lig_path, prot_pdb, pocket_type, ref_contacts, known_smiles=None):
    """Scores models from structure prediction tools, choosing the best match via IFP Tanimoto."""
    best_result, best_ift = None, -1.0
    for cif in model_cifs:
        lig_ext, pred_pdb = extract_ligand_from_structpred(cif, ref_lig_path or "")
        if not lig_ext: continue
        alignment = align_structures(prot_pdb, pred_pdb)
        if alignment is None: continue
        R, t, tm_score = alignment
        mol = load_mol(lig_ext) or Chem.MolFromPDBFile(lig_ext, removeHs=False, sanitize=False)
        if not mol:
            sdf_ext = lig_ext.replace(".pdb", ".sdf")
            if not sdf_ext.endswith(".sdf"): sdf_ext = lig_ext + ".sdf"
            try:
                subprocess.run([OBABEL_BIN, "-ipdb", lig_ext, "-osdf", "-O", sdf_ext, "-h"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=30)
                if os.path.exists(sdf_ext): mol = safe_load_sdf(sdf_ext)
            except: pass
        if not mol: continue
        mol = transform_mol(mol, R, t)
        mol, fix_msg = fix_mol_from_template(mol, ref_mol, known_smiles=known_smiles)
        try: eval_mol = Chem.RemoveHs(mol)
        except: eval_mol = mol
        m = compute_all_metrics(eval_mol, ref_mol, prot_pdb, pocket_type, ref_contacts)
        ift = m["ifp_tanimoto"] if m["ifp_tanimoto"] is not None else -1.0
        rmsd_val, rmsd_msg = calculate_rmsd_tolerant(ref_mol, eval_mol)
        pca, _, _ = compute_pocket_backbone_rmsd(prot_pdb, cif, pocket_type, R, t)
        if ift > best_ift:
            best_ift = ift
            best_result = {
                "tool": tool, "job_id": job_id, "rmsd": rmsd_val, "ifp_tanimoto": m["ifp_tanimoto"],
                "contact_recovery": m["contact_recovery"],
                "burial_frac": m["burial_frac"],
                "validity_pct": m["validity_pct"], "clash_pct": m["clash_pct"],
                "pocket_ca_rmsd": round(pca, 2) if pca is not None else None,
                "tm_score": round(tm_score, 3),
                "msg": f"Best of {len(model_cifs)} by IFP (TM={tm_score:.3f}) {fix_msg} {rmsd_msg}",
            }
    return best_result

def score_experiment(tool, job_id, ref_mol, prot_pdb, center, ref_lig_path=None,
                     pocket_type=None, ref_contacts=None, known_smiles=None):
    """Coordinates pose extraction and quality assessment for a single tool output."""
    result = {"tool": tool, "job_id": job_id}

    if tool == "Boltz":
        models = find_all_boltz_models(job_id)
        if len(models) > 1:
            r = score_structpred_ifp_oracle(tool, job_id, models, ref_mol, ref_lig_path, prot_pdb, pocket_type, ref_contacts, known_smiles)
            if r: return r
    if tool == "Chai-1":
        models = find_all_chai_models(job_id)
        if len(models) > 1:
            r = score_structpred_ifp_oracle(tool, job_id, models, ref_mol, ref_lig_path, prot_pdb, pocket_type, ref_contacts, known_smiles)
            if r: return r

    pred_file = find_pred_file(tool, job_id)
    if not pred_file:
        result["rmsd"] = None; result["msg"] = "Result not found"; return result

    if tool in STRUCTPRED_TOOLS:
        lig_ext, pred_pdb = extract_ligand_from_structpred(pred_file, ref_lig_path or "")
        if not lig_ext: result["rmsd"] = None; result["msg"] = "Extraction failure"; return result
        alignment = align_structures(prot_pdb, pred_pdb)
        if alignment is None: result["rmsd"] = None; result["msg"] = "Alignment failure"; return result
        R, t, tm = alignment
        pca, _, _ = compute_pocket_backbone_rmsd(prot_pdb, pred_file, pocket_type, R, t)
        if pca is not None: result["pocket_ca_rmsd"] = round(pca, 2)
        mol = load_mol(lig_ext) or Chem.MolFromPDBFile(lig_ext, removeHs=False, sanitize=False)
        if not mol:
            se = lig_ext.replace('.pdb', '.sdf')
            if not se.endswith('.sdf'): se = lig_ext + '.sdf'
            try:
                subprocess.run([OBABEL_BIN, "-ipdb", lig_ext, "-osdf", "-O", se, "-h"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=30)
                if os.path.exists(se): mol = safe_load_sdf(se)
            except: pass
        if not mol: result["rmsd"] = None; result["msg"] = "Cannot load ligand"; return result
        mol = transform_mol(mol, R, t)
        mol, fix_msg = fix_mol_from_template(mol, ref_mol, known_smiles=known_smiles)
        try: eval_mol = Chem.RemoveHs(mol)
        except: eval_mol = mol
        rmsd_val, rmsd_msg = calculate_rmsd_tolerant(ref_mol, eval_mol)
        result["rmsd"] = rmsd_val; result["msg"] = f"TM={tm:.3f} {fix_msg} {rmsd_msg}"
        result["tm_score"] = round(tm, 3)
        m = compute_all_metrics(eval_mol, ref_mol, prot_pdb, pocket_type, ref_contacts)
        for k in ["ifp_tanimoto","contact_recovery","burial_frac","validity_pct","clash_pct"]:
            result[k] = m[k]
        return result

    MULTIPOSE = {
        "DiffDock": lambda: find_all_ranked_sdfs(os.path.join(DD_OUT_DIR, job_id)),
        "DiffDock-P": lambda: find_all_ranked_sdfs(os.path.join(DDP_OUT_DIR, job_id)),
        "Boltz+DD": lambda: find_all_ranked_sdfs(os.path.join(BOLTZ_DD_OUT_DIR, job_id)),
    }
    if tool in MULTIPOSE:
        poses = MULTIPOSE[tool]()
        if poses:
            oracle = score_multipose_ifp_oracle(poses, ref_mol, prot_pdb, pocket_type, ref_contacts)
            if oracle: result.update(oracle); return result

    pred_mol = load_mol(pred_file)
    if not pred_mol and pred_file.endswith('.pdbqt'):
        pdb_tmp = pred_file.replace('.pdbqt', '_score_tmp.pdb')
        with open(pred_file, 'r') as fin, open(pdb_tmp, 'w') as fout:
            for line in fin:
                if line.startswith(('ATOM', 'HETATM')): fout.write(line[:66].rstrip() + '\n')
                elif line.startswith('ENDMDL'): fout.write('END\n'); break
            else: fout.write('END\n')
        pred_mol = Chem.MolFromPDBFile(pdb_tmp, removeHs=False, sanitize=False)
        if pred_mol:
            try: Chem.SanitizeMol(pred_mol)
            except: pass
    if not pred_mol:
        result["rmsd"] = None; result["msg"] = f"Failed to load prediction geometry"; return result
    m = compute_all_metrics(pred_mol, ref_mol, prot_pdb, pocket_type, ref_contacts)
    for k in ["rmsd","ifp_tanimoto","contact_recovery","burial_frac","validity_pct","clash_pct"]:
        result[k] = m[k]
    result["msg"] = m.get("rmsd_msg", "")
    return result

def _run_tool(tool, job_id, p, center, template_pdb=None, pocket_type=None,
              prot_pdb_override=None, seq_override=None):
    """Runs the selected tool and returns the wall-clock execution time."""
    prot = prot_pdb_override or p["prot_pdb"]
    seq = seq_override or p["seq"]
    t0 = time.time()
    if tool == "Boltz": run_boltz(job_id, seq, p["smiles"], template_pdb, pocket_type=pocket_type)
    elif tool == "Chai-1": run_chai(job_id, seq, p["smiles"], template_pdb, pocket_type=pocket_type)
    elif tool == "DiffDock": run_diffdock(job_id, prot, p["lig_sdf"])
    elif tool == "DiffDock-P": run_diffdock_pocket(job_id, prot, p["lig_sdf"], center)
    elif tool == "Vina": run_vina(job_id, prot, p["lig_sdf"], center)
    elif tool == "Smina": run_smina(job_id, prot, p["lig_sdf"], center)
    elif tool == "Boltz+Vina": run_boltz_vina(job_id, job_id, prot, p["lig_sdf"], center)
    elif tool == "Boltz+DD": run_boltz_dd(job_id, job_id, prot, p["lig_sdf"], center)
    return time.time() - t0

def run_phase1_redocking(targets, parsed, active_tools):
    print("\nPhase 1: Redocking (Sanity Check)")
    results, timings = {}, {t: [] for t in active_tools}
    for idx, pid in enumerate(targets, 1):
        info = TARGET_INFO[pid]; p = parsed[pid]
        print(f"\n[{idx}/{len(targets)}] {pid} ({info['drug']}, {info['mutation']}, {info['pocket']})")
        if not SKIP_RUN:
            for tool in active_tools:
                elapsed = _run_tool(tool, pid, p, p["center"], os.path.join(RAW_DIR, f"{pid}.pdb"), info['pocket'])
                if elapsed >= 1.0: timings[tool].append(elapsed)
        ref_mol = load_mol(p["lig_sdf"]) or load_mol(p["lig_pdb"])
        if not ref_mol: print(f"  Error: Cannot load reference ligand."); continue
        _, ref_contacts = get_crystal_contacts(p["prot_pdb"], ref_mol, info["pocket"])
        ref_burial = compute_sasa_burial(ref_mol, p["prot_pdb"])
        results[pid] = {}
        for tool in active_tools:
            r = score_experiment(tool, pid, ref_mol, p["prot_pdb"], p["center"], p["lig_sdf"], info["pocket"], ref_contacts, p["smiles"] if info.get("covalent") else None)
            r["ref_burial"] = ref_burial
            results[pid][tool] = r; log_result(tool, r)
    return results, timings

def run_phase2_crossdocking(targets, parsed, active_tools):
    print("\nPhase 2: Cross-Docking (Generalisation)")
    results, timings = {}, {t: [] for t in active_tools}
    for pocket_name, ref_ids in CROSSDOCK_REFERENCES.items():
        pocket_targets = [pid for pid in targets if TARGET_INFO[pid]["pocket"] == pocket_name and not TARGET_INFO[pid].get("covalent")]
        for ref_id in ref_ids:
            if ref_id not in parsed: continue
            p_ref = parsed[ref_id]
            dock_targets, seen = [], set()
            for pid in pocket_targets:
                if pid == ref_id: continue
                lc = parsed[pid]["lig_code"]
                if lc not in seen and lc != parsed[ref_id]["lig_code"]:
                    dock_targets.append(pid); seen.add(lc)
            if not dock_targets: continue
            print(f"\nReference Structure: {ref_id} ({TARGET_INFO[ref_id]['drug']}, {pocket_name})")
            for li, lid in enumerate(dock_targets, 1):
                jid = f"X{ref_id}_{lid}"; p_lig = parsed[lid]
                print(f"\n[{li}/{len(dock_targets)}] {ref_id} + {lid} ({TARGET_INFO[lid]['drug']})")
                ref_sdf = generate_crossdock_reference(p_ref, p_lig)
                if not ref_sdf: continue
                ref_mol = load_mol(ref_sdf)
                if not ref_mol: continue
                if not SKIP_RUN:
                    pc = {"smiles": p_lig["smiles"], "lig_sdf": p_lig["lig_sdf"], "lig_pdb": None, "seq": p_ref["seq"]}
                    for tool in active_tools:
                        elapsed = _run_tool(tool, jid, pc, p_ref["center"], os.path.join(RAW_DIR, f"{ref_id}.pdb"),
                                  pocket_name, p_ref["prot_pdb"])
                        if elapsed >= 1.0: timings[tool].append(elapsed)
                _, xrc = get_crystal_contacts(p_ref["prot_pdb"], ref_mol, pocket_name)
                ref_burial = compute_sasa_burial(ref_mol, p_ref["prot_pdb"])
                pr = {}
                for tool in active_tools:
                    r = score_experiment(tool, jid, ref_mol, p_ref["prot_pdb"], p_ref["center"], p_lig["lig_sdf"], pocket_name, xrc, None)
                    r["ref_burial"] = ref_burial
                    pr[tool] = r; log_result(tool, r)
                results[(ref_id, lid)] = pr
    return results, timings

def _prepare_apo_template(apo_pdb_id, parsed):
    """Downloads and aligns an apo template structure based on a known holo complex."""
    apo_pdb = download_pdb(apo_pdb_id)
    if not apo_pdb: return None
    apo_struct = PDB.PDBParser(QUIET=True).get_structure("apo", apo_pdb)
    apo_seq = max(("".join([AA3TO1.get(r.get_resname(), 'X') for r in ch if PDB.is_aa(r)])
                   for m in apo_struct for ch in m), key=len, default="")
    apo_prot = os.path.join(RAW_DIR, f"{apo_pdb_id}_protein.pdb")
    if not os.path.exists(apo_prot):
        RETAIN = {"GDP","GTP","GNP","GSP","GCP","MG"}
        class PC(Select):
            def accept_residue(self, r):
                return PDB.is_aa(r, standard=True) or r.get_resname().strip() in RETAIN
        io = PDB.PDBIO(); io.set_structure(apo_struct); io.save(apo_prot, PC())

    apo_center = None
    for pid in parsed:
        if TARGET_INFO[pid]["pocket"] == "SII-P" and not TARGET_INFO[pid].get("covalent"):
            al = align_structures(apo_prot, parsed[pid]["prot_pdb"])
            if al:
                R, t, _ = al
                apo_center = R @ np.array(parsed[pid]["center"]) + t
                break
    if apo_center is None:
        cs = []
        for pid in parsed:
            if TARGET_INFO[pid]["pocket"] == "SII-P":
                al = align_structures(apo_prot, parsed[pid]["prot_pdb"])
                if al: R, t, _ = al; cs.append(R @ np.array(parsed[pid]["center"]) + t)
        apo_center = np.mean(cs, axis=0) if cs else None
    if apo_center is None: return None
    return {"pdb_id": apo_pdb_id, "pdb_file": apo_pdb, "prot_pdb": apo_prot,
            "seq": apo_seq, "center": apo_center}

def run_phase3_apo_docking(targets, parsed, active_tools):
    print("\nPhase 3: Apo Docking (Closed Pocket)")

    apo_cache = {}
    needed_mutations = set()
    for pid in targets:
        info = TARGET_INFO[pid]
        if info["pocket"] == "SII-P" and not info.get("covalent"):
            mut = info["mutation"].split("(")[0]
            needed_mutations.add(mut)
    for mut in sorted(needed_mutations):
        apo_pdb_id = APO_TEMPLATES.get(mut)
        if not apo_pdb_id:
            print(f"  Warning: Missing apo template for mutation {mut}, bypassing target.")
            continue
        if apo_pdb_id not in apo_cache:
            print(f"  Fetching apo template {apo_pdb_id} for {mut}...")
            tmpl = _prepare_apo_template(apo_pdb_id, parsed)
            if tmpl: apo_cache[apo_pdb_id] = tmpl
            else: print(f"  Warning: Failed to prepare apo template {apo_pdb_id}.")

    apo_targets = [pid for pid in targets if TARGET_INFO[pid]["pocket"] == "SII-P" and not TARGET_INFO[pid].get("covalent")]
    seen, unique = {}, []
    for pid in apo_targets:
        lc = parsed[pid]["lig_code"]
        if lc not in seen: seen[lc] = pid; unique.append(pid)
    print(f"  Unique ligands: {len(unique)} across {len(needed_mutations)} mutations ({', '.join(sorted(needed_mutations))})")

    results, timings = {}, {t: [] for t in active_tools}
    for idx, lid in enumerate(unique, 1):
        p_lig = parsed[lid]; info = TARGET_INFO[lid]
        mut = info["mutation"].split("(")[0]
        apo_pdb_id = APO_TEMPLATES.get(mut)
        if not apo_pdb_id or apo_pdb_id not in apo_cache:
            print(f"\n[{idx}/{len(unique)}] Skipped {lid} ({info['drug']}) - no template for {mut}")
            continue
        apo = apo_cache[apo_pdb_id]
        jid = f"APO{apo_pdb_id}_{lid}"
        print(f"\n[{idx}/{len(unique)}] APO({apo_pdb_id}/{mut}) + {lid} ({info['drug']})")
        ref_sdf = os.path.join(ALIGNED_REFS_DIR, f"apo{apo_pdb_id}_ref_{lid}.sdf")
        if not os.path.exists(ref_sdf) or os.path.getsize(ref_sdf) < 50:
            al = align_structures(apo["prot_pdb"], p_lig["prot_pdb"])
            if al:
                R, t, _ = al; mol = load_mol(p_lig["lig_sdf"]) or load_mol(p_lig["lig_pdb"])
                if mol: mol = transform_mol(mol, R, t); w = Chem.SDWriter(ref_sdf); w.write(mol); w.close()
        ref_mol = load_mol(ref_sdf) if os.path.exists(ref_sdf) else None
        if not ref_mol: continue
        if not SKIP_RUN:
            pc = {"smiles": p_lig["smiles"], "lig_sdf": p_lig["lig_sdf"], "lig_pdb": None, "seq": apo["seq"]}
            for tool in active_tools:
                elapsed = _run_tool(tool, jid, pc, apo["center"], apo["pdb_file"], "SII-P", apo["prot_pdb"], apo["seq"])
                if elapsed >= 1.0: timings[tool].append(elapsed)
        _, arc = get_crystal_contacts(apo["prot_pdb"], ref_mol, "SII-P")
        ref_burial = compute_sasa_burial(ref_mol, apo["prot_pdb"])
        pr = {}
        for tool in active_tools:
            r = score_experiment(tool, jid, ref_mol, apo["prot_pdb"], apo["center"], p_lig["lig_sdf"], "SII-P", arc, None)
            r["ref_burial"] = ref_burial
            pr[tool] = r; log_result(tool, r)
        results[lid] = pr
    return results, timings

def save_csvs(redock, xdock, apo, targets, active_tools, timings=None):
    os.makedirs(CSV_DIR, exist_ok=True)

    COLUMNS = [
        "Phase", "Job_ID", "PDB_ID", "Ref_PDB", "Lig_PDB", "Mutation", "Pocket", "Drug",
        "Covalent", "Selectivity", "Source", "Tool",
        "RMSD", "IFP_Tanimoto", "Contact_Recovery_Pct", "Burial_Frac", "Burial_Similarity",
        "Validity_Pct", "Clash_Pct", "Pocket_CA_RMSD", "TM_Score",
    ]

    def _fmt(val):
        if val is None: return ""
        if isinstance(val, float): return round(val, 4)
        return val

    def _burial_sim(r):
        pred = r.get("burial_frac")
        ref = r.get("ref_burial")
        if pred is not None and ref is not None:
            return round(1.0 - abs(pred - ref), 4)
        return ""

    def _write_row(w, phase, job_id, pdb_id, ref_pdb, lig_pdb, tool, r):
        info = TARGET_INFO.get(pdb_id) or TARGET_INFO.get(lig_pdb) or {}
        w.writerow([
            phase, job_id, pdb_id, ref_pdb, lig_pdb,
            info.get("mutation", ""), info.get("pocket", ""), info.get("drug", ""),
            info.get("covalent", ""), info.get("selectivity", ""), info.get("source", ""),
            tool,
            _fmt(r.get("rmsd")), _fmt(r.get("ifp_tanimoto")),
            _fmt(r.get("contact_recovery")), _fmt(r.get("burial_frac")), _burial_sim(r),
            _fmt(r.get("validity_pct")), _fmt(r.get("clash_pct")), _fmt(r.get("pocket_ca_rmsd")),
            _fmt(r.get("tm_score")),
        ])

    master_path = os.path.join(CSV_DIR, "all_results.csv")
    with open(master_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)

        for pid in targets:
            for tool in active_tools:
                r = redock.get(pid, {}).get(tool, {})
                if not isinstance(r, dict): continue
                _write_row(w, "P1_Redock", pid, pid, pid, pid, tool, r)

        for key in xdock:
            if isinstance(key, tuple):
                ref_id, lig_id = key
                jid = f"X{ref_id}_{lig_id}"
            else:
                ref_id, lig_id, jid = key, key, str(key)
            for tool in active_tools:
                r = xdock[key].get(tool, {})
                if not isinstance(r, dict): continue
                _write_row(w, "P2_Crossdock", jid, ref_id, ref_id, lig_id, tool, r)

        for lid in apo:
            info = TARGET_INFO.get(lid, {})
            mut = info.get("mutation", "").split("(")[0]
            apo_pdb_id = APO_TEMPLATES.get(mut, "UNK")
            jid = f"APO{apo_pdb_id}_{lid}"
            for tool in active_tools:
                r = apo[lid].get(tool, {})
                if not isinstance(r, dict): continue
                _write_row(w, "P3_Apo", jid, apo_pdb_id, apo_pdb_id, lid, tool, r)

    print(f"\nRaw results exported to {master_path}")

    summary_path = os.path.join(CSV_DIR, "overall_comparison.csv")
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Tool",
                     "P1_RMSD","P1_IFT","P1_CR","P1_BurialSim","P1_Valid","P1_N",
                     "P2_RMSD","P2_IFT","P2_CR","P2_Valid","P2_N",
                     "P3_RMSD","P3_IFT","P3_CR","P3_BurialSim","P3_Valid","P3_N",
                     "Avg_Time_Sec","Min_Time_Sec","Max_Time_Sec","N_Runs"])
        for tool in active_tools:
            row = [tool]
            for pr, ig in [(redock, targets), (xdock, xdock.keys()), (apo, apo.keys())]:
                rs, cs, ifs, bsims, vs = [], [], [], [], []
                for k in ig:
                    r = pr.get(k, {}).get(tool, {})
                    if isinstance(r, dict):
                        if r.get("rmsd") is not None: rs.append(r["rmsd"])
                        if r.get("contact_recovery") is not None: cs.append(r["contact_recovery"])
                        if r.get("ifp_tanimoto") is not None: ifs.append(r["ifp_tanimoto"])
                        bf, rb = r.get("burial_frac"), r.get("ref_burial")
                        if bf is not None and rb is not None: bsims.append(1.0 - abs(bf - rb))
                        if r.get("validity_pct") is not None: vs.append(r["validity_pct"])
                row.extend([
                    format_rmsd(calc_avg(rs)), format_frac(calc_avg(ifs)),
                    format_pct(calc_avg(cs)),
                    format_frac(calc_avg(bsims)) if pr is not xdock else "",
                    format_pct(calc_avg(vs)),
                    len(rs),
                ])
            times = timings.get(tool, []) if timings else []
            if times:
                row.extend([round(np.mean(times), 1), round(min(times), 1),
                           round(max(times), 1), len(times)])
            else:
                row.extend(["", "", "", 0])
            w.writerow(row)
    print(f"Summary metrics exported to {summary_path}")

def main():
    targets = TARGET_LIST if TARGET_LIST else list(TARGET_INFO.keys())
    active_tools = list(DOCKING_TOOLS) if DOCKING_ONLY else list(ALL_TOOLS)

    print("KRAS Docking Benchmark Pipeline")
    print(f"Total targets: {len(targets)} | Active tools: {', '.join(active_tools)}")
    phases = []
    if RUN_PHASE1: phases.append("Phase 1 (Redock)")
    if RUN_PHASE2: phases.append("Phase 2 (Crossdock)")
    if RUN_PHASE3: phases.append("Phase 3 (Apo)")
    print(f"Enabled phases: {', '.join(phases)}")

    if not check_prerequisites(): sys.exit(1)
    for d in [LOG_DIR, RAW_DIR, ALIGNED_REFS_DIR]:
        os.makedirs(d, exist_ok=True)

    print("\nInitialising structural downloads and parsing...")
    parsed = {}
    for pid in targets:
        print(f"  Processing {pid}: ", end="", flush=True)
        p = parse_target(pid)
        if p:
            parsed[pid] = p
            cov = " [Covalent Target]" if TARGET_INFO[pid].get("covalent") else ""
            print(f"Bound to {p['lig_code']} ({p['n_heavy']} heavy atoms), length: {p['n_residues']} residues{cov}")
        else: print("Parsing failed.")
    targets = [t for t in targets if t in parsed]

    r1, r2, r3 = {}, {}, {}
    all_timings = {t: [] for t in active_tools}
    if RUN_PHASE1:
        r1, t1 = run_phase1_redocking(targets, parsed, active_tools)
        for t in active_tools: all_timings[t].extend(t1.get(t, []))
    if RUN_PHASE2:
        r2, t2 = run_phase2_crossdocking(targets, parsed, active_tools)
        for t in active_tools: all_timings[t].extend(t2.get(t, []))
    if RUN_PHASE3:
        r3, t3 = run_phase3_apo_docking(targets, parsed, active_tools)
        for t in active_tools: all_timings[t].extend(t3.get(t, []))

    print("\nCollating metrics and saving data sets...")
    save_csvs(r1, r2, r3, targets, active_tools, all_timings)

    print("\nBenchmark execution completed successfully.")

if __name__ == "__main__":
    main()