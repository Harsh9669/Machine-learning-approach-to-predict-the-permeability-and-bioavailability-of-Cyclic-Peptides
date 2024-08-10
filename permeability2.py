import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
model_path = 'D:/C/trained_model2.sav'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to calculate all molecular descriptors
def calculate_all_descriptors(mol):
    descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]
    descriptors = {}
    for desc_name, desc_func in Descriptors.descList:
        try:
            descriptors[desc_name] = desc_func(mol)
        except Exception as e:
            descriptors[desc_name] = np.nan
    return descriptors

# Function to prepare model input from descriptors
def prepare_model_input(descriptors):
    selected_descriptors = ['MaxEStateIndex', 'MinEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'MaxPartialCharge', 
                     'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 
                     'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_LOGPHI', 
                     'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi1', 
                     'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 
                     'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 
                     'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 
                     'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 
                     'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 
                     'SlogP_VSA5', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 
                     'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 
                     'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 
                     'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 
                     'HeavyAtomCount', 'NHOHCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 
                     'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 
                     'NumAromaticRings', 'NumRotatableBonds', 'NumSaturatedHeterocycles', 
                     'NumSaturatedRings', 'RingCount', 'MolLogP', 'fr_Al_OH', 'fr_Ar_N', 'fr_Ar_OH', 'fr_C_O', 
                     'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_allylic_oxid', 
                     'fr_amide', 'fr_aryl_methyl', 'fr_bicyclic', 'fr_ester', 'fr_ether', 'fr_halogen', 
                     'fr_methoxy', 'fr_morpholine', 'fr_piperdine', 'fr_sulfide', 'fr_unbrch_alkane'
    ]

    return np.array([[descriptors[desc] for desc in selected_descriptors]])

# Add a sidebar with navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Description", "Contact"])

# Home page
if page == "Home":
    st.title("Predict Membrane Permeability of Cyclic Peptides")
    st.subheader("Calculate molecular descriptors from SMILES string")

    # User input: SMILES string or list of SMILES strings
    smiles_input = st.text_area("Enter SMILES string(s), separated by new lines:", "")

    if smiles_input:
        smiles_list = smiles_input.strip().split("\n")
        results = []
        descriptor_data = []

        for smiles in smiles_list:
            try:
                # Convert SMILES string to RDKit molecule object
                mol = Chem.MolFromSmiles(smiles)

                if mol:
                    # Calculate all descriptors
                    descriptors = calculate_all_descriptors(mol)

                    # Prepare the input for the model using selected descriptors
                    model_input = prepare_model_input(descriptors)

                    # Make prediction
                    prediction = model.predict(model_input)[0]
                    if prediction == 0:
                        permeability = "good"
                    elif prediction == 1:
                        permeability = "impermeable"
                    else:
                        permeability = "moderate"

                    # Store SMILES, prediction, and descriptors
                    results.append((smiles, permeability))
                    descriptor_data.append(descriptors)

                else:
                    results.append((smiles, "Invalid SMILES"))
                    descriptor_data.append({desc: np.nan for desc in calculate_all_descriptors(Chem.MolFromSmiles('C'))})  # Placeholder for 208 NaNs

            except Exception as e:
                results.append((smiles, f"Error: {e}"))
                descriptor_data.append({desc: np.nan for desc in calculate_all_descriptors(Chem.MolFromSmiles('C'))})  # Placeholder for 208 NaNs

        # Create a DataFrame with SMILES, Permeability, and Descriptors
        results_df = pd.DataFrame(results, columns=["SMILES", "Permeability"])
        descriptors_df = pd.DataFrame(descriptor_data)

        # Combine the results and descriptors into one DataFrame
        combined_df = pd.concat([results_df, descriptors_df], axis=1)

        # Display results
        st.write("### Predictions and Descriptors")
        st.dataframe(combined_df)



# # Home page
# if page == "Home":
#     st.title("Predict Membrane Permeability of Cyclic Peptides")
#     st.subheader("Calculate molecular descriptors from SMILES string")

#     # User input: SMILES string or list of SMILES strings
#     smiles_input = st.text_area("Enter SMILES string(s), separated by new lines:", "")

#     if smiles_input:
#         smiles_list = smiles_input.strip().split("\n")
#         results = []

#         for smiles in smiles_list:
#             try:
#                 # Convert SMILES string to RDKit molecule object
#                 mol = Chem.MolFromSmiles(smiles)

#                 if mol:
#                     # Calculate all descriptors
#                     descriptors = calculate_all_descriptors(mol)

#                     # Prepare the input for the model using selected descriptors
#                     model_input = prepare_model_input(descriptors)

#                     # Make prediction
#                     prediction = model.predict(model_input)[0]
#                     if prediction == 0:
#                         permeability = "good"
#                     elif prediction == 1:
#                         permeability = "impermeable"
#                     else:
#                         permeability = "moderate"

#                     results.append((smiles, permeability, descriptors))
#                 else:
#                     results.append((smiles, "Invalid SMILES", {}))

#             except Exception as e:
#                 results.append((smiles, f"Error: {e}", {}))

#         # Display results
#         st.write("### Predictions")
#         results_df = pd.DataFrame(results, columns=["SMILES", "Permeability", "Descriptors"])
#         st.dataframe(results_df[["SMILES", "Permeability"]])

#         # Optionally display detailed descriptors
#         if st.checkbox("Show Descriptors"):
#             st.write("### Molecular Descriptors")
#             for idx, row in results_df.iterrows():
#                 st.write(f"**SMILES:** {row['SMILES']}")
#                 desc_df = pd.DataFrame(row["Descriptors"].items(), columns=["Descriptor", "Value"])
#                 st.dataframe(desc_df)
#                 st.write("---")

# Description page
elif page == "Description":
    st.title("Description")
    st.write("### Model Details")
    st.write("This section provides details about the machine learning model used.")

    # Model performance metrics
    performance_metrics = {
        "Metric": ["Accuracy", "F1 Score"],
        "Score": [0.895, 0.889]
    }
    performance_df = pd.DataFrame(performance_metrics)
    st.write("#### Model Performance")
    st.table(performance_df)

    # Permeability criteria
    permeability_criteria = {
        "Permeability range": ["Good", "Moderate/Low", "Impermeable"],
        "Criteria": ["LogP >= -5", "-7 < LogP < -5", "LogP < -7"]
    }
    permeability_df = pd.DataFrame(permeability_criteria)
    st.write("#### Permeability Criteria")
    st.table(permeability_df)

    st.title("Database")
    st.write("This section includes information about the molecular database.")
    st.write("CycPeptMPDB (Cyclic Peptide Membrane Permeability Database) is the largest web-accessible database of membrane permeability of cyclic peptide. The latest version provides the information for 7,334 structurally diverse cyclic peptides collected from 47 publications. These cyclic peptides are composed of 312 types Monomers (substructures). ")
    st.write("### Source of Training Dataset")
    st.write("[Cyclic Peptide Database](http://cycpeptmpdb.com/)")

    df = pd.read_csv("CycPeptMPDB_Peptide.csv")
    st.write(df.head())

# Contact page
elif page == "Contact":
    st.title("Contact")
    st.write("You can reach me at:")
    st.write("Email: Hj728490@gmail.com")
    st.write("Mobile: +91 9024990040")
    st.write("We are open to any Feedbacks, suggestions and contributions", style="center")
