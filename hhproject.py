from knnhh import KNNHH
from rfhh import RFHH
from dqhh import DQHH 
from ppohh import PPOHH
from kp import KP
from bpp import BPP
from vcp import VCP
from ffp import FFP
from typing import List
from phermes import HyperHeuristic
import os
import numpy as np
import pandas as pd
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def characterize(domain : str, folder : str, features : List[str]):
  """
    Characterizes the instances contained in a folder.
  """
  text = ""
  files = os.listdir(folder)
  files.sort()
  text += "INSTANCE\t" + "\t".join(features) + "\r\n"
  for file in files:    
    if domain == "KP":
      problem = KP(folder + "/" + file)
    elif domain == "BPP":
      problem = BPP(folder + "/" + file)
    elif domain == "VCP":
      problem = VCP(folder + "/" + file)
    elif domain == "FFP":
      problem = FFP(folder + "/" + file)
    else:
      raise Exception("Problem domain '" + domain + "' is not recognized by the system.") 
    text += file + "\t"
    for f in features:
      text += str(round(problem.getFeature(f), 3)) + "\t"
    text += "\r\n"  
  print(text)

def solve(domain : str, folder : str, heuristics : List[str]):
  text = ""
  files = os.listdir(folder)
  files.sort()
  text += "INSTANCE\t" + "\t".join(heuristics) + "\r\n"  
  for i in range(len(files)):    
    text += files[i] + "\t"   
    for h in heuristics:
      if domain == "KP":
        problem = KP(folder + "/" + files[i])
      elif domain == "BPP":
        problem = BPP(folder + "/" + files[i])
      elif domain == "VCP":
        problem = VCP(folder + "/" + files[i])
      elif domain == "FFP":
        np.random.seed(i)
        problem = FFP(folder + "/" + files[i])
      else:
        raise Exception("Problem domain '" + domain + "' is not recognized by the system.")      
      problem.solve(h)
      text += str(round(problem.getObjValue(), 3)) + "\t"
    text += "\r\n"  
  print(text)

def solveHH(domain: str, folder: str, hyperHeuristic: HyperHeuristic, hhstr: str, n: int):
    files = os.listdir(folder)
    files.sort()
    results = []

    for file in files:
        if domain == "KP":
            problem = KP(folder + "/" + file)
        elif domain == "BPP":
            problem = BPP(folder + "/" + file)
        elif domain == "VCP":
            problem = VCP(folder + "/" + file)
        elif domain == "FFP":
            problem = FFP(folder + "/" + file)
        else:
            raise Exception("Problem domain '" + domain + "' is not recognized by the system.")
        
        problem.solveHH(hyperHeuristic)
        results.append((file, round(problem.getObjValue(), 3)))

    df = pd.DataFrame(results, columns=['INSTANCE', f'HH_{hhstr}_{n}'])

    file_name = hhstr + ".csv"
    if n == 1:
        if os.path.exists(file_name):
            os.remove(file_name)  # Delete the file if it exists
        existing_df = df
    else:
        if os.path.exists(file_name):
            existing_df = pd.read_csv(file_name)
            existing_df = existing_df.merge(df, on='INSTANCE', how='left')
        else:
            existing_df = df

    existing_df.to_csv(file_name, index=False)
# Trains and tests a KNN hyper-heuristic on any of the given problem domains.
# To test it, uncomment the corresponding code.



def calculate_similarity(n):
    # Initialize arrays to store similarity percentages for each method
    knnhh_similarities = []
    dqhh_similarities = []
    rfhh_similarities = []

    # Specify the columns you want to import
    columns2Import = ['INSTANCE', 'ORACLE']

    # Read the CSV file, importing only the specified columns
    groundTruthDF = pd.read_csv('Instances/VCP/VCP-Test I.csv', usecols=columns2Import)

    dqhhDF = pd.read_csv('DQHH.csv')
    knnhhDF = pd.read_csv('KNNHH.csv')
    rfhhDF = pd.read_csv('RFHH.csv')

    vcpValDF = pd.merge(groundTruthDF, knnhhDF, on='INSTANCE')
    vcpValDF = pd.merge(vcpValDF, dqhhDF, on='INSTANCE')
    vcpValDF = pd.merge(vcpValDF, rfhhDF, on='INSTANCE')

    # Verify the presence of the columns before proceeding
    required_columns = ['ORACLE']
    for method in ['DQHH']:
        required_columns += [f'HH_{method}_{i}' for i in range(1, n)]

    for column in required_columns:
        if column not in vcpValDF.columns:
            raise KeyError(f"Column '{column}' is missing in the DataFrame")

    for method in ['DQHH']:
        method_similarities = []
        for i in range(1, n):
            col_name = f'HH_{method}_{i}'

            # Calculate the number of matches
            matches = (vcpValDF['ORACLE'] == vcpValDF[col_name]).sum()

            # Calculate the  similarity
            similarity_percent = (matches / len(vcpValDF)) * 100

            # Append similarity percentage to method_similarities array
            method_similarities.append(similarity_percent)

            if method =='DQHH':
              dqhh_similarities.append(method_similarities)
        
    # Calculate the number of matches for HH_KNNHH
    matches_KNNHH = (vcpValDF['ORACLE'] == vcpValDF['HH_KNNHH_1']).sum()

    # Calculate the percentage similarity for HH_RFHH
    knnhh_similarities = (matches_KNNHH / len(vcpValDF)) * 100


    # Calculate the number of matches for HH_RFHH
    matches_RFHH = (vcpValDF['ORACLE'] == vcpValDF['HH_RFHH_1']).sum()

    # Calculate the percentage similarity for HH_RFHH
    rfhh_similarities = (matches_RFHH / len(vcpValDF)) * 100

    # Return the similarity percentages for each method
    return knnhh_similarities, dqhh_similarities, rfhh_similarities



def main():
    parser = argparse.ArgumentParser(description='Run VCP recommender system.')
    parser.add_argument('n', type=int, help='Number of test iterations')
    args = parser.parse_args()
    
    features = ["DENSITY", "MAX_DEG", "MIN_DEG"]
    heuristics = ["DEF", "DEG", "COL_DEG", "UNCOL_DEG"]
      
   
    print("\nCALCULATING WITH KNNHH")
    hh = KNNHH(features, heuristics)
    hh.train("Instances/VCP/VCP-Training.csv")
    solveHH("VCP", "Instances/VCP/Test I", hh,"KNNHH",1)
    
    print("\nCALCULATING WITH RFHH")
    hh = RFHH(features, heuristics)
    hh.train("Instances/VCP/VCP-Training.csv")
    solveHH("VCP", "Instances/VCP/Test I", hh,"RFHH",1)
  
    for i in range(1, args.n + 1):
      print(f"\nCALCULATING WITH DQHH {i}")
      hh = DQHH(features, heuristics)
      hh.train("Instances/VCP/VCP-Training.csv")
      solveHH("VCP", "Instances/VCP/Test I", hh, "DQHH", i)

    knnhh_similarities, dqhh_similarities, rfhh_similarities = calculate_similarity(args.n + 1)
    #print(knnhh_similarities, rfhh_similarities, np.mean(dqhh_similarities))
    #print(dqhh_similarities)
    y = np.array([item[1] for item in dqhh_similarities])
    x = np.arange(len(y))

    print("Length of x:", len(x))
    print("Length of y:", len(y))

    # Create the scatter plot
    plt.scatter(x, y, label='Data Points')

    # Add labels, title, and legend for clarity
    plt.xlabel('Index')
    plt.ylabel('DQHH Similarities')
    plt.title('Scatter Plot of DQHH Similarities')
    plt.legend()

    # Display the plot
    plt.show() 

    # Calculate means
    knnhh_mean = round(knnhh_similarities,2)
    dqhh_mean = round(np.mean(dqhh_similarities),2)
    rfhh_mean = round(rfhh_similarities,2)

    # Create comparison table
    comparison_table = {
        "Method": ["KNNHH", "DQHH", "RFHH"],
        "Mean Similarity": [knnhh_mean, dqhh_mean, rfhh_mean]
    }

    # Display comparison table
    df_comparison = pd.DataFrame(comparison_table)
    print(df_comparison)


    

if __name__ == "__main__":
    main()