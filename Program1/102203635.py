import pandas as pd
import numpy as np
import sys

def main():
    if len(sys.argv) < 5:
        return
    inp = sys.argv[1]
    weight = list(map(float, sys.argv[2].split(',')))
    impact = sys.argv[3].split(',')
    result = sys.argv[4]

    try:
        data = pd.read_csv(inp)
    except FileNotFoundError:
        print("File not found")
        return

    if len(weight) != len(impact) or len(impact) != len(data.columns) - 1:
        print("Error: The number of weight and impact must match the number of criteria (columns).")
        return

    if not all(imp in ['+', '-'] for imp in impact):
        print("Error: impact must be either '+' or '-'.")
        return

    try:
        mat = data.iloc[:, 1:].to_numpy(dtype=float)
    except ValueError:
        print("Error: All criteria values must be numeric.")
        return

    norm_data = mat / np.sqrt((mat**2).sum(axis=0)) * weight
    ideal_best, ideal_worst = [], []
    for i, impact in enumerate(impact):
        if impact == '+':
            ideal_best.append(np.max(norm_data[:, i]))
            ideal_worst.append(np.min(norm_data[:, i]))
        elif impact == '-':
            ideal_best.append(np.min(norm_data[:, i]))
            ideal_worst.append(np.max(norm_data[:, i]))
    ideal_best, ideal_worst= np.array(ideal_best), np.array(ideal_worst)
    distances_best = np.sqrt(((norm_data - ideal_best)**2).sum(axis=1))
    distances_worst = np.sqrt(((norm_data - ideal_worst)**2).sum(axis=1))
    scores = distances_worst / (distances_best + distances_worst)
    rankings = scores.argsort()[::-1] + 1

    # Save Topsis to score and save the final result
    data['Topsis Score'] = scores
    data['Rank'] = rankings
    data.to_csv(result, index=False)
    print(f"Result file successfully saved")

if __name__ == "__main__":
    main()
