import math
import random
import ROOT
import array
import matplotlib.pyplot as plt
import numpy as np

def evaluate_distance(FileName, x_model_file, y_model_file):
    ROOT.TMVA.Tools.Instance()
    DataFile = ROOT.TFile(FileName);
    DataTree = DataFile.Get("TrackWithinCrossStripDetectorTMVA");

    IgnoredBranches = [ 'SimulationID' ]
    Branches = DataTree.GetListOfBranches()
    for branch in Branches:
        print(branch.GetName())
    VariableMap = {}

    reader_x = ROOT.TMVA.Reader("!Color:!Silent")
    reader_y = ROOT.TMVA.Reader("!Color:!Silent")

    for Name in IgnoredBranches:
        VariableMap[Name] = array.array('f', [0])
        DataTree.SetBranchAddress(Name, VariableMap[Name])
        reader_x.AddSpectator(Name, VariableMap[Name])
        reader_y.AddSpectator(Name, VariableMap[Name])

    xDim = 0
    yDim = 0
    for B in list(Branches):
        if not B.GetName() in IgnoredBranches:
          if not B.GetName().startswith("Result"):
            VariableMap[B.GetName()] = array.array('f', [0])
            reader_x.AddVariable(B.GetName(), VariableMap[B.GetName()])
            reader_y.AddVariable(B.GetName(), VariableMap[B.GetName()])
            DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])
            if "XStripID" in B.GetName():
              xDim += 1
            if "YStripID" in B.GetName():
              yDim += 1

    # Add the target variables
    VariableMap["ResultPositionX"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultPositionX", VariableMap["ResultPositionX"])
    reader_y.AddSpectator("ResultPositionX", VariableMap["ResultPositionX"])

    VariableMap["ResultPositionY"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultPositionY", VariableMap["ResultPositionY"])
    reader_x.AddSpectator("ResultPositionY", VariableMap["ResultPositionY"])

    VariableMap["ResultDirectionX"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultDirectionX", VariableMap["ResultDirectionX"])
    reader_x.AddSpectator("ResultDirectionX", VariableMap["ResultDirectionX"])
    reader_y.AddSpectator("ResultDirectionX", VariableMap["ResultDirectionX"])

    VariableMap["ResultDirectionY"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultDirectionY", VariableMap["ResultDirectionY"])
    reader_x.AddSpectator("ResultDirectionY", VariableMap["ResultDirectionY"])
    reader_y.AddSpectator("ResultDirectionY", VariableMap["ResultDirectionY"])
    
    VariableMap["ResultPositionZ"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultPositionZ", VariableMap["ResultPositionZ"])
    reader_x.AddSpectator("ResultPositionZ", VariableMap["ResultPositionZ"])
    reader_y.AddSpectator("ResultPositionZ", VariableMap["ResultPositionZ"])
    VariableMap["ResultDirectionZ"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultDirectionZ", VariableMap["ResultDirectionZ"])
    reader_x.AddSpectator("ResultDirectionZ", VariableMap["ResultDirectionZ"])
    reader_y.AddSpectator("ResultDirectionZ", VariableMap["ResultDirectionZ"])

    reader_x.BookMVA("BDTD", x_model_file)
    reader_y.BookMVA("BDTD", y_model_file)

    n_entries = DataTree.GetEntries()
    difference = []
    for i in range(min(10000, n_entries)):
        DataTree.GetEntry(i)
        InputPosX = VariableMap["ResultPositionX"][0]
        InputPosY = VariableMap["ResultPositionY"][0]

        predicted_x = reader_x.EvaluateMVA("BDTD")
        predicted_y = reader_y.EvaluateMVA("BDTD")

        error_x = InputPosX - predicted_x
        error_y = InputPosY - predicted_y
        # difference.append((error_x**2 + error_y**2)**(1/2))
        difference.append((error_x**2)**(1/2))

    DataFile.Close()
    return difference

def evaluate_direction(FileName, x_model_file, y_model_file):
    ROOT.TMVA.Tools.Instance()
    DataFile = ROOT.TFile(FileName);
    DataTree = DataFile.Get("TrackWithinCrossStripDetectorTMVA");

    IgnoredBranches = [ 'SimulationID' ]
    Branches = DataTree.GetListOfBranches()
    for branch in Branches:
        print(branch.GetName())
    VariableMap = {}

    reader_x = ROOT.TMVA.Reader("!Color:!Silent")
    reader_y = ROOT.TMVA.Reader("!Color:!Silent")

    for Name in IgnoredBranches:
        VariableMap[Name] = array.array('f', [0])
        DataTree.SetBranchAddress(Name, VariableMap[Name])
        reader_x.AddSpectator(Name, VariableMap[Name])
        reader_y.AddSpectator(Name, VariableMap[Name])

    xDim = 0
    yDim = 0
    for B in list(Branches):
        if not B.GetName() in IgnoredBranches:
          if not B.GetName().startswith("Result"):
            VariableMap[B.GetName()] = array.array('f', [0])
            reader_x.AddVariable(B.GetName(), VariableMap[B.GetName()])
            reader_y.AddVariable(B.GetName(), VariableMap[B.GetName()])
            DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])
            if "XStripID" in B.GetName():
              xDim += 1
            if "YStripID" in B.GetName():
              yDim += 1

    # Add the target variables
    VariableMap["ResultPositionX"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultPositionX", VariableMap["ResultPositionX"])
    reader_x.AddSpectator("ResultPositionX", VariableMap["ResultPositionX"])
    reader_y.AddSpectator("ResultPositionX", VariableMap["ResultPositionX"])

    VariableMap["ResultPositionY"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultPositionY", VariableMap["ResultPositionY"])
    reader_x.AddSpectator("ResultPositionY", VariableMap["ResultPositionY"])
    reader_y.AddSpectator("ResultPositionY", VariableMap["ResultPositionY"])

    VariableMap["ResultDirectionX"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultDirectionX", VariableMap["ResultDirectionX"])
    reader_y.AddSpectator("ResultDirectionX", VariableMap["ResultDirectionX"])

    VariableMap["ResultDirectionY"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultDirectionY", VariableMap["ResultDirectionY"])
    reader_x.AddSpectator("ResultDirectionY", VariableMap["ResultDirectionY"])

    VariableMap["ResultPositionZ"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultPositionZ", VariableMap["ResultPositionZ"])
    reader_x.AddSpectator("ResultPositionZ", VariableMap["ResultPositionZ"])
    reader_y.AddSpectator("ResultPositionZ", VariableMap["ResultPositionZ"])
    VariableMap["ResultDirectionZ"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultDirectionZ", VariableMap["ResultDirectionZ"])
    reader_x.AddSpectator("ResultDirectionZ", VariableMap["ResultDirectionZ"])
    reader_y.AddSpectator("ResultDirectionZ", VariableMap["ResultDirectionZ"])

    reader_x.BookMVA("BDTD", x_model_file)
    reader_y.BookMVA("BDTD", y_model_file)

    n_entries = DataTree.GetEntries()
    angles = []
    print("ResultDirectionX:", VariableMap["ResultDirectionX"])
    print("ResultDirectionY:", VariableMap["ResultDirectionY"])
    for i in range(min(10000, n_entries)):
        DataTree.GetEntry(i)
        OutputDirX = VariableMap["ResultDirectionX"][0]
        OutputDirY = VariableMap["ResultDirectionY"][0]

        predicted_x = reader_x.EvaluateMVA("BDTD")
        predicted_y = reader_y.EvaluateMVA("BDTD")
        DotProduct = predicted_x*OutputDirX + predicted_y*OutputDirY
        Mag1 = math.sqrt(predicted_x**2 + predicted_y**2)
        Mag2 = math.sqrt(OutputDirX**2 + OutputDirY**2)
        Direction = math.degrees(math.acos(DotProduct / Mag1 / Mag2)) * random.choice([1, -1])

        # difference.append((error_x**2 + error_y**2)**(1/2))
        angles.append(Direction)
        HistDirection.Fill(Direction)

    DataFile.Close()
    return angles

if __name__ == "__main__":
    test_data_file = "GeDSSDElectronTracking.x3.y3.electrontrackingwithcrossstripdetector.tmva.root"  # or any file in self.AllFileNames
    x_pos_file =  "Results/weights/TMVARegression_ResultPositionX_BDTD.weights.xml"
    y_pos_file = "Results/weights/TMVARegression_ResultPositionY_BDTD.weights.xml"
    x_dir_file =  "Results/weights/TMVARegression_ResultDirectionX_BDTD.weights.xml"
    y_dir_file = "Results/weights/TMVARegression_ResultDirectionY_BDTD.weights.xml"

    # Evaluate the models
    distance = evaluate_distance(test_data_file, x_pos_file, y_pos_file)

    plt.hist(distance, bins=100, edgecolor='skyblue')

    plt.xlabel('Distance (cm)')
    plt.ylabel('Count')
    plt.title('Distance vs. Count')
    plt.show()

    HistDirection = ROOT.TH1F("Direction", "Direction", 180, -180, 180)
    HistDirection.SetLineColor(ROOT.kRed)
    HistDirection.SetXTitle("Difference between real and reconstructed recoil electron direction in degrees")
    HistDirection.SetYTitle("counts")
    HistDirection.SetMinimum(0)
    CanvasDirection = ROOT.TCanvas()
    CanvasDirection.SetTitle("Direction")
    HistDirection.Draw()
    CanvasDirection.Update()

    Direction = evaluate_direction(test_data_file, x_dir_file, y_dir_file)

    # Fit
    Fit = ROOT.TF1("Fit", "[0] + gaus(1)", -180, 180)
    # Set initial parameters for the Gaussians
    Fit.SetParameters(20, 2900, 0, 70)  # Initial counts for each Gaussian

    # Fit the histogram with the function
    HistDirection.Fit(Fit, "R")
    Fit.SetLineColor(ROOT.kGreen+3)
    Fit.Draw("SAME")
    CanvasDirection.Update()
    

    print("\nResult:")
    print("Peak: {}".format(Fit.GetParameter(1)))
    print("Offset: {}".format(Fit.GetParameter(0)))
    print("Peak/Offset: {}".format(Fit.GetParameter(1)/Fit.GetParameter(0)))
    ROOT.gApplication.Run()
