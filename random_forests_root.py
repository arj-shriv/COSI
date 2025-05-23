import ROOT
import array
import sys
import math
import random
import os
 

import itertools

class GeDSSDElectronTracking:
  #init the class
  def __init__(self, FileName, OutputPrefix, Layout, MaximumEvents, UseAll):
    self.FileName = FileName
    self.Layout = Layout
    self.OutputPrefix = OutputPrefix
    self.MaximumEvents = MaximumEvents
    self.Methods = "BDTD"
    self.UseAll = UseAll

    self.FileTag = "TrackWithinCrossStripDetectorTMVA"

    print(self.FileName)

    self.AllFileNames = []
    if self.UseAll == True:
      Index = self.FileName.find('.x')
      BaseName = self.FileName[:Index]
      for x in range(1, 10):
        for y in range(1, 10):
          if x == 1 and y == 1:
            continue
          NewFile = BaseName + ".x" + str(x) + ".y" + str(y) + ".electrontrackingwithcrossstripdetector.tmva.root"
          print(NewFile)
          if os.path.exists(NewFile):
            self.AllFileNames.append(NewFile)
    else:
      self.AllFileNames.append(self.FileName)
    
    print(self.AllFileNames)

    if len(self.AllFileNames) == 0:
      print("Error: No input files available")
      sys.exit()

  # helper functions
  def trainSegments(self, DataTree, name, allVars, index):
    # Initialize TMVA
    ROOT.TMVA.Tools.Instance()
    # PART 1: Train the neural network
    # The output file
    ResultsFileName = self.OutputPrefix + ".root"
    ResultsFile = ROOT.TFile(ResultsFileName, "RECREATE")

    # Create the Factory, responible for training and evaluation
    Factory = ROOT.TMVA.Factory("TMVARegression_" + name, ResultsFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Regression")

    # Create the data loader - give it the name of the output directory
    DataLoader = ROOT.TMVA.DataLoader(self.OutputPrefix)
    IgnoredBranches = [ 'SimulationID' ]
    Branches = DataTree.GetListOfBranches()

    # We need to add everything we do not use as spectators, otherwise we do not have access after the training! (I consider this a ROOT bug!)
    for Name in IgnoredBranches:
      DataLoader.AddSpectator(Name, "F")

    # Add the input variables
    for B in list(Branches):
      if not B.GetName() in IgnoredBranches:
        if not B.GetName().startswith("Result"):
          DataLoader.AddVariable(B.GetName(), "F")

    # Add the target variables:
    for i in range(len(allVars)):
      if i == index:
        DataLoader.AddTarget(allVars[i], "F")
      else: 
        DataLoader.AddSpectator(allVars[i], "F")

    DataLoader.AddSpectator("ResultPositionZ", "F")
    DataLoader.AddSpectator("ResultDirectionZ", "F")

    # Add the regressions tree with weight = 1.0
    DataLoader.AddRegressionTree(DataTree, 1.0);

    # Random split between training and test data
    Cut = ROOT.TCut("")
    DataLoader.PrepareTrainingAndTestTree(Cut, "SplitMode=random:!V");

    # Book a multi-layer perceptron
    Parameters = ROOT.TString()
    Parameters += "!H:!V:NTrees=400:MinNodeSize=5%:MaxDepth=3:BoostType=Grad:SeparationType=GiniIndex:nCuts=20"
    Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDTD", Parameters);
    
    # Train, test, and evaluate internally
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()

  #train
  def train(self):
    FileName = self.AllFileNames[0]
    DataFile = ROOT.TFile(FileName);
    if DataFile.IsOpen() == False:
      print("Error: Opening DataFile: {}".format(FileName))
      sys.exit()

    DataTree = DataFile.Get(self.FileTag);
    if DataTree == 0:
      print("Error: Reading data tree from root file")
      sys.exit()

    if DataTree.GetEntries() > self.MaximumEvents:
      NewDataTree = DataTree.CloneTree(0);
      NewDataTree.SetDirectory(0);

      for i in range(0, self.MaximumEvents):
        DataTree.GetEntry(i);
        NewDataTree.Fill();

      DataFile.Close();
      DataTree = NewDataTree

    print("Shrunk data tree to size: {}", DataTree.GetEntries())

    #run the training helper function
    allVars = ["ResultPositionX", "ResultPositionY", "ResultDirectionX", "ResultDirectionY"]
    for i in range(0): #len(allVars)
      self.trainSegments(DataTree, allVars[i], allVars, i)

  #test
  def test(self):
    # Keeps objects otherwise removed by garbage collected in a list
    ROOTSaver = []

    # Create a new 2D histogram with fine binning
    HistDistance = ROOT.TH1F("Distance", "Difference in Distance in Y", 50, 0, 0.5)
    HistDistance.SetLineColor(ROOT.kGreen)
    HistDistance.SetXTitle("Distance in cm")
    HistDistance.SetYTitle("counts")
    HistDistance.SetMinimum(0)

    HistDirection = ROOT.TH1F("Direction", "Direction", 180, -180, 180)
    HistDirection.SetLineColor(ROOT.kRed)
    HistDirection.SetXTitle("Difference between real and reconstructed recoil electron direction in degrees")
    HistDirection.SetYTitle("counts")
    HistDirection.SetMinimum(0)
     
    # Intialize counters
    NEvents = 0
    AverageEnergy = 0
    AverageDistance = 0
    AverageDirection = 0

    SaveFilePrefix = "AllPatterns"

    for FileName in self.AllFileNames:
      # (1) Read the data tree
      DataFile = ROOT.TFile(FileName);
      if DataFile.IsOpen() == False:
        print("Error: Opening DataFile")
        sys.exit()

      DataTree = DataFile.Get(self.FileTag);
      if DataTree == 0:
        print("Error: Reading data tree from root file")
        sys.exit()

      # Initialize TMVA
      ROOT.TMVA.Tools.Instance()

      # Setup the reader:
      Reader = ROOT.TMVA.Reader("!Color:!Silent");

      IgnoredBranches = [ 'SimulationID' ]
      Branches = DataTree.GetListOfBranches()

      VariableMap = {}

      # We need to add everything we do not use as spectators, otherwise we do not have access after the training
      for Name in IgnoredBranches:
        VariableMap[Name] = array.array('f', [0])
        DataTree.SetBranchAddress(Name, VariableMap[Name])
        Reader.AddSpectator(Name, VariableMap[Name])


      # Add the input variables
      xDim = 0
      yDim = 0
      for B in list(Branches):
        if not B.GetName() in IgnoredBranches:
          if not B.GetName().startswith("Result"):
            VariableMap[B.GetName()] = array.array('f', [0])
            Reader.AddVariable(B.GetName(), VariableMap[B.GetName()])
            DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])
            print("Added: " + B.GetName())
            if "XStripID" in B.GetName():
              xDim += 1
            if "YStripID" in B.GetName():
              yDim += 1

      # Add the target variables:
      VariableMap["ResultPositionX"] = array.array('f', [0])
      DataTree.SetBranchAddress("ResultPositionX", VariableMap["ResultPositionX"])
      Reader.AddSpectator("ResultPositionX", VariableMap["ResultPositionX"])
      VariableMap["ResultPositionY"] = array.array('f', [0])
      DataTree.SetBranchAddress("ResultPositionY", VariableMap["ResultPositionY"])
      # Reader.AddSpectator("ResultPositionY", VariableMap["ResultPositionY"])

      VariableMap["ResultDirectionX"] = array.array('f', [0])
      DataTree.SetBranchAddress("ResultDirectionX", VariableMap["ResultDirectionX"])
      Reader.AddSpectator("ResultDirectionX", VariableMap["ResultDirectionX"])
      VariableMap["ResultDirectionY"] = array.array('f', [0])
      DataTree.SetBranchAddress("ResultDirectionY", VariableMap["ResultDirectionY"])
      Reader.AddSpectator("ResultDirectionY", VariableMap["ResultDirectionY"])
      
      VariableMap["ResultPositionZ"] = array.array('f', [0])
      DataTree.SetBranchAddress("ResultPositionZ", VariableMap["ResultPositionZ"])
      Reader.AddSpectator("ResultPositionZ", VariableMap["ResultPositionZ"])
      VariableMap["ResultDirectionZ"] = array.array('f', [0])
      DataTree.SetBranchAddress("ResultDirectionZ", VariableMap["ResultDirectionZ"])
      Reader.AddSpectator("ResultDirectionZ", VariableMap["ResultDirectionZ"])


      FileName = ROOT.TString(self.OutputPrefix)
      FileName += "/weights/TMVARegression_ResultPositionY_BDTD.weights.xml"
      Reader.BookMVA("BDTD", FileName)

      # Create histograms of the test statistic values:

      # keeps objects otherwise removed by garbage collected in a list
      ROOTSaver = []

      # Rename the histograms if we just have one pattern
      if self.UseAll == True:
        Pattern = "Pattern x={} y={}".format(xDim, yDim)
        HistDistance.SetTitle("Distance (" + Pattern + ")")
        HistDirection.SetTitke("Direction (" + Pattern + ")")
        SaveFilePrefix = Pattern

      # Read the simulated events
      for x in range(0, min(100000, DataTree.GetEntries())):
        DataTree.GetEntry(x)

        SimID = int(VariableMap["SimulationID"][0])
        InputPosX = VariableMap["ResultPositionY"][0]
        # InputPosY = VariableMap["ResultPositionY"][0]
        # InputDirX = VariableMap["ResultDirectionX"][0]
        # InputDirY = VariableMap["ResultDirectionY"][0]

        Energy = 0
        for B in list(Branches):
          if "XStripEnergy" in B.GetName():
            Energy += VariableMap[B.GetName()][0]


        #print("\nSimulation ID: {}:", SimID)

        Result = Reader.EvaluateRegression(self.Methods)
        OutputPosX = Result[0]
        # OutputPosY = Result[1]
        # OutputDirX = Result[2]
        # OutputDirY = Result[3]

        Distance = math.sqrt((OutputPosX - InputPosX)**2)

        # DotProduct = InputDirX*OutputDirX + InputDirY*OutputDirY
        # Mag1 = math.sqrt(InputDirX**2 + InputDirY**2)
        # Mag2 = math.sqrt(OutputDirX**2 + OutputDirY**2)
        # Direction = math.degrees(math.acos(DotProduct / Mag1 / Mag2)) * random.choice([1, -1])

        #print("Distance: {}".format(Distance))
        #print("Direction: {}".format(Direction))
        #print("Input dir ({}/{}), output dir ({}/{})".format(InputDirX, InputDirY, OutputDirX, OutputDirY))

        HistDistance.Fill(Distance)
        # HistDirection.Fill(Direction)

        NEvents += 1
        AverageDistance += Distance
        # AverageDirection += Direction
        AverageEnergy += Energy
    # end for all file names
    
    # create a new TCanvas
    CanvasDistance = ROOT.TCanvas()
    CanvasDistance.SetTitle("Difference in X Distance")
    CanvasDistance.cd()
    HistDistance.Draw()
    CanvasDistance.Update()

    #ROOTSaver.append(ROOT.TCanvas())
    # CanvasDirection = ROOT.TCanvas()
    # CanvasDistance.SetTitle("Direction")
    # HistDirection.Draw()
    # CanvasDirection.Update()

    # Fit
    Fit = ROOT.TF1("Fit", "[0] + gaus(1)", -180, 180)

    # Set initial parameters for the Gaussians
    Fit.SetParameters(200, 2900, 0, 30)  # Initial counts for each Gaussian

    # Fit the histogram with the function
    HistDirection.Fit(Fit, "R")

    Fit.SetLineColor(ROOT.kGreen+3)
    Fit.Draw("SAME")
    # CanvasDirection.Update()

    # CanvasDirection.SaveAs(SaveFilePrefix + ".png")
    #ROOTSaver.append(ROOT.TCanvas())

    print("\nResult:")
    print("All events: {}".format(NEvents))
    print("Average distance: {}".format(AverageDistance/NEvents))
    print("Average direction: {}".format(AverageDirection/NEvents))
    print("Average energy: {}".format(AverageEnergy/NEvents))

    print("Peak: {}".format(Fit.GetParameter(1)))
    print("Offset: {}".format(Fit.GetParameter(0)))
    print("Peak/Offset: {}".format(Fit.GetParameter(1)/Fit.GetParameter(0)))

    # prevent Canvases from closing
    #wait()
    print("Close the ROOT window via File -> Close!")
    ROOT.gApplication.Run()