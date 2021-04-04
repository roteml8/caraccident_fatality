from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import pyspark.sql.functions
from pyspark.sql.functions import col,when
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

-----------------------------------------------------------------------------------------------------------------

accident = pd.read_csv('/home/linuxu/Desktop/1.csv')
modifiedAccident = accident.dropna()

-----------------------------------------------------------------------------------------------------------------

# if you would like to use a portion of the data
subData = modifiedAccident[0:10000]

# if you would like to use the entire data file
#subData = modifiedAccident

subData.to_csv('/home/linuxu/Desktop/subData.csv',index=False)

-----------------------------------------------------------------------------------------------------------------

allColumns = ['Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude', 'Latitude', 'Police_Force', 'Number_of_Vehicles',
       'Number_of_Casualties', 'Date', 'Day_of_Week', 'Time', 'Local_Authority_(District)', 'Local_Authority_(Highway)', '1st_Road_Class', '1st_Road_Number',
       'Road_Type', 'Speed_limit', 'Junction_Detail', 'Junction_Control', '2nd_Road_Class', '2nd_Road_Number', 'Pedestrian_Crossing-Human_Control',
       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Special_Conditions_at_Site',
       'Carriageway_Hazards', 'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident', 'LSOA_of_Accident_Location']

-----------------------------------------------------------------------------------------------------------------

maxAcc = 0
maxIter = 0

i = 1
while i <= 10:
    
    #open a spark session
  spark = SparkSession.builder.appName('ml-bank').getOrCreate()

    #read the data
  df = spark.read.csv('/home/linuxu/Desktop/subData.csv', header = True, inferSchema = True)
    
    #random select from all columns
  comb = random.sample(allColumns, k = 4)

    #print the combinations that was choosen
  print(comb)

    #open and write CombinationsTried file
  fc = open("/home/linuxu/Desktop/CombinationsTried.txt","a+")
  sentence = str(i)+ ".  "+ ', '.join(map(str, comb))  + "\n"
  fc.write(sentence)
  fc.close()

  #binarization
  df = df.withColumn("Accident_Severity", when(col("Accident_Severity")  == 3,0).otherwise(1))
    
     #select the first col to be accident sevirity and the rest are random
  df = df.select('Accident_Severity', comb[0], comb[1], comb[2], comb[3])  
    
  cols = df.columns

    #differentiate between strings and int
  categoricalColumns = []
  numericCols = []
  for cmb in range(4):
    if df.dtypes[cmb+1][1] == 'string':
      categoricalColumns.append(cols[cmb+1])
    elif df.dtypes[cmb+1][1] == 'int':
      numericCols.append(cols[cmb+1])

    #print the categorial cols - strings
  print("categoricalColumns", categoricalColumns)

    #prinf the numeric cols - ints
  print("numericCols", numericCols)

    #Create a list to combinely have both type of columns
  stages = []
    
    #Prepare Data for Machine Learning
  for categoricalCol in categoricalColumns:
      stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
      encoder = OneHotEncoderEstimator(inputCols= [stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
      stages += [stringIndexer, encoder]

  label_stringIdx = StringIndexer(inputCol = 'Accident_Severity', outputCol = 'label')
  stages += [label_stringIdx]

  assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
  assembler = VectorAssembler(inputCols = assemblerInputs, outputCol="features")
  stages += [assembler]

  pipeline = Pipeline(stages = stages)
  pipelineModel = pipeline.fit(df)
    
  df = pipelineModel.transform(df)
  selectedCols = ['label', 'features'] + cols
    
  df = df.select(selectedCols)

    #split the data
  train, test = df.randomSplit([0.5, 0.5])
  print("Data splitted")

    
    ###################
####### algorithm ########
    ###################
    
  lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=100)
  lrModel = lr.fit(train)
  predictions = lrModel.transform(test)
  print("LR pridictions saving...")

  filename = "comb" + str(i) + "_LR.csv"
  predictions.toPandas().to_csv("/home/linuxu/Desktop/"+filename)
  print("LR predicitions saved")

  #Evaluation Measures
  trainingSummary = lrModel.summary

  fr = open("/home/linuxu/Desktop/results.txt","a+")
  fr.write(str(i)+ ".  " + "Column Combination: "+ ', '.join(map(str, comb)) + "\n")
  fr.write("Results for Logistic Regression: \n")
  fr.write("Accuracy: " + str((trainingSummary.accuracy)*100) + "\n")
  fr.write("Precision: " + str(trainingSummary.weightedPrecision) + "\n")
  fr.write("Recall: " + str(trainingSummary.weightedRecall) + "\n")
  fr.write("TruePositiveRate: " + str(trainingSummary.weightedTruePositiveRate) + "\n")
  fr.write("FalsePositiveRate: " + str(trainingSummary.weightedFalsePositiveRate) + "\n\n\n\n")
  fr.close()

  print("LR done")

  if maxAcc < (trainingSummary.accuracy)*100:
    maxAcc = (trainingSummary.accuracy)*100
    maxIter = i
  

  spark.catalog.clearCache()
  del comb
  del categoricalColumns
  del numericCols
  del stages
  print(i, "pass completed")
  i = i + 1

#write max
fr = open("/home/linuxu/Desktop/results.txt","a+")
fr.write("Maximum Accuracy is: " + str(maxAcc) + " Of Combination Number: " + str(maxIter))
fr.close()
