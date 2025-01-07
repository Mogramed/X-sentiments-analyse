import java.util.Locale

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.IntegerType
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.category.DefaultCategoryDataset
import javax.swing.{JFrame, WindowConstants}

object TwitterSentimentAnalysis  {
  def main(args: Array[String]) {

    // Printing application name
    println("SentimentTrainer")

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)


    // Creating a SparkSession
    println("Creating SparkSession...")
    val spark = SparkSession
      .builder()
      .appName("Spark Sentiment")
      .config("spark.master", "local")
      .getOrCreate()

    // Defining paths for training data
    val twitterTrainPath = "src/twitter_data/train.csv"

    // Printing the path
    println("Reading Twitter data from: " + twitterTrainPath)

    // Reading and preprocessing Twitter data
    val twitterData = readTwitterData(twitterTrainPath, spark)

    // Tokenizing the data
    println("Tokenizing data...")
    val tokenizer = new RegexTokenizer()
      .setInputCol("Preprocessed")
      .setOutputCol("Tokenized All")
      .setPattern("\\s+")

    // Tokenizing words
    println("Tokenizing words...")
    val wordTokenizer = new RegexTokenizer()
      .setInputCol("Preprocessed")
      .setOutputCol("Tokenized Words")
      .setPattern("\\W")

    // Setting default locale
    Locale.setDefault(Locale.ENGLISH)

    // Removing stop words
    println("Removing stop words...")
    val stopW = new StopWordsRemover()
      .setInputCol("Tokenized Words")
      .setOutputCol("Stopped")

    // Creating n-grams
    println("Creating n-grams...")
    val ngram = new NGram()
      .setN(2)
      .setInputCol("Stopped")
      .setOutputCol("Grams")

    // Vectorizing tokens
    println("Vectorizing tokens...")
    val tokenVectorizer = new CountVectorizer()
      .setInputCol("Tokenized All")
      .setOutputCol("Token Vector")

    // Vectorizing n-grams
    println("Vectorizing n-grams...")
    val gramVectorizer = new CountVectorizer()
      .setInputCol("Grams")
      .setOutputCol("Gram Vector")

    // Assembling features
    println("Assembling features...")
    val assembler = new VectorAssembler()
      .setInputCols(Array("Token Vector"))
      .setOutputCol("features")

    // Creating a Logistic Regression model
    println("Creating Logistic Regression model...")
    val model = new LogisticRegression()
      .setFamily("multinomial")
      .setLabelCol("Sentiment")

    // Defining the pipeline
    println("Defining pipeline...")
    val pipe = new Pipeline()
      .setStages(Array(tokenizer,
        wordTokenizer,
        stopW,
        ngram,
        tokenVectorizer,
        gramVectorizer,
        assembler,
        model))

    // Defining parameters for tuning
    println("Defining parameters for tuning...")
    val paramMap = new ParamMap()
      .put(tokenVectorizer.vocabSize, 10000)
      .put(gramVectorizer.vocabSize, 10000)
      .put(model.elasticNetParam, .8)
      .put(model.tol, 1e-20)
      .put(model.maxIter, 100)

    // Fitting the pipeline with parameters to the data
    println("Fitting pipeline with parameters to the data...")
    val lr = pipe.fit(twitterData, paramMap)

    // Transforming the data
    println("Transforming the data...")
    val tr = lr.transform(twitterData).select("Sentiment", "probability", "prediction")

    // Printing the first 10 rows
    println("Printing the first 10 rows...")
    tr.take(10).foreach(println)

    // Evaluating the model using BinaryClassificationEvaluator
    println("Evaluating the model...")
    val eval = new BinaryClassificationEvaluator()
      .setLabelCol("Sentiment")
      .setRawPredictionCol("prediction")

    // Calculating ROC
    val roc = eval.evaluate(tr)
    println(s"ROC: ${roc}")

    // Printing schema of the transformed data
    println("Printing schema of the transformed data...")
    tr.printSchema()

    // Printing schema again for confirmation
    println("Printing schema again for confirmation...")
    tr.printSchema()

    // Building parameter grid for cross-validation
    println("Building parameter grid for cross-validation...")
    val paramGrid = new ParamGridBuilder()
      .addGrid(tokenVectorizer.vocabSize, Array(10000))
      .addGrid(gramVectorizer.vocabSize, Array(10000))
      .addGrid(model.elasticNetParam, Array(.8))
      .addGrid(model.tol, Array(1e-20))
      .addGrid(model.maxIter, Array(100))
      .build()

    // Configuring cross-validation
    println("Configuring cross-validation...")
    val cv = new CrossValidator()
      .setEstimator(pipe)
      .setEvaluator(new BinaryClassificationEvaluator()
        .setRawPredictionCol("prediction")
        .setLabelCol("Sentiment"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(1)

    println("---------------------------------")

    // Fitting cross-validator to the data
    println("Fitting cross-validator to the data...")
    val cvmodel = cv.fit(twitterData)

    // Transforming data using cross-validator
    println("Transforming data using cross-validator...")
    val predictions = cvmodel.transform(twitterData)
      .select("ItemID","Preprocessed", "probability", "prediction")
      .collect()

    printPredictions(predictions)

    generateSentimentDistributionChart(predictions)


    println("---------------------------------")


    // Transforming data using cross-validator
    println("Transforming data using cross-validator...")
    cvmodel.transform(twitterData)
      .select("ItemID","Preprocessed", "probability", "prediction")
      .collect().take(10)
      .foreach(println)


    // Printing evaluation metrics
    println("Printing evaluation metrics...\n\n\n")
    cvmodel.avgMetrics.foreach(println)
    println("\n\n\n")

    // Writing the model to disk
    println("Writing the model to disk...")
    cvmodel.write.overwrite().save("twitter-sentiment-model")

  }

  // Function to read and preprocess Twitter data
  def readTwitterData(path: String, spark: SparkSession) = {

    val data = spark.read.format("csv")
      .option("header", "true")
      .load(path)

    // Defining preprocessing function
    val preprocess: String => String = {
      _.replaceAll("((.))\\1+","$1")
    }

    // Applying preprocessing function as a User Defined Function (UDF)
    val preprocessUDF = udf(preprocess)

    // Applying the UDF to the data
    val newCol = preprocessUDF.apply(data("SentimentText"))
    val label = data("Sentiment").cast(IntegerType)

    // Selecting required columns
    data.withColumn("Preprocessed", newCol)
      .withColumn("Sentiment",label)
      .select("ItemID","Sentiment","Preprocessed")
  }

  def printPredictions(predictions: Array[org.apache.spark.sql.Row]): Unit = {
    // ANSI color codes
    val ANSI_RESET = "\u001B[0m"
    val ANSI_BOLD = "\u001B[1m"
    val ANSI_RED = "\u001B[31m" // Red color for "Négatif"
    val ANSI_GREEN = "\u001B[32m" // Green color for "Positif"

    println("\nPredictions:")
    predictions.foreach { row =>
      val itemID = row.getAs[String]("ItemID")
      val preprocessed = row.getAs[String]("Preprocessed")
      val probability = row.getAs[org.apache.spark.ml.linalg.Vector]("probability")
      val prediction = row.getAs[Double]("prediction")

      // Convert probabilities to percentage and format them
      val formattedProbability = probability.toArray.map(p => f"${p * 100}%.2f%%")

      // Interpret the prediction using labels
      val predictionLabel = if (prediction == 1.0) s"${ANSI_GREEN}Positif 😁${ANSI_RESET}" else s"${ANSI_RED}Négatif 🫤${ANSI_RESET}"

      // Construct probability message with colors
      val probabilityMessage = s"${ANSI_BOLD}Probabilité Positif:${ANSI_RESET} ${formattedProbability(1)}, ${ANSI_BOLD}Négatif:${ANSI_RESET} ${formattedProbability(0)}"

      println(s"${ANSI_BOLD}ItemID:${ANSI_RESET} $itemID")
      println(s"${ANSI_BOLD}Texte prétraité:${ANSI_RESET} $preprocessed")
      println(probabilityMessage)
      println(s"${ANSI_BOLD}Prédiction:${ANSI_RESET} $predictionLabel")
      println()
    }
  }

  def generateSentimentDistributionChart(predictions: Array[org.apache.spark.sql.Row]): Unit = {
    // Create a dataset
    val dataset = new DefaultCategoryDataset()

    // Count positive and negative predictions
    var positiveCount = 0
    var negativeCount = 0
    predictions.foreach { row =>
      val prediction = row.getAs[Double]("prediction")
      if (prediction == 1.0) {
        positiveCount += 1
      } else {
        negativeCount += 1
      }
    }

    // Add data to the dataset
    dataset.addValue(positiveCount, "Sentiment", "Positive")
    dataset.addValue(negativeCount, "Sentiment", "Negative")

    // Create the chart
    val chart = ChartFactory.createBarChart(
      "Sentiment Distribution", // Chart title
      "Sentiment", // X-axis label
      "Count", // Y-axis label
      dataset, // Dataset
      PlotOrientation.VERTICAL,
      true, true, false)

    // Display the chart in a frame
    val frame = new JFrame("Sentiment Distribution")
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    val chartPanel = new ChartPanel(chart)
    frame.setContentPane(chartPanel)
    frame.pack()
    frame.setVisible(true)
  }

}
