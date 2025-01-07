ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.7"

val sparkVersion = "2.4.0"

lazy val root = (project in file("."))
  .settings(
    name := "UVT-SENTIMENT-ANALYSIS",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "log4j" % "log4j" % "1.2.17",
      "org.jfree" % "jfreechart" % "1.5.3"
    )
  )
