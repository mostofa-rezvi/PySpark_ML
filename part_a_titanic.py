import os
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-17"
os.environ.pop("SPARK_SUBMIT_OPTS", None)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, desc
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("--- Starting Part A: Titanic Survival Prediction ---")
    
    # 1. Initialize SparkSession
    spark = SparkSession.builder \
        .appName("TitanicSurvivalPrediction") \
        .config("spark.sql.warehouse.dir", "temp-spark-warehouse") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # 2. Load the dataset
    data_path = os.path.join("data", "titanic.csv")
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # Save stdout printouts to a results file
    results_path = os.path.join("outputs", "titanic_results.txt")
    with open(results_path, "w") as f:
        f.write("=== Titanic Survival Prediction Results ===\n\n")
        
        # 3. Explore the dataset
        f.write("--- Schema ---\n")
        # Capture schema string
        schema_str = df._jdf.schema().treeString()
        f.write(schema_str + "\n")
        
        # Capture basic statistics
        f.write("--- Basic Descriptive Statistics ---\n")
        stats_df = df.describe()
        stats_pd = stats_df.toPandas()
        f.write(stats_pd.to_string(index=False) + "\n\n")
        
        # Identify missing values
        f.write("--- Missing Value Counts ---\n")
        missing_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()
        f.write(missing_counts.to_string(index=False) + "\n\n")
        
    print("EDA completed and saved.")
    
    # 4. Handle Missing Values
    # Impute Age with median
    age_median = df.approxQuantile("Age", [0.5], 0.001)[0]
    print(f"Imputing missing Age values with median: {age_median}")
    df = df.withColumn("Age", when(col("Age").isNull(), age_median).otherwise(col("Age")))
    
    # Impute Embarked with mode (most frequent value)
    embarked_mode_df = df.groupBy("Embarked").count().orderBy(desc("count"))
    embarked_mode = embarked_mode_df.filter(col("Embarked").isNotNull()).first()["Embarked"]
    print(f"Imputing missing Embarked values with mode: {embarked_mode}")
    df = df.withColumn("Embarked", when(col("Embarked").isNull(), embarked_mode).otherwise(col("Embarked")))
    
    # Drop Cabin column due to high proportion of missing values (>70%), and unnecessary string columns
    df = df.drop("Cabin", "PassengerId", "Name", "Ticket")
    
    # 5. Preprocessing & Encoding Pipeline
    # Convert Sex and Embarked to index representation
    sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_Indexed")
    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_Indexed")
    
    # One-Hot Encode categorical features
    sex_encoder = OneHotEncoder(inputCol="Sex_Indexed", outputCol="Sex_Vec")
    embarked_encoder = OneHotEncoder(inputCol="Embarked_Indexed", outputCol="Embarked_Vec")
    
    # 6. Combine features into a single vector
    feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_Vec", "Embarked_Vec"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # 7. Split dataset into train and test sets (80/20)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # 8. Train a Random Forest Classifier
    rf = RandomForestClassifier(labelCol="Survived", featuresCol="features", numTrees=100, maxDepth=5, seed=42)
    
    # Assemble pipeline
    pipeline = Pipeline(stages=[sex_indexer, embarked_indexer, sex_encoder, embarked_encoder, assembler, rf])
    
    print("Training Random Forest model...")
    model = pipeline.fit(train_df)
    
    # 9. Generate predictions
    predictions = model.transform(test_df)
    
    # 10. Evaluate the model
    # Evaluators
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
    prec_evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision")
    rec_evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
    auc_evaluator = BinaryClassificationEvaluator(labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    
    accuracy = acc_evaluator.evaluate(predictions)
    precision = prec_evaluator.evaluate(predictions)
    recall = rec_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")
    
    # Compute Confusion Matrix
    confusion_matrix = predictions.groupBy("Survived").pivot("prediction", [0.0, 1.0]).count().na.fill(0).orderBy("Survived").toPandas()
    print("Confusion Matrix:\n", confusion_matrix)
    
    # Append results to the text file
    with open(results_path, "a") as f:
        f.write("--- Model Evaluation Metrics ---\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
        f.write(f"AUC (ROC): {auc:.4f}\n\n")
        
        f.write("--- Confusion Matrix ---\n")
        f.write(confusion_matrix.to_string(index=False) + "\n\n")
        
        # 11. Display sample predictions
        f.write("--- Sample Prediction Results ---\n")
        sample_preds = predictions.select("Pclass", "Sex", "Age", "Fare", "Survived", "prediction", "probability").limit(10).toPandas()
        f.write(sample_preds.to_string(index=False) + "\n")
        
    # 12. Plot and save Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    cm_data = [[confusion_matrix.iloc[0, 1], confusion_matrix.iloc[0, 2]],
               [confusion_matrix.iloc[1, 1], confusion_matrix.iloc[1, 2]]]
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Not Survived (0)", "Survived (1)"],
                yticklabels=["Not Survived (0)", "Survived (1)"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Titanic Random Forest Confusion Matrix\n(Accuracy: {accuracy*100:.2f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "titanic_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # 13. Plot Feature Importances
    # Extract feature importances from random forest model
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray()
    
    # We can reconstruct the feature names
    # numeric: Pclass, Age, SibSp, Parch, Fare (5)
    # Sex_Vec (1 category after OHE if binary, since OneHotEncoder drops last category by default)
    # Embarked_Vec (2 categories after OHE if 3 categories)
    # Let's map them to their names for simplicity:
    feature_names = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex (Male)", "Embarked (Q)", "Embarked (S)"]
    # Pad or slice to match length if necessary
    if len(importances) < len(feature_names):
        feature_names = feature_names[:len(importances)]
    elif len(importances) > len(feature_names):
        feature_names += [f"Feature_{i}" for i in range(len(feature_names), len(importances))]
        
    feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df, palette="viridis")
    plt.title("Titanic Random Forest Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "titanic_feature_importances.png"), dpi=150)
    plt.close()
    
    print("Plots saved to outputs directory.")
    spark.stop()
    print("--- Part A Completed successfully ---")

if __name__ == "__main__":
    main()
