import os
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-17"
os.environ.pop("SPARK_SUBMIT_OPTS", None)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("--- Starting Part B: Mall Customer Segmentation ---")
    
    # 1. Initialize SparkSession
    spark = SparkSession.builder \
        .appName("MallCustomerSegmentation") \
        .config("spark.sql.warehouse.dir", "temp-spark-warehouse") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    
    # 2. Load the dataset
    data_path = os.path.join("data", "Mall_Customers.csv")
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    results_path = os.path.join("outputs", "mall_results.txt")
    with open(results_path, "w") as f:
        f.write("=== Mall Customer Segmentation Results ===\n\n")
        
        # Explore the dataset
        f.write("--- Schema ---\n")
        schema_str = df._jdf.schema().treeString()
        f.write(schema_str + "\n")
        
        # Capture basic statistics
        f.write("--- Basic Descriptive Statistics ---\n")
        stats_df = df.describe()
        stats_pd = stats_df.toPandas()
        f.write(stats_pd.to_string(index=False) + "\n\n")
    
    print("EDA completed and saved.")
    
    # 3. Select appropriate numerical features for clustering
    # Columns are typically: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
    # Check if column names are correct. We'll rename them if they have special characters.
    # Standard names: Age, Annual Income (k$), Spending Score (1-100)
    features_to_scale = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    
    # 4. Create a feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=features_to_scale, outputCol="raw_features")
    assembled_df = assembler.transform(df)
    
    # 5. Normalize or standardize selected features using StandardScaler
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)
    
    # Function to run KMeans and return results
    def run_kmeans(k):
        print(f"Running KMeans with k={k}...")
        kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42)
        model = kmeans.fit(scaled_df)
        
        # Make predictions
        predictions = model.transform(scaled_df)
        
        # Get cluster centers in the scaled space
        scaled_centers = model.clusterCenters()
        
        # Transform scaled cluster centers back to original scale
        # StandardScaler formula: scaled = (x - mean) / std
        # Therefore: x = scaled * std + mean
        means = scaler_model.mean.toArray()
        stds = scaler_model.std.toArray()
        
        original_centers = []
        for center in scaled_centers:
            orig_center = center * stds + means
            original_centers.append(orig_center)
            
        return model, predictions, original_centers
    
    # 6. Apply K-Means clustering for k=3 and k=5
    results = {}
    for k in [3, 5]:
        model, preds, centers = run_kmeans(k)
        results[k] = {
            "model": model,
            "predictions": preds,
            "centers": centers
        }
        
        # Write to results file
        with open(results_path, "a") as f:
            f.write(f"\n--- K-Means Clustering (k = {k}) ---\n")
            f.write("Cluster Centers (Age, Annual Income, Spending Score):\n")
            for idx, center in enumerate(centers):
                f.write(f"Cluster {idx}: Age={center[0]:.2f}, Income={center[1]:.2f}, Spending Score={center[2]:.2f}\n")
            
            f.write("\nSample Cluster Assignments:\n")
            sample_assignments = preds.select("CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "cluster").limit(10).toPandas()
            f.write(sample_assignments.to_string(index=False) + "\n\n")
            
        # Convert predictions to Pandas for plotting
        preds_pd = preds.select("Age", "Annual Income (k$)", "Spending Score (1-100)", "cluster").toPandas()
        
        # 7. Display and save the predicted cluster assignment
        # Plot Annual Income vs Spending Score
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x="Annual Income (k$)", 
            y="Spending Score (1-100)", 
            hue="cluster", 
            palette="Set1" if k==3 else "tab10", 
            data=preds_pd, 
            s=100, 
            alpha=0.8,
            edgecolor="w"
        )
        
        # Also plot cluster centers on the same plot
        # We need the 1st and 2nd dimensions of original centers: Annual Income (idx 1) and Spending Score (idx 2)
        centers_pd = pd.DataFrame(centers, columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
        plt.scatter(
            centers_pd["Annual Income (k$)"], 
            centers_pd["Spending Score (1-100)"], 
            c="black", 
            s=300, 
            marker="*", 
            label="Centroids"
        )
        
        plt.title(f"Mall Customer Segments (k = {k})")
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        plt.legend(title="Cluster")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("outputs", f"mall_clusters_k{k}.png"), dpi=150)
        plt.close()
        
    print("K-Means training and plotting completed.")
    spark.stop()
    print("--- Part B Completed successfully ---")

if __name__ == "__main__":
    main()
