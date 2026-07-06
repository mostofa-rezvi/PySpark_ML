import os
import urllib.request

def download_file(url, filepath):
    print(f"Downloading {url} to {filepath}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    # Create data and outputs directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Dataset URLs
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    mall_url = "https://raw.githubusercontent.com/kennedykwangari/Mall-Customer-Segmentation-Data/master/Mall_Customers.csv"
    
    # Target paths
    titanic_path = os.path.join("data", "titanic.csv")
    mall_path = os.path.join("data", "Mall_Customers.csv")
    
    # Download datasets
    download_file(titanic_url, titanic_path)
    download_file(mall_url, mall_path)

if __name__ == "__main__":
    main()
