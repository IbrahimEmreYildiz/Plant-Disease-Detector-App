import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

ZENODO_RECORD_ID = "13958858"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DATA_DIR = Path("data/plantseg")

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def main():
    print(f"Fetching record information from Zenodo (Record: {ZENODO_RECORD_ID})...")
    response = requests.get(ZENODO_API_URL)
    if response.status_code != 200:
        print(f"Failed to fetch record from Zenodo. Status Code: {response.status_code}")
        print("Please visit https://zenodo.org/records/13958858 and download it manually.")
        return

    data = response.json()
    files = data.get('files', [])
    
    if not files:
        print("No files found in the Zenodo record.")
        return
        
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded_zips = []
    
    for file_info in files:
        filename = file_info['key']
        download_url = file_info['links']['self']
        output_path = DATA_DIR / filename
        
        # Only download if it doesn't exist
        if not output_path.exists():
            print(f"\nDownloading {filename}...")
            download_file(download_url, output_path)
        else:
            print(f"\n{filename} already exists. Skipping download.")
            
        if filename.endswith('.zip'):
            downloaded_zips.append(output_path)
            
    # Extract zip files
    for zip_file in downloaded_zips:
        print(f"\nExtracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # We assume it creates the necessary 'images' and 'annotations' folders
            # If the zip already contains a root folder, we might need to handle it
            zip_ref.extractall(DATA_DIR)
        print(f"Extracted {zip_file.name}.")
        
    print("\nDataset download and extraction complete!")
    print("You can now run: python convert_dataset.py")

if __name__ == "__main__":
    main()
