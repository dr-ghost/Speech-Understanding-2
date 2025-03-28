import os
import requests
import subprocess

def download_data():
    # Download data from the Google Drive
    # Google Drive file URLs
    url1 = "https://drive.google.com/file/d/1AC-Q8dEw1LTPdpEi5ofS04ZmZWdl8BBg/view?usp=drive_link"
    url2 = "https://drive.google.com/file/d/1Onu4jzcyasrxTR1rRT9rl9AfkUrueM6o/view?usp=drive_link"
    url3 = "https://drive.google.com/file/d/1vAoILnZvbYNWbFkVutlqa4qUjoO-t3xL/view?usp=drive_link"

    urls = [url1, url2, url3]

    current_dir = os.path.dirname(os.path.abspath(__file__))

    file_names = ["vox_1", "vox_2", "vox_2_text"]
    for i, url in enumerate(urls, start=1):
        zip_filename = os.path.join(current_dir, f"{file_names[i-1]}.zip")

        print(f"Downloading file {i} from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"File {i} downloaded successfully: {zip_filename}")
        else:
            print(f"Failed to download file {i} from {url}")
            continue
        

def unzipy(directory=os.path.dirname(os.path.abspath(__file__))):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.zip'):
            zip_path = os.path.join(directory, filename)
            print(f"Extracting {zip_path}...")
            try:
                # Run the unzip command with the -o flag to overwrite without prompting
                result = subprocess.run(
                    ["unzip", "-o", zip_path, "-d", directory],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error extracting {zip_path}: {e.stderr}")
        else:
            print(f"Skipping {filename}: not a .zip file.")

            

if __name__ == "__main__":
    unzipy()