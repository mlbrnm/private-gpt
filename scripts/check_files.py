import requests
import json
import urllib3

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def list_files():
    """Check what files are in the vector database"""
    try:
        response = requests.get(
            'https://wsmwsllm01:8001/v1/ingest/list',
            verify=False  # Disable SSL verification
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        files = response.json()
        print("\nFiles known to PrivateGPT:")
        print(json.dumps(files, indent=2))
        
        return files
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        print("Try running with SSL verification disabled")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
        print("Check if PrivateGPT is running and the URL is correct")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error accessing PrivateGPT API: {e}")
        return None

if __name__ == "__main__":
    print("Checking PrivateGPT's known files...")
    list_files()
