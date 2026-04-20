import requests
import sys

# Point directly to the original image-only endpoint
BRIDGE_URL = "http://localhost:8004/v1/analyze-llava"

def test_image_only(image_path):
    print(f"📸 Testing Image-Only LLaVA using {image_path}...")
    
    try:
        with open(image_path, "rb") as f:
            # Notice we ONLY send 'files'. We do NOT send a 'data' dictionary here.
            files = {"file": f}
            response = requests.post(BRIDGE_URL, files=files)
            
        print("Response Status:", response.status_code)
        print("Response Body:\n", response.json())
        
    except FileNotFoundError:
        print(f"❌ Could not find {image_path}.")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the Bridge. Is 'python3 app.py' running?")
        
    print("-" * 50)

if __name__ == "__main__":
    print("🚀 Starting Local Bridge Test for Original Endpoint...")
    
    # Test just the image
    test_image_only("download.jpeg")