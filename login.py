# File: data/kite_client.py

from kiteconnect import KiteConnect
import webbrowser
import os

API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

kite = KiteConnect(api_key=API_KEY)

# Step 1: Get request token
login_url = kite.login_url()
print("Login here to get request_token:", login_url)
webbrowser.open(login_url)  # opens browser automatically

# Paste your request_token manually after login
request_token = input("Enter the request_token from URL after login: ")

# Step 2: Generate access token
try:
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    kite.set_access_token(data["access_token"])
    print("✅ Login successful!")
    print("Access Token:", data["access_token"])

    # Optional: Save token to file
    with open("access_token.txt", "w") as f:
        f.write(data["access_token"])

except Exception as e:
    print("❌ Login failed:", e)