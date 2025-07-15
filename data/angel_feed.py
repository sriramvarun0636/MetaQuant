from SmartAPI.smartConnect import SmartConnect

API_KEY = "W4qv9Yok"
CLIENT_ID = "1234"
PASSWORD = "Varun@0636" 

obj = SmartConnect(api_key=API_KEY)

# Login
data = obj.generateSession(CLIENT_ID, PASSWORD, TOTP)
refreshToken = data['data']['refreshToken']

# Fetch profile (optional)
profile = obj.getProfile(refreshToken)

print("Working")