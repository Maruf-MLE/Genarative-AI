import serial
import time
from supabase import create_client, Client

# -------------------
# Supabase Configuration
# -------------------
url = "https://imtfmdnxktmfcjinpeus.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltdGZtZG54a3RtZmNqaW5wZXVzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwMjcwNjcsImV4cCI6MjA3NTYwMzA2N30.J-0Kys0SdHxgi7tJueGhpZFHGb-aXUMD0NfFZSr2mK4"
supabase: Client = create_client(url, key)

# -------------------
# Serial Port Configuration
# -------------------
ser = serial.Serial('COM3', 9600)  # তোমার Arduino যে port এ connected সেটি দাও
time.sleep(2)  # serial connection stable হতে একটু সময় দাও

print("Listening to Arduino data...")

while True:
    try:
        # Read one line from Serial
        line = ser.readline().decode('utf-8').strip()
        
        # Check if line has all expected values
        data = line.split(',')
        if len(data) == 6:
            bpm, temp, humidity, lat, lon, dust = data
            
            # Convert values to proper types
            record = {
                "bpm": float(bpm),
                "temperature": float(temp),
                "humidity": float(humidity),
                "latitude": float(lat),
                "longitude": float(lon),
                "dust": float(dust),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Insert into Supabase
            supabase.table("health_data").insert(record).execute()
            
            print("✅ Data uploaded:", record)
        else:
            print("⚠️ Invalid data format:", line)

    except Exception as e:
        print("Error:", e)
        time.sleep(2)
