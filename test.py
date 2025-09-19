import cv2
import numpy as np
import requests
import time

def test_ip_webcam():
    # Replace with your phone's IP address from the IP Webcam app
    phone_ip = "192.168.1.100"  # Change this to your phone's IP
    port = "8080"  # Default IP Webcam port
    
    # Different stream URLs to try
    urls = {
        "MJPEG Stream": f"http://{phone_ip}:{port}/video",
        "Alternative MJPEG": f"http://{phone_ip}:{port}/stream/video.mjpeg",
        "Single Shot": f"http://{phone_ip}:{port}/shot.jpg"
    }
    
    print(f"Testing IP Webcam connection to {phone_ip}:{port}")
    print("Make sure your phone and computer are on the same WiFi network")
    print("And that IP Webcam app is running on your phone\n")
    
    # Test connection first
    try:
        response = requests.get(f"http://{phone_ip}:{port}", timeout=5)
        print("✓ Successfully connected to IP Webcam interface")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to IP Webcam: {e}")
        print("Please check:")
        print("1. Phone and computer are on same WiFi")
        print("2. IP address is correct")
        print("3. IP Webcam app is running")
        return
    
    # Test video stream
    print("\nTesting video streams...")
    
    for stream_name, url in urls.items():
        if "shot.jpg" in url:
            continue  # Skip single shot for now
            
        print(f"\nTrying {stream_name}: {url}")
        
        cap = cv2.VideoCapture(url)
        
        # Set properties for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"✗ Failed to open {stream_name}")
            continue
            
        print(f"✓ Successfully opened {stream_name}")
        
        # Test frame capture
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 30:  # Test 30 frames
            ret, frame = cap.read()
            
            if not ret:
                print(f"✗ Failed to read frame from {stream_name}")
                break
                
            frame_count += 1
            
            # Show frame info
            if frame_count == 1:
                h, w = frame.shape[:2]
                print(f"  Frame size: {w}x{h}")
            
            # Display frame
            cv2.imshow(f'IP Webcam Test - {stream_name}', frame)
            
            # Calculate FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"  FPS: {fps:.1f}")
            
            # Press 'q' to quit or 'n' for next stream
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break
        
        cap.release()
        cv2.destroyWindow(f'IP Webcam Test - {stream_name}')
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Total frames: {frame_count}")
        
        if avg_fps > 15:
            print(f"✓ {stream_name} works well for real-time use")
            break
        else:
            print(f"⚠ {stream_name} may be too slow for real-time use")
    
    cv2.destroyAllWindows()

def test_single_shot():
    """Test single frame capture"""
    phone_ip = "192.168.1.100"  # Change this to your phone's IP
    port = "8080"
    
    url = f"http://{phone_ip}:{port}/shot.jpg"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            # Convert to OpenCV image
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                print("✓ Single shot capture successful")
                cv2.imshow('Single Shot Test', img)
                cv2.waitKey(3000)  # Show for 3 seconds
                cv2.destroyAllWindows()
            else:
                print("✗ Failed to decode image")
        else:
            print(f"✗ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Single shot test failed: {e}")

if __name__ == "__main__":
    print("IP Webcam Test Script")
    print("====================")
    print("Instructions:")
    print("1. Make sure IP Webcam app is running on your Android phone")
    print("2. Note the IP address shown in the app (e.g., 192.168.1.100)")
    print("3. Update the 'phone_ip' variable in this script")
    print("4. Make sure both devices are on the same WiFi network")
    print()
    
    # Update this with your phone's IP address from the IP Webcam app
    phone_ip = input("Enter your phone's IP address (or press Enter for 192.168.1.100): ").strip()
    if not phone_ip:
        phone_ip = "192.168.1.100"
    
    # Update the IP in the functions
    globals()['phone_ip'] = phone_ip
    
    print("\nPress 'n' to try next stream, 'q' to quit")
    test_ip_webcam()
    
    print("\nTesting single shot capture...")
    test_single_shot()
    
    print("\nTest complete!")