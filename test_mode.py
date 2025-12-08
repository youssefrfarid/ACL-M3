"""
Test mode switching logic
"""

q = input("Enter command: ")
print(f"Raw input: '{q}'")
print(f"Stripped: '{q.strip()}'")
print(f"Lower: '{q.strip().lower()}'")
print(f"Startswith 'mode': {q.strip().lower().startswith('mode')}")

if q.strip().lower().startswith("mode"):
    print("Mode command detected!")
    try:
        parts = q.split()
        print(f"Split parts: {parts}")
        if len(parts) > 1:
            new_mode = int(parts[1])
            print(f"Parsed mode: {new_mode}")
            if new_mode in [1, 2, 3]:
                print(f"✓ Valid mode: {new_mode}")
            else:
                print("Invalid mode number")
        else:
            print("No mode number provided")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Not a mode command")
