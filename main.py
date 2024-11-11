import cv2
import easyocr
import imutils
import os
import csv

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# State code dictionary (from your list)
state_codes = {
    "Andhra Pradesh": "AP",
    "Arunachal Pradesh": "AR",
    "Assam": "AS",
    "Bihar": "BR",
    "Chandigarh": "CH",
    "Chhattisgarh": "CG",
    "Dadra & Nagar Haveli and Daman & Diu": "DD",
    "Delhi": "DL",
    "Goa": "GA",
    "Gujarat": "GJ",
    "Haryana": "HR",
    "Himachal Pradesh": "HP",
    "Jammu & Kashmir": "JK",
    "Karnataka": "KA",
    "Kerala": "KL",
    "Ladakh": "LD",
    "Lakshadweep": "LD",
    "Madhya Pradesh": "MP",
    "Maharashtra": "MH",
    "Manipur": "MN",
    "Meghalaya": "ML",
    "Mizoram": "MZ",
    "Nagaland": "NL",
    "Odisha": "OR",
    "Puducherry": "PY",
    "Punjab": "PB",
    "Rajasthan": "RJ",
    "Sikkim": "SK",
    "Tamil Nadu": "TN",
    "Telangana": "TS",
    "Tripura": "TR",
    "Uttar Pradesh": "UP",
    "Uttarakhand": "UK",
    "West Bengal": "WB"
}

# Define the post-processing function to correct misread characters
correction_dict = {
    'ICL': '1CL',  # Ensuring uppercase 'I' is replaced by '1'
    'ZG': '26',  # Ensuring uppercase 'I' is replaced by '1'
    'AG': '46',  # Ensuring uppercase 'I' is replaced by '1'
    'I2': '12',  # Ensuring uppercase 'I' is replaced by '1'
    'cQ': 'CQ',  # Ensuring uppercase 'I' is replaced by '1'
}

# Function to correct misreadings
def correct_text(text):
    for wrong, correct in correction_dict.items():
        text = text.replace(wrong, correct)
    return text

# Specify the folder containing the images and output CSV file
image_folder = './data/'
output_csv = 'ocr_results.csv'

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as csvfile:
    fieldnames = ['Filename', 'Detected Text', 'State', 'Confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # Write the header to the CSV file

    # Loop through each file in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter for image files
            image_path = os.path.join(image_folder, filename)
            
            # Load the image
            image = cv2.imread(image_path)
            image = imutils.resize(image, width=800)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use EasyOCR to detect and extract text
            results = reader.readtext(gray_image)
            
            # Loop through the results and process only text with more than 8 characters
            for (bbox, text, prob) in results:
                if len(text) > 8:  # Only process text with more than 8 characters
                    # Correct the detected text
                    corrected_text = correct_text(text)
                    
                    print(f"Detected text in {filename}: {corrected_text} (confidence: {prob:.2f})")
                    
                    # Get the first two characters for state code comparison
                    state_code = corrected_text[:2].upper()  # Make it uppercase to avoid case issues
                    
                    # Find the state name based on the state code
                    state = None
                    for key, value in state_codes.items():
                        if state_code == value:
                            state = key
                            break
                    
                    if not state:
                        state = "Unknown"  # If no state code match found
                    
                    # Write detected text, state, filename, and confidence to CSV
                    writer.writerow({'Filename': filename, 'Detected Text': corrected_text, 'State': state, 'Confidence': prob})
                    
                    # Draw a rectangle around the text region
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    top_left = tuple(map(int, top_left))
                    bottom_right = tuple(map(int, bottom_right))
                    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
                    
                    # Set up blue color for the state (above the number plate) and number plate text (below)
                    (text_x, text_y) = top_left
                    # Draw state text (on top, above number plate)
                    (state_width, state_height), _ = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    state_y_position = top_left[1] - state_height - 0  # Adjust position slightly to lower the text
                    
                    # Draw the blue background for state text
                    cv2.rectangle(image, (text_x, state_y_position - state_height - 5), 
                                  (text_x + state_width + 10, state_y_position + 5), 
                                  (255, 0, 0), -1)  # Blue background for state
                    
                    # Adjust text position within the blue background (center text horizontally)
                    state_text_x = text_x + 5  # Add a small margin
                    state_text_y = state_y_position + state_height - 20  # Center vertically
                    
                    # Place the state text in the blue background (white color for text)
                    cv2.putText(image, state, (state_text_x, state_text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # White text for state
                    
                    # Draw number plate text (below the state)
                    (text_width, text_height), _ = cv2.getTextSize(corrected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    text_background_y = bottom_right[1] + 10  # Position for background behind number plate text
                    
                    # Draw the blue background for number plate text
                    cv2.rectangle(image, (text_x, text_background_y), 
                                  (text_x + text_width + 10, text_background_y + text_height + 10), 
                                  (255, 0, 0), -1)  # Blue background for number plate
                    
                    # Add the number plate text in white
                    cv2.putText(image, corrected_text, (text_x + 5, text_background_y + text_height + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # White text for number plate
            
            # Display the image with the OCR results and state info highlighted
            cv2.imshow(f"OCR Result - {filename}", image)
            
            # Wait for 5 seconds (5000 milliseconds) and then close the window automatically
            cv2.waitKey(5000)  # 5000 milliseconds = 5 seconds
            cv2.destroyAllWindows()  # Close the window

print(f"OCR results with state info have been saved to {output_csv}.")
