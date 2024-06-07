# Import necessary libraries

import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from PIL import Image

# The fitz library, also known as PyMuPDF, is used for working with PDF documents.
# The cv2 library, also known as OpenCV, is used for image processing tasks such as converting images and finding contours.
# The numpy library is used for array operations and data manipulation.
# The os library is used for interacting with the file system, such as creating directories and joining paths.
# The PIL (Pillow) library is used for saving images in different formats and manipulating them.

# Function to preprocess a specific page from a PDF file and return the image.
def preprocess_page(pdf_path, page_number):
    # Open the PDF document using the fitz library.
    pdf_document = fitz.open(pdf_path)
    
    # Retrieve the specified page from the PDF document.
    page = pdf_document[page_number]
    
    # Convert the page into a pixmap (an image representation) in RGB format.
    pix = page.get_pixmap(matrix="RGB")
    
    # Convert the pixmap samples into a numpy array and reshape it into an image.
    # The array is reshaped into the dimensions (height, width, 3) for RGB channels.
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
    
    # Return the preprocessed image.
    return img

# Function to extract figures from a specific page of a PDF file and save them as images.
def extract_figures_from_page(pdf_path, page_number, output_folder, use_grayscale_threshold=True):
    # Preprocess the specified page to obtain the image.
    img = preprocess_page(pdf_path, page_number)
    
    # Initialize an empty list to store contours found in the image.
    contours = []

    # If using grayscale thresholding for figure detection:
    if use_grayscale_threshold:
        # Convert the image from RGB to grayscale format.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply binary inverse thresholding to the grayscale image to create a binary mask.
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find external contours in the binary mask.
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Convert the image from RGB to HSV format.
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define lower and upper bounds for HSV masking.
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([179, 255, 255])
        
        # Create a mask using the defined bounds.
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        
        # Find external contours in the masked image.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a minimum area threshold to filter out small contours.
    min_area = 500
    
    # Filter contours based on the minimum area threshold.
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Iterate through the filtered contours to extract and save figures.
    for idx, cnt in enumerate(filtered_contours):
        # Obtain the bounding rectangle for the contour.
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Define an expansion factor to expand the crop region slightly.
        expansion_factor = 0.1
        
        # Calculate expanded crop coordinates with the expansion factor.
        x1 = max(0, x - int(expansion_factor * w))
        y1 = max(0, y - int(expansion_factor * h))
        x2 = min(img.shape[1], x + w + int(expansion_factor * w))
        y2 = min(img.shape[0], y + h + int(expansion_factor * h))
        
        # Crop the figure from the image using the calculated coordinates.
        cropped_figure = img[y1:y2, x1:x2]
        
        # Generate the output filename for the cropped figure.
        output_filename = f"page{page_number + 1}_figure{idx + 1}.png"
        
        # Combine the output folder path with the output filename to create the full path.
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the cropped figure as an image file using PIL.
        Image.fromarray(cropped_figure).save(output_path, quality=100)

# Function to extract figures from an entire PDF file.
def extract_figures_from_pdf(pdf_path, output_folder):
    # Open the PDF document using fitz.
    pdf_document = fitz.open(pdf_path)
    
    # Iterate through each page of the PDF document.
    for page_number in range(len(pdf_document)):
        # Extract figures from the current page and save them to the output folder.
        extract_figures_from_page(pdf_path, page_number, output_folder)
    
    # Close the PDF document after processing all pages.
    pdf_document.close()

# Define the path to the PDF file to be processed.
pdf_file_path = r"C:\Users\lkkpk\Desktop\Final Project\Chapter 1.pdf"  # Update this path if necessary.

# Define the output folder path to save the extracted figures.
output_folder = "Output_Final"  # Folder where extracted figures will be saved.

# If the output folder does not exist, create it.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Call the function to extract figures from the PDF file and save them to the specified output folder.
extract_figures_from_pdf(pdf_file_path, output_folder)

# Print a message indicating the successful extraction of figures.
print("Figures extracted successfully.")
