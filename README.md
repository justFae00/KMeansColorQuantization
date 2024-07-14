# K-Means Clustering Color Quantization App with GUI

The Color Quantization App is a Python-based application that reduces the number of distinct colors in an image while preserving its visual appearance. This technique is useful for image compression, reducing file sizes, and improving the efficiency of image processing algorithms. The app provides a user-friendly interface to load images, perform color quantization using K-Means clustering, and visualize the results.





https://github.com/user-attachments/assets/28462657-ca1f-451f-ad21-48a235a86530

### Functionality 
`preprocess_image`: Converts the loaded image into a NumPy array and reshapes it for K-Means clustering. Then returns the reshaped image data along with its dimensions.

`process_image`: Performs color quantization on the loaded image using K-Means clustering and displays the quantized image on the canvas. Also handles error checking for image loading and retrieves the number of colors from the user input.

`k_means_clustering`: The k_means_clustering function applies the K-Means algorithm to partition image data into k clusters. It initializes centroids randomly, assigns each pixel to the nearest centroid, updates the centroids based on the mean of assigned pixels, and repeats until convergence or a maximum number of iterations. The function returns the final centroids and cluster labels, which represent the reduced color palette and pixel assignments for the quantized image.


### Libraries Used
- PIL (Python Imaging Library)
- NumPy
- Matplotlib
- Tkinter

### UI Features
-	Zoom and Pan
-	Undo/Redo Functionality
-	Extract Palette
-	Interactive Buttons
