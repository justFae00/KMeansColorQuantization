import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from scipy.cluster.vq import kmeans, vq
from collections import deque

# Constants for appearance
BACKGROUND_COLOR = "#605CA8"
SIDE_BUTTONS_COLOR = "#E83E8C"
SIDE_BUTTONS_ON_HOVER = "#25153f"
TOP_BUTTONS_COLOR = "#00C0EF"
TOP_BUTTONS_ON_HOVER = "#25153f"
FONT = "Fixedsys"
FONT_SIZE = 14
FONT_COLOR = "white"
SIDE_BUTTONS_SIZE = 20

# Constants for side options
img_path = None
original_img = None
quantized_img = None
undo_stack = deque()
redo_stack = deque()
zoom_level = 1.0
start_x = 0
start_y = 0
img1_id = None
img2_id = None

# Icons locations
ZOOM_IN_IMG = './images/search-plus.png'
ZOOM_OUT_IMG = './images/search-minus.png'
PALETTE_IMG = './images/palette.png'
UNDO_IMG = './images/undo-alt.png'
REDO_IMG = './images/redo-alt.png'


def browse_image():
    global img_path, original_img, undo_stack, redo_stack, zoom_level, img1_id

    # Open a file dialog to select an image
    img_path = filedialog.askopenfilename()

    # If an image is selected
    if img_path:
        # Load the image using PIL
        original_img = Image.open(img_path)

        # Convert the image to a format compatible with Tkinter
        img_tk = ImageTk.PhotoImage(original_img)

        # Display the image on the canvas
        img1_id = canvas.create_image(canvas_width // 4, canvas_height // 2, anchor=tk.CENTER, image=img_tk)
        canvas.image1 = img_tk

        # Clear undo and redo stacks
        undo_stack.clear()
        redo_stack.clear()

        # Reset zoom level
        zoom_level = 1.0


def preprocess_image(img):
    # Convert the image to a NumPy array with float32 data type
    data = np.array(img, dtype=np.float32)

    # Get the dimensions of the image (height, width, depth)
    h, w, d = data.shape

    # Reshape the image to a 2D array of pixels (each pixel being a 3D vector of R, G, B values)
    return data.reshape((h * w, d)), h, w


def process_image():
    # Check if an image is loaded
    if not img_path:
        messagebox.showerror("Error", "No image loaded!")
        return

    # Get the number of colors (clusters) from the entry widget
    k = int(k_entry.get())

    # Convert the original image to RGB
    img = original_img.convert('RGB')

    # Preprocess the image to get the data and dimensions
    data, h, w = preprocess_image(img)

    # Perform K-Means clustering without using any built-in library
    centroids, labels = k_means_clustering(data, k)

    # Create a new image with the quantized colors
    new_data = centroids[labels].astype(np.uint8)

    global quantized_img, img2_id

    # Convert the quantized data back to an image
    quantized_img = Image.fromarray(new_data.reshape((h, w, 3)), 'RGB')

    # Convert the image to a format compatible with Tkinter
    img_tk = ImageTk.PhotoImage(quantized_img)

    # Display the quantized image on the canvas
    img2_id = canvas.create_image(3 * canvas_width // 4, canvas_height // 2, anchor=tk.CENTER, image=img_tk)
    canvas.image2 = img_tk

    # Add the quantized image to the undo stack
    undo_stack.append(quantized_img.copy())


def k_means_clustering(data, k, max_iters=100, tol=1e-4):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Initialize centroids by randomly selecting k points from the data
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Step 1: Assign clusters
        # Calculate distances between each point and each centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Step 2: Update centroids
        # Calculate the new centroids as the mean of the points assigned to each cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Step 3: Check for convergence
        # If centroids do not change significantly, break the loop
        if np.linalg.norm(centroids - new_centroids) < tol:
            break

        # Update centroids
        centroids = new_centroids

    return centroids, labels


# Zoom option
def update_zoom():
    global img1_id, img2_id

    zoomed_img_original = original_img.resize((int(original_img.width * zoom_level), int(original_img.height * zoom_level)), Image.LANCZOS)
    img_tk_original = ImageTk.PhotoImage(zoomed_img_original)
    canvas.delete(img1_id)
    img1_id = canvas.create_image(canvas_width // 4, canvas_height // 2, anchor=tk.CENTER, image=img_tk_original)
    canvas.image1 = img_tk_original

    zoomed_img_quantized = quantized_img.resize((int(quantized_img.width * zoom_level), int(quantized_img.height * zoom_level)), Image.LANCZOS)
    img_tk_quantized = ImageTk.PhotoImage(zoomed_img_quantized)
    canvas.delete(img2_id)
    img2_id = canvas.create_image(3 * canvas_width // 4, canvas_height // 2, anchor=tk.CENTER, image=img_tk_quantized)
    canvas.image2 = img_tk_quantized


def zoom_in():
    global zoom_level
    if quantized_img and original_img:
        zoom_level *= 1.2
        update_zoom()


def zoom_out():
    global zoom_level
    if quantized_img and original_img:
        zoom_level /= 1.2
        update_zoom()


def save_image():
    if not img_path or not quantized_img:
        messagebox.showerror("Error", "No image to save!")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if save_path:
        quantized_img.save(save_path)
        messagebox.showinfo("Save Image", f"Image saved successfully as {save_path}")


# Move around the images
def on_mouse_press(event):
    global start_x, start_y
    start_x = event.x
    start_y = event.y


def on_mouse_drag(event):
    global img1_id, img2_id, start_x, start_y
    dx = event.x - start_x
    dy = event.y - start_y

    if img1_id:
        canvas.move(img1_id, dx, dy)
    if img2_id:
        canvas.move(img2_id, dx, dy)

    start_x = event.x
    start_y = event.y


def extract_palette():
    if not quantized_img:
        messagebox.showerror("Error", "No image to extract palette from!")
        return
    palette = quantized_img.getcolors(quantized_img.width * quantized_img.height)
    colors = [color[1] for color in sorted(palette, reverse=True)]
    display_palette(colors)


def display_palette(colors):
    palette_window = tk.Toplevel(app)
    palette_window.title("Extracted Palette")
    palette_frame = tk.Frame(palette_window)
    palette_frame.pack()
    for color in colors:
        color_label = tk.Label(palette_frame, bg=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}', width=10, height=2)
        color_label.pack(side=tk.LEFT)


def undo():
    if undo_stack:
        redo_stack.append(undo_stack.pop())
        if undo_stack:
            global quantized_img
            quantized_img = undo_stack[-1]
            update_zoom()


def redo():
    if redo_stack:
        undo_stack.append(redo_stack.pop())
        global quantized_img
        quantized_img = undo_stack[-1]
        update_zoom()


def hover_effects(widget, original_color, hover_color):
    widget.bind("<Enter>", lambda e: e.widget.config(bg=hover_color, cursor="hand2"))
    widget.bind("<Leave>", lambda e: e.widget.config(bg=original_color, cursor=""))


app = tk.Tk()
app.title("Color Quantization App")
app.configure(bg=BACKGROUND_COLOR)


# Load the images
zoom_in_image = tk.PhotoImage(file = ZOOM_IN_IMG)
zoom_out_image = tk.PhotoImage(file = ZOOM_OUT_IMG)
palette_image = tk.PhotoImage(file = PALETTE_IMG)
undo_image = tk.PhotoImage(file = UNDO_IMG)
redo_image = tk.PhotoImage(file = REDO_IMG)


# Control Frame for top buttons and labels
control_frame = tk.Frame(app, bg=BACKGROUND_COLOR)
control_frame.grid(row=0, column=0, columnspan=3, pady=20)

load_button = tk.Button(control_frame, text="Browse Image", command=browse_image, bg=TOP_BUTTONS_COLOR, bd=2, relief="solid", font=(FONT, FONT_SIZE), fg=FONT_COLOR)
load_button.grid(row=0, column=0, padx=5)
hover_effects(load_button, TOP_BUTTONS_COLOR, TOP_BUTTONS_ON_HOVER)

k_label = tk.Label(control_frame, text="Number of Colors:", bg=BACKGROUND_COLOR, font=(FONT, FONT_SIZE), fg=FONT_COLOR)
k_label.grid(row=0, column=1, padx=5)

k_entry = tk.Entry(control_frame, font=(FONT, 18), width=5, bd=2, relief="solid",)
k_entry.grid(row=0, column=2, padx=5)

process_button = tk.Button(control_frame, text="Process", command=process_image, bg=TOP_BUTTONS_COLOR, bd=2, relief="solid", font=(FONT, FONT_SIZE), fg=FONT_COLOR)
process_button.grid(row=0, column=3, padx=5)
hover_effects(process_button, TOP_BUTTONS_COLOR, TOP_BUTTONS_ON_HOVER)


# Canvas (where images are loaded)
canvas_width = 900
canvas_height = 500
canvas = tk.Canvas(app, width=canvas_width, height=canvas_height, bg=BACKGROUND_COLOR, bd=0, highlightthickness=0)
canvas.grid(row=1, column=0, columnspan=2)
canvas.bind("<ButtonPress-1>", on_mouse_press)
canvas.bind("<B1-Motion>", on_mouse_drag)

save_button = tk.Button(app, text="Save Image", command=save_image, bg=TOP_BUTTONS_COLOR, bd=2, relief="solid", font=(FONT, FONT_SIZE), fg=FONT_COLOR)
save_button.grid(row=2, column=0, columnspan=3, pady=20)
hover_effects(save_button, TOP_BUTTONS_COLOR, TOP_BUTTONS_ON_HOVER)


# Side panel: for extra options
side_panel = tk.Frame(app, bg=BACKGROUND_COLOR, padx=20)
side_panel.grid(row=1, column=2, sticky="n")

zoom_in_button = tk.Button(side_panel, image=zoom_in_image, command=zoom_in, bg=SIDE_BUTTONS_COLOR, bd=2, relief="solid",)
zoom_in_button.grid(row=0, column=0, pady=5)
hover_effects(zoom_in_button, SIDE_BUTTONS_COLOR, SIDE_BUTTONS_ON_HOVER)

zoom_out_button = tk.Button(side_panel, image=zoom_out_image, command=zoom_out, bg=SIDE_BUTTONS_COLOR, bd=2, relief="solid",)
zoom_out_button.grid(row=1, column=0, pady=5)
hover_effects(zoom_out_button, SIDE_BUTTONS_COLOR, SIDE_BUTTONS_ON_HOVER)

palette_button = tk.Button(side_panel, image=palette_image, command=extract_palette, bg=SIDE_BUTTONS_COLOR, bd=2, relief="solid",)
palette_button.grid(row=2, column=0, pady=5)
hover_effects(palette_button, SIDE_BUTTONS_COLOR, SIDE_BUTTONS_ON_HOVER)

undo_button = tk.Button(side_panel, image=undo_image, command=undo, bg=SIDE_BUTTONS_COLOR, bd=2, relief="solid",)
undo_button.grid(row=3, column=0, pady=5)
hover_effects(undo_button, SIDE_BUTTONS_COLOR, SIDE_BUTTONS_ON_HOVER)

redo_button = tk.Button(side_panel, image=redo_image, command=redo, bg=SIDE_BUTTONS_COLOR, bd=2, relief="solid",)
redo_button.grid(row=4, column=0, pady=5)
hover_effects(redo_button, SIDE_BUTTONS_COLOR, SIDE_BUTTONS_ON_HOVER)


# Run app
app.mainloop()