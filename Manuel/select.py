import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk


class ImageClassifierApp:
    def __init__(self, root, input_folder):
        self.root = root
        self.root.title("Image Classifier")

        self.input_folder = input_folder
        self.image_files = [
            f for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        self.current_image_index = 0
        self.points = []  # Stores data points for plotting

        # Main frame for layout
        self.main_frame = Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        # Image display
        self.image_label = Label(self.main_frame)
        self.image_label.pack()

        # Info display
        self.info_label = Label(self.main_frame, text="", font=("Arial", 14))
        self.info_label.pack()

        # Button frame
        self.button_frame = Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        self.yes_button = Button(self.button_frame, text="Ja", command=lambda: self.classify_image("good"), bg="green",
                                 fg="white", font=("Arial", 14))
        self.yes_button.pack(side="left", padx=20)

        self.no_button = Button(self.button_frame, text="Nein", command=lambda: self.classify_image("not good"),
                                bg="red", fg="white", font=("Arial", 14))
        self.no_button.pack(side="right", padx=20)

        # Show the first image
        self.show_image()

    def compute_features(self, image_path):
        """
        Computes edge density and color variance of an image.
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density = np.sum(edges) / edges.size
        color_variance = np.mean(np.var(img, axis=(0, 1)))
        return color_variance, edge_density

    def resize_image(self, image, max_width=800, max_height=600):
        """
        Resizes an image to fit within the specified max dimensions, maintaining aspect ratio.
        """
        width, height = image.size
        if width > max_width or height > max_height:
            scale_width = max_width / width
            scale_height = max_height / height
            scale = min(scale_width, scale_height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return image.resize((new_width, new_height), Image.LANCZOS)
        return image

    def show_image(self):
        """
        Displays the current image in the GUI.
        """
        if self.current_image_index < len(self.image_files):
            image_path = os.path.join(self.input_folder, self.image_files[self.current_image_index])
            pil_image = Image.open(image_path)
            resized_image = self.resize_image(pil_image)
            tk_image = ImageTk.PhotoImage(resized_image)

            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            color_variance, edge_density = self.compute_features(image_path)
            self.current_color_variance = color_variance
            self.current_edge_density = edge_density
            self.info_label.config(
                text=f"Bild {self.current_image_index + 1}/{len(self.image_files)}\nFarbvarianz: {color_variance:.2f}, Kantendichte: {edge_density:.4f}")
        else:
            self.image_label.config(image=None)
            self.image_label.image = None
            self.info_label.config(text="Alle Bilder klassifiziert!")
            self.yes_button.config(state="disabled")
            self.no_button.config(state="disabled")
            self.plot_results()

    def classify_image(self, label):
        """
        Saves the classification result and moves to the next image.
        """
        if self.current_image_index < len(self.image_files):
            self.points.append({
                "x": self.current_color_variance,
                "y": self.current_edge_density,
                "label": label
            })
            self.current_image_index += 1
            self.show_image()

    def plot_results(self):
        """
        Plots the classification results.
        """
        good_points = [point for point in self.points if point["label"] == "good"]
        not_good_points = [point for point in self.points if point["label"] == "not good"]

        x_good = [p["x"] for p in good_points]
        y_good = [p["y"] for p in good_points]
        x_not_good = [p["x"] for p in not_good_points]
        y_not_good = [p["y"] for p in not_good_points]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_good, y_good, color="green", label="Gut", alpha=0.7)
        plt.scatter(x_not_good, y_not_good, color="red", label="Nicht gut", alpha=0.7)
        plt.xlabel("Farbvarianz")
        plt.ylabel("Kantendichte")
        plt.title("Klassifikationsergebnisse")
        plt.legend()
        plt.grid(True)
        plt.show()


def select_folder_and_run():
    """
    Allows the user to select a folder and starts the classification app.
    """
    folder = filedialog.askdirectory(title="Wähle einen Ordner mit Bildern")
    if folder:
        root = Tk()
        app = ImageClassifierApp(root, folder)
        root.mainloop()


if __name__ == "__main__":
    select_folder_and_run()
