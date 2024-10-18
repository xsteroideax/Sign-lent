import tkinter as tk
from PIL import Image, ImageTk
import mungming  # Import the mungming module
import main

def sign_to_word():
    def camera():
        main.camera()

    def video():
        import hea1

    def exit_app():
        root2.destroy()

    root2 = tk.Toplevel(root)  # Use Toplevel to create a child window
    root2.title("Sign to Word")
    root2.attributes("-fullscreen", True)  # Set the window to full-screen

    try:
        image2 = Image.open(r"C:\Users\Asus\OneDrive\kosen2\signlent\photo\okay.jpg")
        image2 = image2.resize((root2.winfo_screenwidth(), root2.winfo_screenheight()), Image.LANCZOS)
        photo2 = ImageTk.PhotoImage(image2)
    except Exception as e:
        print(f"Error loading image: {e}")
        root2.quit()

    # Create a label to display the image
    label2 = tk.Label(root2, image=photo2)
    label2.image = photo2  # Keep a reference to avoid garbage collection
    label2.pack()

    # Create buttons
    button_camera = tk.Button(root2, text="Camera", command=camera, width=85, height=7)
    button_camera.place(x=419, y=337)

    button_video = tk.Button(root2, text="Video", command=video, width=85, height=7)
    button_video.place(x=419, y=497)

    # Bind the Escape key to exit full-screen
    button_exit = tk.Button(root2, text="X", command=exit_app, width=3, height=1)
    button_exit.place(x=1406, y=1)
    root2.bind("<Escape>", lambda event: root2.destroy())

def word_to_sign():
    def exit_app():
        root3.destroy()
    # Create a new window for the "Word to Sign" feature
    root3 = tk.Toplevel(root)
    root3.title("Word to Sign")
    root3.attributes("-fullscreen", True)  # Set the window to full-screen

    # Get the screen width and height to calculate the center
    #screen_width = root3.winfo_screenwidth()
    #screen_height = root3.winfo_screenheight()

    # Entry widget for text input
    entry = tk.Entry(root3, font=("Arial", 20), width=30)
    entry.place(relx=0.5, rely=0.6, anchor='center')  # Place in the center

    # Label for output
    output_label = tk.Label(root3, text="", wraplength=600, font=("Arial", 20))
    output_label.place(relx=0.5, rely=0.5, anchor='center')  # Place in the center

    # Convert button
    button = tk.Button(root3, text="Convert", command=lambda: mungming.convert_to_sign(entry, output_label), font=("Arial", 20), width=20, height=2)
    button.place(relx=0.5, rely=0.7, anchor='center')  # Place in the center

    # Bind the Escape key to close the window
    button_exit = tk.Button(root3, text="X", command=exit_app, width=3, height=1)
    button_exit.place(x=1406, y=1)

    root3.bind("<Escape>", lambda event: root3.destroy())


def exit_app():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Sign Language Converter")
root.attributes("-fullscreen", True)

try:
    # Load an image using Pillow
    image = Image.open(r"C:\Users\Asus\OneDrive\kosen2\signlent\photo\again.jpg")
    image = image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
except Exception as e:
    print(f"Error loading image: {e}")
    root.quit()  # Exit if image cannot be loaded

# Create a label to display the image
label = tk.Label(root, image=photo)
label.image = photo  # Keep a reference to avoid garbage collection
label.pack()

# Create buttons
button_sign_to_word = tk.Button(root, text="Sign to Word", command=sign_to_word, width=95, height=5)
button_sign_to_word.place(x=387, y=302)

button_word_to_sign = tk.Button(root, text="Word to Sign", command=word_to_sign, width=95, height=5)
button_word_to_sign.place(x=387, y=419)

button_exit = tk.Button(root, text="Exit", command=exit_app, width=95, height=5)
button_exit.place(x=387, y=531)

# Bind the Escape key to exit full-screen
root.bind("<Escape>", lambda event: root.destroy())

# Start the Tkinter event loop
root.mainloop()
