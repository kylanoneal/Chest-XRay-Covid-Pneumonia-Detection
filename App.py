from Model import get_prediction
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

win = Tk()
win.geometry("500x500")

Label(win, text="Upload a Chest X-Ray Image", font='Arial 20 bold').pack(pady=15)


def open_img_file():
    for widget in frame.winfo_children():
        widget.destroy()

    filepath = filedialog.askopenfilename(title="Open an Image File")

    # Run image through trained model
    class_prediction = get_prediction(filepath)

    img = ImageTk.PhotoImage(Image.open(filepath).resize((150, 150)))
    img_label = Label(frame, image=img)
    img_label.pack()
    Label(frame, text=class_prediction, font='Arial 20 bold').pack()

    win.mainloop()


# Create a button to allow user to select file
button = Button(win, text="Select File", command=open_img_file)
button.pack()

# Create a frame to store the prediction
frame = Frame(win)
frame.pack(side="top", expand=True, fill="both")

win.mainloop()
