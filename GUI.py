import tkinter as tk
from tkinter import messagebox
from tkinter import *
from tkinter import filedialog
window=tk.Tk()
window.title("Sudoku Solver With AI ")
canvas = Canvas(window, width = 650, height = 400)
# canvas.pack()


img = PhotoImage(file="logo.png")
canvas.create_image(45,0, anchor=NW, image=img)
canvas.grid(column=0,row=0)
l1=tk.Label(window,text="Enter the camera ip address ( if any )",font=("Algerian",14))
l1.place(x=10,y=320)
t1=tk.Entry(window,width=50,bd=5)
t1.place(x=400,y=320)



# Function for opening the file
def video_opener():
    filename = filedialog.askopenfilename(filetypes=(("All files", "*.*")
                                                     # ,("Template files", "*.tplate")
                                                       , ("HTML files", "*.html;*.htm")))
    print(filename)

def image_opener():
    filename = filedialog.askopenfilename(filetypes=(("All files", "*.*")
                                                     # ,("Template files", "*.tplate")
                                                       , ("HTML files", "*.html;*.htm")))
    print(filename)

def Gui_webcam():
    if(t1.get()==""):
        path=0
    else:
        path=str(t1.get())
    # solve_webcam(path,True)
    print(type(path))
    print(path)


b1=tk.Button(window,text="Solve Image",font=("Algerian",20),bg='tan1',fg='black', command = lambda:image_opener())
# b1.grid(column=0,row=1)
b1.place(x=15,y=400)
b2=tk.Button(window,text="Webcam Acess",font=("Algerian",20),bg='tan1',fg='black',command= lambda:Gui_webcam())
b2.place(x=230,y=400)
b3=tk.Button(window,text="Using Video",font=("Algerian",20),bg='tan1',fg='black', command = lambda:video_opener())
b3.place(x=470,y=400)

window.geometry("700x500")
window.mainloop()